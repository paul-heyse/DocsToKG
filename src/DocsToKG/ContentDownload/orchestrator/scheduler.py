# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.scheduler",
#   "purpose": "Orchestrator dispatcher with worker pool, heartbeat, and bounded concurrency",
#   "sections": [
#     {"id": "orchestrator", "name": "Orchestrator", "anchor": "#class-orchestrator", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Orchestrator dispatcher and worker pool management.

This module provides the Orchestrator class that:
- Manages a bounded pool of worker threads
- Implements a dispatcher loop for leasing jobs
- Handles heartbeat extension for active workers
- Coordinates graceful shutdown
- Emits OTel metrics for observability

**Architecture:**

    Orchestrator (main)
      ├─ Dispatcher Loop: Leases jobs to worker queue
      ├─ Heartbeat Loop: Extends leases for active workers
      ├─ Worker Threads: Process jobs from queue
      └─ Metrics: Queue depth, throughput, job timing

**Usage:**

    from DocsToKG.ContentDownload.orchestrator import Orchestrator
    from DocsToKG.ContentDownload.orchestrator.config import OrchestratorConfig

    config = OrchestratorConfig(max_workers=8, max_per_host=4)
    orch = Orchestrator(config, queue, pipeline, telemetry)

    # Start dispatcher, heartbeat, and workers
    orch.start()

    # Monitor queue depth
    stats = queue.stats()

    # Graceful shutdown
    orch.stop()

**Feature Flags:**

- `DOCSTOKG_ENABLE_HEARTBEAT_SYNC`: Config-aware heartbeat TTL (default: true)
- `DOCSTOKG_ENABLE_JOB_BATCHING`: Batch lease requests (default: true)
- `DOCSTOKG_JOB_BATCH_SIZE`: Max jobs per batch (default: 10)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Optional

from . import feature_flags

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.orchestrator.queue import WorkQueue
    from DocsToKG.ContentDownload.orchestrator.workers import Worker
    from DocsToKG.ContentDownload.pipeline import ResolverPipeline
    from DocsToKG.ContentDownload.telemetry import RunTelemetry

__all__ = ["Orchestrator", "OrchestratorConfig"]

logger = logging.getLogger(__name__)


class OrchestratorConfig:
    """Configuration for Orchestrator."""

    def __init__(
        self,
        max_workers: int = 8,
        max_per_resolver: Optional[dict[str, int]] = None,
        max_per_host: int = 4,
        lease_ttl_seconds: int = 600,
        heartbeat_seconds: int = 30,
        max_job_attempts: int = 3,
        retry_backoff_seconds: int = 60,
        jitter_seconds: int = 15,
    ) -> None:
        """Initialize orchestrator configuration.

        Args:
            max_workers: Global worker concurrency limit
            max_per_resolver: Per-resolver concurrency overrides
            max_per_host: Per-host concurrency limit
            lease_ttl_seconds: Job lease duration
            heartbeat_seconds: Heartbeat interval
            max_job_attempts: Max attempts before error
            retry_backoff_seconds: Base retry delay
            jitter_seconds: Retry jitter range
        """
        self.max_workers = max_workers
        self.max_per_resolver = max_per_resolver or {}
        self.max_per_host = max_per_host
        self.lease_ttl_seconds = lease_ttl_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.max_job_attempts = max_job_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.jitter_seconds = jitter_seconds


class Orchestrator:
    """Worker pool orchestrator with dispatcher and heartbeat.

    Manages:
    - Worker thread pool for job execution
    - Dispatcher loop for job leasing
    - Heartbeat for lease extension
    - Graceful shutdown

    Attributes:
        config: OrchestratorConfig with tuning parameters
        queue: WorkQueue for job management
        pipeline: ResolverPipeline for job execution
        telemetry: Optional RunTelemetry for metrics
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        queue: "WorkQueue",
        pipeline: "ResolverPipeline",
        telemetry: Optional["RunTelemetry"] = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            config: OrchestratorConfig
            queue: WorkQueue instance
            pipeline: ResolverPipeline instance
            telemetry: Optional telemetry sink
        """
        self.config = config
        self.queue = queue
        self.pipeline = pipeline
        self.telemetry = telemetry
        self.worker_id = f"orch-{uuid.uuid4()}"

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._jobs_queue: Queue[dict[str, Any]] = Queue(maxsize=config.max_workers * 2)

        # Import here to avoid circular deps
        from DocsToKG.ContentDownload.orchestrator.limits import KeyedLimiter
        from DocsToKG.ContentDownload.orchestrator.workers import Worker

        self._KeyedLimiter = KeyedLimiter
        self._Worker = Worker

        self._resolver_limiter = KeyedLimiter(
            default_limit=config.max_workers,
            per_key=config.max_per_resolver,
        )
        self._host_limiter = KeyedLimiter(default_limit=config.max_per_host)

        logger.info(f"Orchestrator initialized: {self.worker_id}")

    def start(self) -> None:
        """Start dispatcher, heartbeat, and worker threads."""
        logger.info(f"Starting orchestrator with {self.config.max_workers} workers")

        # Start worker threads
        for i in range(self.config.max_workers):
            worker = self._Worker(
                worker_id=f"{self.worker_id}-worker-{i}",
                queue=self.queue,
                pipeline=self.pipeline,
                resolver_limiter=self._resolver_limiter,
                host_limiter=self._host_limiter,
                heartbeat_sec=self.config.heartbeat_seconds,
                max_job_attempts=self.config.max_job_attempts,
                retry_backoff=self.config.retry_backoff_seconds,
                jitter=self.config.jitter_seconds,
            )
            t = threading.Thread(
                target=self._worker_loop,
                args=(worker,),
                daemon=True,
                name=f"worker-{i}",
            )
            t.start()
            self._threads.append(t)

        # Start dispatcher
        dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True,
            name="dispatcher",
        )
        dispatcher_thread.start()
        self._threads.append(dispatcher_thread)

        # Start heartbeat
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="heartbeat",
        )
        heartbeat_thread.start()
        self._threads.append(heartbeat_thread)

        logger.info(f"Orchestrator started: {len(self._threads)} threads")

    def stop(self) -> None:
        """Signal stop and wait for threads."""
        logger.info(f"Orchestrator stopping: {self.worker_id}")
        self._stop.set()

        # Join all threads with timeout
        for t in self._threads:
            t.join(timeout=5)

        logger.info("Orchestrator stopped")

    def _dispatcher_loop(self) -> None:
        """Lease jobs and feed worker queue.

        Implements optional job batching to reduce database contention.
        Can be disabled via DOCSTOKG_ENABLE_JOB_BATCHING feature flag.
        """
        logger.debug("Dispatcher loop started")
        batching_enabled = feature_flags.is_enabled("job_batching")

        while not self._stop.is_set():
            try:
                free_slots = self._jobs_queue.maxsize - self._jobs_queue.qsize()

                if free_slots <= 0:
                    batch_size = 0
                elif batching_enabled:
                    max_batch = feature_flags.get_batch_size()  # default 10
                    batch_size = min(max_batch, free_slots)
                else:
                    # No batching: lease 1 job at a time
                    batch_size = 1

                if batch_size > 0:
                    # Lease jobs (batched or single)
                    jobs = self.queue.lease(
                        self.worker_id,
                        limit=batch_size,
                        lease_ttl_sec=self.config.lease_ttl_seconds,
                    )

                    for job in jobs:
                        self._jobs_queue.put(job, block=False)

                    if jobs:
                        if batching_enabled:
                            logger.debug(f"Leased {len(jobs)} jobs (batch_size={batch_size})")
                        else:
                            logger.debug(f"Leased {len(jobs)} jobs (batching disabled)")
                    else:
                        # Backoff if no jobs available
                        time.sleep(0.05)
                else:
                    # No free slots, backoff
                    time.sleep(0.1)

                # Emit metrics
                stats = self.queue.stats()
                logger.debug(f"Queue stats: {stats}")

                # Sleep before next lease attempt
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Dispatcher error: {e}", exc_info=True)
                time.sleep(1.0)

        logger.debug("Dispatcher loop stopped")

    def _heartbeat_loop(self) -> None:
        """Extend leases for active workers.

        Sends periodic heartbeats to extend job leases, preventing
        crashed workers' jobs from being reclaimed. Can be disabled
        via DOCSTOKG_ENABLE_HEARTBEAT_SYNC feature flag.
        """
        logger.debug("Heartbeat loop started")
        heartbeat_sync_enabled = feature_flags.is_enabled("heartbeat_sync")

        while not self._stop.is_set():
            try:
                if heartbeat_sync_enabled:
                    # Pass lease_ttl_seconds for config-aware heartbeat (OPTIMIZATION #2)
                    self.queue.heartbeat(self.worker_id, self.config.lease_ttl_seconds)
                    logger.debug(f"Heartbeat sent with TTL sync for {self.worker_id}")
                else:
                    # Fallback: minimal heartbeat with hardcoded extension
                    self.queue.heartbeat(self.worker_id)
                    logger.debug(f"Heartbeat sent (sync disabled) for {self.worker_id}")

                time.sleep(self.config.heartbeat_seconds)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)
                time.sleep(1.0)

        logger.debug("Heartbeat loop stopped")

    def _worker_loop(self, worker: "Worker") -> None:
        """Worker thread execution loop."""
        logger.debug(f"Worker loop started: {worker.worker_id}")

        while not self._stop.is_set():
            try:
                # Get job with timeout
                job = self._jobs_queue.get(timeout=1.0)

                if job is None:
                    break

                # Process job
                worker.run_one(job)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

        logger.debug(f"Worker loop stopped: {worker.worker_id}")
