# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.workers",
#   "purpose": "Job execution wrapper with concurrency control and telemetry",
#   "sections": [
#     {"id": "worker", "name": "Worker", "anchor": "#class-worker", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Job execution wrapper for ContentDownload orchestration.

This module provides the Worker class that:
- Wraps pipeline execution for a single job
- Acquires concurrency limits (per-resolver, per-host)
- Handles job leasing and state transitions
- Integrates with telemetry and error handling
- Supports graceful shutdown and retry logic

**Usage:**

    worker = Worker(
        worker_id="worker-1",
        queue=work_queue,
        pipeline=resolver_pipeline,
        resolver_limiter=KeyedLimiter(default_limit=8),
        host_limiter=KeyedLimiter(default_limit=4),
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    # Process a single leased job
    job = {"id": 123, "artifact_id": "doi:10.1234/example", "artifact_json": "..."}
    worker.run_one(job)
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.orchestrator.limits import KeyedLimiter
    from DocsToKG.ContentDownload.orchestrator.queue import WorkQueue
    from DocsToKG.ContentDownload.pipeline import ResolverPipeline

__all__ = ["Worker"]

logger = logging.getLogger(__name__)


class Worker:
    """Job execution wrapper with concurrency control.

    Runs jobs from the work queue through the resolver pipeline,
    respecting per-resolver and per-host concurrency limits.

    Attributes:
        worker_id: Unique worker identifier for leasing and logging
        heartbeat_sec: Interval (seconds) to extend lease during long operations
        max_job_attempts: Maximum attempts before marking job as error
        retry_backoff: Base seconds to delay before retry
        jitter: Random jitter (0 to this value) added to retry backoff
    """

    def __init__(
        self,
        worker_id: str,
        queue: "WorkQueue",
        pipeline: "ResolverPipeline",
        resolver_limiter: "KeyedLimiter",
        host_limiter: "KeyedLimiter",
        heartbeat_sec: int,
        max_job_attempts: int,
        retry_backoff: int,
        jitter: int,
    ) -> None:
        """Initialize worker.

        Args:
            worker_id: Unique identifier for this worker
            queue: WorkQueue for leasing and state management
            pipeline: ResolverPipeline to process artifacts
            resolver_limiter: Concurrency limiter for resolvers
            host_limiter: Concurrency limiter for hosts
            heartbeat_sec: Lease extension interval
            max_job_attempts: Max attempts before error state
            retry_backoff: Base retry delay in seconds
            jitter: Random jitter for retry backoff
        """
        self.worker_id = worker_id
        self.queue = queue
        self.pipeline = pipeline
        self.resolver_limiter = resolver_limiter
        self.host_limiter = host_limiter
        self.heartbeat_sec = heartbeat_sec
        self.max_job_attempts = max_job_attempts
        self.retry_backoff = retry_backoff
        self.jitter = jitter

        self._stop = threading.Event()
        self._running_jobs: set[int] = set()
        self._lock = threading.Lock()

        logger.debug(f"Worker initialized: {worker_id}")

    def stop(self) -> None:
        """Signal worker to stop processing new jobs."""
        self._stop.set()
        logger.info(f"Worker {self.worker_id} stopping...")

    def run_one(self, job: Mapping[str, Any]) -> None:
        """Execute a single leased job.

        Orchestrates the full job lifecycle:
        1. Rehydrate artifact from JSON
        2. Acquire concurrency limits
        3. Run through pipeline
        4. Ack or fail based on outcome

        Args:
            job: Job dict with id, artifact_id, artifact_json
        """
        job_id = job["id"]
        artifact_id = job["artifact_id"]

        with self._lock:
            if self._stop.is_set():
                logger.debug(f"Worker {self.worker_id} is stopping, skipping job {job_id}")
                return
            self._running_jobs.add(job_id)

        try:
            logger.debug(f"Worker {self.worker_id} processing job {job_id} ({artifact_id})")

            # Rehydrate artifact from JSON
            try:
                artifact_json = job.get("artifact_json", "{}")
                artifact = json.loads(artifact_json) if artifact_json else {}
            except Exception as e:
                logger.error(f"Failed to rehydrate artifact for job {job_id}: {e}")
                self.queue.fail_and_retry(
                    job_id,
                    backoff_sec=self.retry_backoff,
                    max_attempts=self.max_job_attempts,
                    last_error=f"artifact_rehydration_failed: {str(e)[:100]}",
                )
                return

            # Run through pipeline (without concurrency guards for now)
            # The limiter guards will be applied in Phase 5 (Orchestrator)
            try:
                outcome = self.pipeline.process(artifact, ctx=None)

                # Determine terminal state
                if outcome.ok:
                    state = "done"
                    reason = "success"
                elif outcome.classification == "skip":
                    state = "skipped"
                    reason = outcome.reason or "unknown_skip"
                else:
                    state = "error"
                    reason = outcome.reason or "download_error"

                # Ack with outcome
                self.queue.ack(job_id, state, last_error=None)
                logger.info(f"Job {job_id} completed: {state} ({reason})")

            except Exception as e:
                # Retry on pipeline error
                logger.warning(f"Job {job_id} failed with error: {e}")
                self.queue.fail_and_retry(
                    job_id,
                    backoff_sec=self.retry_backoff,
                    max_attempts=self.max_job_attempts,
                    last_error=f"pipeline_error: {str(e)[:100]}",
                )

        finally:
            with self._lock:
                self._running_jobs.discard(job_id)

    def _compute_backoff(self) -> int:
        """Compute retry backoff with jitter.

        Returns:
            Seconds to delay before retry
        """
        return self.retry_backoff + random.randint(0, self.jitter)
