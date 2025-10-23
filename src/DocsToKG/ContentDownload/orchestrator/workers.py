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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from DocsToKG.ContentDownload.orchestrator.limits import KeyedLimiter
    from DocsToKG.ContentDownload.orchestrator.queue import WorkQueue
    from DocsToKG.ContentDownload.pipeline import ResolverPipeline

from DocsToKG.ContentDownload.core import DownloadContext, WorkArtifact
from DocsToKG.ContentDownload.orchestrator.limits import host_key

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

            try:
                artifact, ctx = self._deserialize_job(job)
            except _ArtifactValidationError as exc:
                logger.error(
                    f"Failed to deserialize artifact for job {job_id}: {exc}",
                )
                self.queue.fail_and_retry(
                    job_id,
                    backoff_sec=self.retry_backoff,
                    max_attempts=self.max_job_attempts,
                    last_error=f"artifact_validation_failed: {str(exc)[:100]}",
                )
                return
            except Exception as e:  # Defensive: unexpected validation failure
                logger.exception("Unexpected error while deserializing artifact for job %s", job_id)
                self.queue.fail_and_retry(
                    job_id,
                    backoff_sec=self.retry_backoff,
                    max_attempts=self.max_job_attempts,
                    last_error=f"artifact_rehydration_failed: {str(e)[:100]}",
                )
                return

            limiter_tokens: list[tuple["KeyedLimiter", str]] = []
            try:
                limiter_tokens = self._acquire_limiter_slots(job, artifact)
            except Exception as e:
                logger.error(
                    "Failed to acquire limiter slots for job %s (%s): %s",
                    job_id,
                    artifact_id,
                    e,
                )
                self.queue.fail_and_retry(
                    job_id,
                    backoff_sec=self.retry_backoff,
                    max_attempts=self.max_job_attempts,
                    last_error=f"limiter_acquire_failed: {str(e)[:100]}",
                )
                return

            try:
                outcome = self.pipeline.run(artifact, ctx)

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
                self._release_limiter_slots(limiter_tokens)

        finally:
            with self._lock:
                self._running_jobs.discard(job_id)

    def _compute_backoff(self) -> int:
        """Compute retry backoff with jitter.

        Returns:
            Seconds to delay before retry
        """
        return self.retry_backoff + random.randint(0, self.jitter)

    def _acquire_limiter_slots(
        self, job: Mapping[str, Any], artifact: Any
    ) -> list[tuple["KeyedLimiter", str]]:
        """Acquire resolver and host limiter slots for this job.

        Returns a list of (limiter, key) tuples representing acquired slots.
        Slots are released in LIFO order via :meth:`_release_limiter_slots`.
        """

        tokens: list[tuple["KeyedLimiter", str]] = []

        resolver_key = self._extract_resolver_key(job, artifact)
        host_key_value = self._extract_host_key(job, artifact)

        try:
            if resolver_key:
                logger.debug(
                    "Worker %s acquiring resolver limiter for key=%s",
                    self.worker_id,
                    resolver_key,
                )
                self.resolver_limiter.acquire(resolver_key)
                tokens.append((self.resolver_limiter, resolver_key))

            if host_key_value:
                logger.debug(
                    "Worker %s acquiring host limiter for key=%s",
                    self.worker_id,
                    host_key_value,
                )
                self.host_limiter.acquire(host_key_value)
                tokens.append((self.host_limiter, host_key_value))
        except Exception:
            # Release any already-acquired slots before propagating
            self._release_limiter_slots(tokens)
            raise

        return tokens

    def _release_limiter_slots(self, tokens: list[tuple["KeyedLimiter", str]]) -> None:
        """Release limiter slots acquired for this job."""

        while tokens:
            limiter, key = tokens.pop()
            try:
                limiter.release(key)
                logger.debug(
                    "Worker %s released limiter for key=%s",
                    self.worker_id,
                    key,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Worker %s failed releasing limiter key=%s: %s",
                    self.worker_id,
                    key,
                    exc,
                )

    def _extract_resolver_key(self, job: Mapping[str, Any], artifact: Any) -> Optional[str]:
        """Determine resolver limiter key for the job if available."""

        candidates: list[Mapping[str, Any]] = [job]
        if isinstance(artifact, Mapping):
            candidates.append(artifact)

        resolver_fields = ("resolver_hint", "resolver", "resolver_name")

        for candidate in candidates:
            for field in resolver_fields:
                value = candidate.get(field) if isinstance(candidate, Mapping) else None
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _extract_host_key(self, job: Mapping[str, Any], artifact: Any) -> Optional[str]:
        """Determine host limiter key for the job if available."""

        for mapping in (job, artifact if isinstance(artifact, Mapping) else None):
            if not isinstance(mapping, Mapping):
                continue

            url = self._first_url(mapping)
            if url:
                return host_key(url)

        return None

    def _first_url(self, payload: Mapping[str, Any]) -> Optional[str]:
        """Return the first URL candidate found in payload."""

        url_fields = (
            "url",
            "source_url",
            "origin_url",
            "download_url",
            "landing_url",
            "pdf_url",
        )

        list_fields = ("urls", "landing_urls", "pdf_urls")

        for field in url_fields:
            value = payload.get(field)
            url = self._coerce_url(value)
            if url:
                return url

        for field in list_fields:
            value = payload.get(field)
            url = self._coerce_url(value)
            if url:
                return url

        # Nested plan metadata
        plan = payload.get("plan")
        if isinstance(plan, Mapping):
            url = self._coerce_url(plan.get("url"))
            if url:
                return url

        return None

    @staticmethod
    def _coerce_url(value: Any) -> Optional[str]:
        """Normalize different URL field shapes to a string."""

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()

        return None

    def _deserialize_job(
        self, job: Mapping[str, Any]
    ) -> Tuple[WorkArtifact, Optional[DownloadContext]]:
        """Return a typed artifact/context pair for the leased job."""

        artifact_json = job.get("artifact_json", "{}")
        try:
            raw_payload = json.loads(artifact_json) if artifact_json else {}
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise _ArtifactValidationError(f"invalid JSON payload: {exc}") from exc

        if isinstance(raw_payload, WorkArtifact):
            return raw_payload, None

        if not isinstance(raw_payload, Mapping):
            raise _ArtifactValidationError(
                "expected artifact payload to be a mapping after JSON decode"
            )

        artifact_payload: Any
        context_payload: Any = None

        if "artifact" in raw_payload and isinstance(raw_payload["artifact"], Mapping):
            artifact_payload = raw_payload["artifact"]
        else:
            artifact_payload = raw_payload

        for key in ("context", "ctx", "download_context"):
            if key in raw_payload:
                context_payload = raw_payload[key]
                break

        if isinstance(artifact_payload, WorkArtifact):
            artifact = artifact_payload
        else:
            artifact = self._build_work_artifact(artifact_payload)

        context: Optional[DownloadContext] = None
        if isinstance(context_payload, DownloadContext):
            context = context_payload
        elif context_payload:
            if not isinstance(context_payload, Mapping):
                raise _ArtifactValidationError("context payload must be a mapping")
            try:
                context = DownloadContext.from_mapping(context_payload)
            except Exception as exc:  # pragma: no cover - validation guard
                raise _ArtifactValidationError(f"invalid context payload: {exc}") from exc

        return artifact, context

    def _build_work_artifact(self, payload: Mapping[str, Any]) -> WorkArtifact:
        """Construct a WorkArtifact from a mapping payload."""

        if not isinstance(payload, Mapping):
            raise _ArtifactValidationError("artifact payload must be a mapping")

        try:
            pdf_dir = Path(str(payload["pdf_dir"]))
            html_dir = Path(str(payload["html_dir"]))
            xml_dir = Path(str(payload["xml_dir"]))
        except KeyError as exc:  # pragma: no cover - ensures deterministic failure
            raise _ArtifactValidationError(f"missing required field: {exc.args[0]}") from exc

        try:
            artifact = WorkArtifact(
                work_id=str(payload["work_id"]),
                title=str(payload.get("title", "")),
                publication_year=payload.get("publication_year"),
                doi=payload.get("doi"),
                pmid=payload.get("pmid"),
                pmcid=payload.get("pmcid"),
                arxiv_id=payload.get("arxiv_id"),
                landing_urls=list(payload.get("landing_urls", [])),
                pdf_urls=list(payload.get("pdf_urls", [])),
                open_access_url=payload.get("open_access_url"),
                source_display_names=list(payload.get("source_display_names", [])),
                base_stem=str(payload.get("base_stem", payload.get("work_id", ""))),
                pdf_dir=pdf_dir,
                html_dir=html_dir,
                xml_dir=xml_dir,
                failed_pdf_urls=list(payload.get("failed_pdf_urls", [])),
                metadata=dict(payload.get("metadata", {})),
            )
        except KeyError as exc:
            raise _ArtifactValidationError(f"missing required field: {exc.args[0]}") from exc
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise _ArtifactValidationError(f"invalid artifact payload: {exc}") from exc

        return artifact


class _ArtifactValidationError(RuntimeError):
    """Raised when an artifact payload cannot be converted for the pipeline."""
