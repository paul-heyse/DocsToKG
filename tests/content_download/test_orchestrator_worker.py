"""Tests for Worker job execution wrapper.

Tests cover:
- Worker initialization and lifecycle
- Successful job execution and ack
- Job failure and retry logic
- Artifact rehydration errors
- Pipeline errors
- Graceful shutdown
- Concurrent job tracking
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping
from unittest.mock import MagicMock, Mock, patch

import pytest

from DocsToKG.ContentDownload.api.types import DownloadOutcome
from DocsToKG.ContentDownload.core import ReasonCode
from DocsToKG.ContentDownload.orchestrator.models import JobState
from DocsToKG.ContentDownload.orchestrator.workers import Worker


class MockWorkQueue:
    """Mock WorkQueue for testing."""

    def __init__(self) -> None:
        self.acked_jobs: list[tuple[int, str, Any]] = []
        self.failed_jobs: list[tuple[int, int, int, str]] = []

    def ack(self, job_id: int, outcome: str, last_error: Any = None) -> None:
        self.acked_jobs.append((job_id, outcome, last_error))

    def fail_and_retry(
        self, job_id: int, backoff_sec: int, max_attempts: int, last_error: str
    ) -> None:
        self.failed_jobs.append((job_id, backoff_sec, max_attempts))


class MockPipeline:
    """Mock ResolverPipeline for testing."""

    def __init__(self) -> None:
        self.processed_artifacts: list[Any] = []
        self.outcome_to_return = DownloadOutcome(
            ok=True, classification="success", path="/tmp/file.pdf"
        )

    def process(self, artifact: Any, ctx: Any = None) -> DownloadOutcome:
        self.processed_artifacts.append(artifact)
        return self.outcome_to_return


class MockLimiter:
    """Mock KeyedLimiter for testing."""

    def acquire(self, key: str) -> None:
        pass

    def release(self, key: str) -> None:
        pass


def test_worker_initialization() -> None:
    """Test Worker initialization."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    assert worker.worker_id == "worker-1"
    assert worker.heartbeat_sec == 30
    assert worker.max_job_attempts == 3
    assert worker.retry_backoff == 60
    assert worker.jitter == 15


def test_worker_run_successful_job() -> None:
    """Test worker successfully executes and acks a job."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    artifact_data = {"doi": "10.1234/example"}
    job = {
        "id": 123,
        "artifact_id": "doi:10.1234/example",
        "artifact_json": json.dumps(artifact_data),
    }

    worker.run_one(job)

    # Verify job was processed
    assert len(pipeline.processed_artifacts) == 1
    assert pipeline.processed_artifacts[0] == artifact_data

    # Verify job was acked with success
    assert len(queue.acked_jobs) == 1
    job_id, state, error = queue.acked_jobs[0]
    assert job_id == 123
    assert state == "done"
    assert error is None


def test_worker_run_skipped_job() -> None:
    """Test worker acks skipped job."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    pipeline.outcome_to_return = DownloadOutcome(
        ok=False,
        classification="skip",
        reason=ReasonCode.ROBOTS_DISALLOWED,
    )
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    job = {
        "id": 456,
        "artifact_id": "doi:10.5678/test",
        "artifact_json": "{}",
    }

    worker.run_one(job)

    # Verify skipped state
    assert len(queue.acked_jobs) == 1
    job_id, state, error = queue.acked_jobs[0]
    assert job_id == 456
    assert state == "skipped"


def test_worker_artifact_rehydration_error() -> None:
    """Test worker retries on artifact rehydration error."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    # Job with invalid JSON
    job = {
        "id": 789,
        "artifact_id": "doi:10.9999/bad",
        "artifact_json": "invalid json",
    }

    worker.run_one(job)

    # Verify job was not processed
    assert len(pipeline.processed_artifacts) == 0

    # Verify job was marked for retry
    assert len(queue.failed_jobs) == 1
    job_id, backoff, max_attempts = queue.failed_jobs[0]
    assert job_id == 789
    assert backoff == 60
    assert max_attempts == 3


def test_worker_pipeline_error_retry() -> None:
    """Test worker retries on pipeline error."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    pipeline.process = Mock(side_effect=RuntimeError("Pipeline failed"))
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    job = {
        "id": 1000,
        "artifact_id": "doi:10.1111/error",
        "artifact_json": "{}",
    }

    worker.run_one(job)

    # Verify job was marked for retry
    assert len(queue.failed_jobs) == 1
    job_id, backoff, max_attempts = queue.failed_jobs[0]
    assert job_id == 1000


def test_worker_stop_signal() -> None:
    """Test worker respects stop signal."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    # Signal stop
    worker.stop()

    job = {
        "id": 1111,
        "artifact_id": "doi:10.2222/stop",
        "artifact_json": "{}",
    }

    worker.run_one(job)

    # Verify job was not processed
    assert len(pipeline.processed_artifacts) == 0
    assert len(queue.acked_jobs) == 0
    assert len(queue.failed_jobs) == 0


def test_worker_concurrent_job_tracking() -> None:
    """Test worker tracks concurrent jobs."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    job1 = {
        "id": 2000,
        "artifact_id": "doi:10.3333/job1",
        "artifact_json": "{}",
    }

    job2 = {
        "id": 2001,
        "artifact_id": "doi:10.3333/job2",
        "artifact_json": "{}",
    }

    # Run jobs sequentially
    worker.run_one(job1)
    worker.run_one(job2)

    # Both should be processed
    assert len(pipeline.processed_artifacts) == 2
    assert len(queue.acked_jobs) == 2


def test_worker_compute_backoff_with_jitter() -> None:
    """Test backoff computation includes jitter."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    # Test backoff is within expected range
    backoffs = [worker._compute_backoff() for _ in range(100)]

    assert min(backoffs) >= 60
    assert max(backoffs) <= 75
    assert len(set(backoffs)) > 1  # Should have variation from jitter


def test_worker_empty_artifact_json() -> None:
    """Test worker handles empty artifact JSON."""
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    resolver_limiter = MockLimiter()
    host_limiter = MockLimiter()

    worker = Worker(
        worker_id="worker-1",
        queue=queue,
        pipeline=pipeline,
        resolver_limiter=resolver_limiter,
        host_limiter=host_limiter,
        heartbeat_sec=30,
        max_job_attempts=3,
        retry_backoff=60,
        jitter=15,
    )

    job = {
        "id": 3000,
        "artifact_id": "doi:10.4444/empty",
        "artifact_json": "",  # Empty
    }

    worker.run_one(job)

    # Should handle gracefully and pass empty dict to pipeline
    assert len(pipeline.processed_artifacts) == 1
    assert pipeline.processed_artifacts[0] == {}
    assert len(queue.acked_jobs) == 1
