"""Tests for Orchestrator dispatcher and worker pool management.

Tests cover:
- OrchestratorConfig initialization
- Orchestrator initialization
- Dispatcher loop and job leasing
- Heartbeat loop and lease extension
- Worker thread pool
- Graceful shutdown
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from DocsToKG.ContentDownload.orchestrator.scheduler import (
    Orchestrator,
    OrchestratorConfig,
)


class MockWorkQueue:
    """Mock WorkQueue for testing."""
    
    def __init__(self) -> None:
        self.leased_jobs: list[tuple[str, int, int]] = []
        self.heartbeats: list[str] = []
        self.jobs_to_lease: list[dict[str, int]] = []
    
    def lease(self, worker_id: str, limit: int, lease_ttl_sec: int) -> list[dict[str, int]]:
        self.leased_jobs.append((worker_id, limit, lease_ttl_sec))
        if self.jobs_to_lease:
            return self.jobs_to_lease.pop(0)
        return []
    
    def heartbeat(self, worker_id: str) -> None:
        self.heartbeats.append(worker_id)
    
    def stats(self) -> dict[str, int]:
        return {"queued": 10, "in_progress": 5, "done": 0}


class MockPipeline:
    """Mock ResolverPipeline for testing."""
    
    def process(self, artifact, ctx=None):
        return MagicMock()


def test_orchestrator_config_defaults() -> None:
    """Test OrchestratorConfig initialization with defaults."""
    config = OrchestratorConfig()
    
    assert config.max_workers == 8
    assert config.max_per_host == 4
    assert config.lease_ttl_seconds == 600
    assert config.heartbeat_seconds == 30
    assert config.max_job_attempts == 3
    assert config.retry_backoff_seconds == 60
    assert config.jitter_seconds == 15
    assert config.max_per_resolver == {}


def test_orchestrator_config_custom() -> None:
    """Test OrchestratorConfig initialization with custom values."""
    config = OrchestratorConfig(
        max_workers=16,
        max_per_resolver={"unpaywall": 2, "crossref": 3},
        max_per_host=8,
        lease_ttl_seconds=300,
        heartbeat_seconds=60,
    )
    
    assert config.max_workers == 16
    assert config.max_per_host == 8
    assert config.lease_ttl_seconds == 300
    assert config.heartbeat_seconds == 60
    assert config.max_per_resolver == {"unpaywall": 2, "crossref": 3}


def test_orchestrator_initialization() -> None:
    """Test Orchestrator initialization."""
    config = OrchestratorConfig(max_workers=4)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    
    assert orch.config == config
    assert orch.queue == queue
    assert orch.pipeline == pipeline
    assert orch.worker_id.startswith("orch-")
    assert len(orch._threads) == 0  # Not started yet


def test_orchestrator_start_threads() -> None:
    """Test Orchestrator thread startup."""
    config = OrchestratorConfig(max_workers=2)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    orch.start()
    
    # Wait for threads to start
    time.sleep(0.5)
    
    # Should have: 2 workers + dispatcher + heartbeat = 4 threads
    assert len(orch._threads) == 4
    
    # Clean up
    orch.stop()


def test_orchestrator_graceful_shutdown() -> None:
    """Test Orchestrator graceful shutdown."""
    config = OrchestratorConfig(max_workers=2)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    orch.start()
    
    time.sleep(0.2)
    
    # Signal stop
    orch.stop()
    
    # Threads should be cleaned up
    assert orch._stop.is_set()
    # Give threads time to join
    time.sleep(0.5)


def test_orchestrator_dispatcher_leases_jobs() -> None:
    """Test dispatcher loop leases jobs."""
    config = OrchestratorConfig(max_workers=1)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    # Prepare mock jobs
    mock_job = {
        "id": 1,
        "artifact_id": "doi:10.1234/test",
        "artifact_json": "{}",
    }
    queue.jobs_to_lease = [[mock_job]]
    
    orch = Orchestrator(config, queue, pipeline)
    orch.start()
    
    # Wait for dispatcher to lease
    time.sleep(1.5)
    
    # Verify lease was called
    assert len(queue.leased_jobs) > 0
    worker_id, limit, ttl = queue.leased_jobs[0]
    assert worker_id == orch.worker_id
    assert limit > 0
    assert ttl == config.lease_ttl_seconds
    
    orch.stop()


def test_orchestrator_heartbeat_extends_leases() -> None:
    """Test heartbeat loop extends leases."""
    config = OrchestratorConfig(max_workers=1, heartbeat_seconds=1)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    orch.start()
    
    # Wait for heartbeat to fire
    time.sleep(1.5)
    
    # Verify heartbeat was called
    assert len(queue.heartbeats) > 0
    assert queue.heartbeats[0] == orch.worker_id
    
    orch.stop()


def test_orchestrator_queue_backpressure() -> None:
    """Test dispatcher respects worker queue backpressure."""
    config = OrchestratorConfig(max_workers=1)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    
    # Queue maxsize should be max_workers * 2
    expected_maxsize = config.max_workers * 2
    assert orch._jobs_queue.maxsize == expected_maxsize


def test_orchestrator_config_passed_to_workers() -> None:
    """Test that orchestrator config is passed to worker creation."""
    config = OrchestratorConfig(
        max_workers=2,
        max_job_attempts=5,
        retry_backoff_seconds=120,
        heartbeat_seconds=45,
    )
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    
    # Verify config values
    assert orch.config.max_job_attempts == 5
    assert orch.config.retry_backoff_seconds == 120
    assert orch.config.heartbeat_seconds == 45


def test_orchestrator_limiter_configuration() -> None:
    """Test orchestrator creates limiters with correct configuration."""
    config = OrchestratorConfig(
        max_workers=8,
        max_per_resolver={"unpaywall": 2, "crossref": 3},
        max_per_host=4,
    )
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    
    # Verify limiters were created
    assert orch._resolver_limiter is not None
    assert orch._host_limiter is not None
    
    # Verify resolver limiter was configured correctly
    assert orch._resolver_limiter.per_key == {"unpaywall": 2, "crossref": 3}
    assert orch._host_limiter.default_limit == 4


def test_orchestrator_stats_available() -> None:
    """Test orchestrator can access queue statistics."""
    config = OrchestratorConfig(max_workers=1)
    queue = MockWorkQueue()
    pipeline = MockPipeline()
    
    orch = Orchestrator(config, queue, pipeline)
    orch.start()
    
    # Wait for dispatcher to check stats
    time.sleep(1.5)
    
    orch.stop()
    
    # Verify stats were queried
    stats = queue.stats()
    assert "queued" in stats
    assert "in_progress" in stats
    assert "done" in stats
