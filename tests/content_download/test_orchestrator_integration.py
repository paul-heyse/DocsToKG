# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.test_orchestrator_integration",
#   "purpose": "End-to-end integration tests for work orchestration system",
#   "sections": [
#     {"id": "lifecycle", "name": "JobLifecycleTests", "anchor": "#class-joblifecycletests", "kind": "test"},
#     {"id": "concurrency", "name": "ConcurrencyTests", "anchor": "#class-concurrencytests", "kind": "test"},
#     {"id": "recovery", "name": "CrashRecoveryTests", "anchor": "#class-crashrecoverytests", "kind": "test"}
#   ]
# }
# === /NAVMAP ===

"""End-to-end integration tests for work orchestration.

Tests the complete job lifecycle: enqueue → lease → process → ack,
including failure scenarios, crash recovery, and concurrent processing.
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Mapping

from DocsToKG.ContentDownload.config.models import (
    OrchestratorConfig,
    QueueConfig,
)
from DocsToKG.ContentDownload.orchestrator import (
    KeyedLimiter,
    WorkQueue,
)
from DocsToKG.ContentDownload.orchestrator.models import JobResult, JobState


class TestJobLifecycle(unittest.TestCase):
    """Tests for complete job lifecycle."""

    def setUp(self) -> None:
        """Create temporary queue for testing."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.queue_path = str(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.tmpdir.cleanup()

    def test_complete_job_lifecycle(self) -> None:
        """Test job from enqueue through completion."""
        queue = WorkQueue(self.queue_path, wal_mode=True)

        # 1. Enqueue
        enqueued = queue.enqueue("doi:test-1", {"doi": "test-1"})
        self.assertTrue(enqueued)
        stats = queue.stats()
        self.assertEqual(stats["queued"], 1)

        # 2. Lease
        jobs = queue.lease("worker-1", limit=1, lease_ttl_sec=600)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        stats = queue.stats()
        self.assertEqual(stats["queued"], 0)
        self.assertEqual(stats["in_progress"], 1)

        # 3. Ack as done
        queue.ack(job["id"], "done")
        stats = queue.stats()
        self.assertEqual(stats["in_progress"], 0)
        self.assertEqual(stats["done"], 1)

    def test_job_idempotence(self) -> None:
        """Test that enqueuing same ID twice is idempotent."""
        queue = WorkQueue(self.queue_path)

        # First enqueue
        result1 = queue.enqueue("doi:test-1", {"doi": "test-1"})
        self.assertTrue(result1)

        # Second enqueue (same ID)
        result2 = queue.enqueue("doi:test-1", {"doi": "test-1"})
        self.assertFalse(result2)  # Already exists

        stats = queue.stats()
        self.assertEqual(stats["total"], 1)

    def test_job_skip_state(self) -> None:
        """Test job marked as skipped."""
        queue = WorkQueue(self.queue_path)

        queue.enqueue("doi:test-1", {"doi": "test-1"})
        jobs = queue.lease("worker-1", 1, 600)
        queue.ack(jobs[0]["id"], "skipped", last_error="Not found")

        stats = queue.stats()
        self.assertEqual(stats["skipped"], 1)

    def test_job_error_state(self) -> None:
        """Test job marked as error."""
        queue = WorkQueue(self.queue_path)

        queue.enqueue("doi:test-1", {"doi": "test-1"})
        jobs = queue.lease("worker-1", 1, 600)
        queue.ack(jobs[0]["id"], "error", last_error="Network timeout")

        stats = queue.stats()
        self.assertEqual(stats["error"], 1)

    def test_retry_respects_backoff(self) -> None:
        """Jobs scheduled for retry should not be leased until the backoff expires."""

        queue = WorkQueue(self.queue_path)
        queue.enqueue("doi:test-1", {"doi": "test-1"})

        leased = queue.lease("worker-1", 1, 600)
        self.assertEqual(len(leased), 1)
        job = leased[0]

        queue.fail_and_retry(job["id"], backoff_sec=2, max_attempts=3, last_error="boom")

        # Immediate lease attempt should not return the deferred job.
        self.assertEqual(
            queue.lease("worker-2", 1, 600),
            [],
        )

        time.sleep(2.1)

        retried = queue.lease("worker-2", 1, 600)
        self.assertEqual(len(retried), 1)
        self.assertEqual(retried[0]["id"], job["id"])

        # The job should have its delay metadata cleared once leased again.
        conn = sqlite3.connect(self.queue_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT available_at FROM jobs WHERE id = ?", (job["id"],)
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertIsNone(row["available_at"])


class TestConcurrency(unittest.TestCase):
    """Tests for concurrent job processing."""

    def setUp(self) -> None:
        """Create temporary queue and limiters."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.queue_path = str(Path(self.tmpdir.name) / "test.sqlite")
        self.queue = WorkQueue(self.queue_path, wal_mode=True)

    def tearDown(self) -> None:
        """Clean up."""
        self.tmpdir.cleanup()

    def test_multiple_workers_concurrent_lease(self) -> None:
        """Multiple workers can lease jobs concurrently."""
        # Enqueue 10 jobs
        for i in range(10):
            self.queue.enqueue(f"doi:test-{i}", {"doi": f"test-{i}"})

        # Lease with 3 workers
        leased_jobs = []
        lock = threading.Lock()

        def lease_jobs(worker_id: str) -> None:
            jobs = self.queue.lease(worker_id, limit=5, lease_ttl_sec=600)
            with lock:
                leased_jobs.extend(jobs)

        threads = [threading.Thread(target=lease_jobs, args=(f"worker-{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have leased some (but not all due to contention)
        self.assertGreater(len(leased_jobs), 0)
        self.assertLessEqual(len(leased_jobs), 10)

    def test_keyed_limiter_per_resolver_concurrency(self) -> None:
        """KeyedLimiter respects per-resolver concurrency limits."""
        limiter = KeyedLimiter(default_limit=10, per_key={"resolver-a": 2, "resolver-b": 3})

        # Track concurrent accesses
        concurrent_count = []
        active = []
        lock = threading.Lock()

        def acquire_and_hold(resolver: str, duration: float) -> None:
            limiter.acquire(resolver)
            with lock:
                active.append(resolver)
                concurrent_count.append(len(active))
            time.sleep(duration)
            with lock:
                active.remove(resolver)
            limiter.release(resolver)

        # 5 threads for resolver-a (limit 2)
        threads = []
        for i in range(5):
            t = threading.Thread(target=acquire_and_hold, args=("resolver-a", 0.1))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Max concurrent should not exceed 2 for resolver-a
        max_concurrent = max(concurrent_count) if concurrent_count else 0
        self.assertLessEqual(max_concurrent, 2)

    def test_keyed_limiter_per_host_concurrency(self) -> None:
        """KeyedLimiter respects per-host concurrency limits."""
        limiter = KeyedLimiter(default_limit=3)

        acquired = []
        lock = threading.Lock()

        def acquire_key(key: str, count: int) -> None:
            for _ in range(count):
                limiter.acquire(key)
                with lock:
                    acquired.append(key)
                time.sleep(0.05)
                limiter.release(key)

        # Try to acquire same key 10 times concurrently (limit 3)
        threads = [threading.Thread(target=acquire_key, args=("host-a", 10)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(acquired), 40)  # All acquired eventually


class TestCrashRecovery(unittest.TestCase):
    """Tests for crash recovery mechanisms."""

    def setUp(self) -> None:
        """Create temporary queue."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.queue_path = str(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        """Clean up."""
        self.tmpdir.cleanup()

    def test_stale_lease_recovery(self) -> None:
        """Stale leases are recovered and re-leased."""
        queue = WorkQueue(self.queue_path)

        # Enqueue and lease with short TTL
        queue.enqueue("doi:test-1", {"doi": "test-1"})
        jobs1 = queue.lease("worker-1", 1, lease_ttl_sec=1)  # 1 second TTL
        self.assertEqual(len(jobs1), 1)

        # Wait for lease to expire
        time.sleep(1.2)

        # Lease again (should get same job from stale lease)
        jobs2 = queue.lease("worker-2", 1, lease_ttl_sec=600)
        self.assertEqual(len(jobs2), 1)
        self.assertEqual(jobs2[0]["artifact_id"], "doi:test-1")

    def test_heartbeat_extends_lease(self) -> None:
        """Heartbeat updates lease timestamp without raising errors."""
        queue = WorkQueue(self.queue_path)

        queue.enqueue("doi:test-1", {"doi": "test-1"})
        jobs = queue.lease("worker-1", 1, lease_ttl_sec=600)
        self.assertEqual(len(jobs), 1)

        # Call heartbeat (should not raise)
        queue.heartbeat("worker-1")

        # Verify job is still in progress
        stats = queue.stats()
        self.assertEqual(stats["in_progress"], 1)
        self.assertEqual(stats["queued"], 0)

    def test_max_attempts_retry_logic(self) -> None:
        """Test retry escalation to error state."""
        queue = WorkQueue(self.queue_path)

        queue.enqueue("doi:test-1", {"doi": "test-1"})

        # Attempt 1: fail and retry
        jobs = queue.lease("worker-1", 1, 600)
        queue.fail_and_retry(jobs[0]["id"], backoff_sec=1, max_attempts=2, last_error="Error 1")

        time.sleep(1.1)

        # Job should be re-queued
        jobs = queue.lease("worker-2", 1, 600)
        self.assertEqual(len(jobs), 1)

        # Attempt 2: fail and retry (max attempts exceeded)
        queue.fail_and_retry(jobs[0]["id"], backoff_sec=1, max_attempts=2, last_error="Error 2")

        # Job should be in error state
        stats = queue.stats()
        self.assertEqual(stats["error"], 1)


class TestConfigIntegration(unittest.TestCase):
    """Tests for configuration integration."""

    def setUp(self) -> None:
        """Create temporary queue."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.queue_path = str(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        """Clean up."""
        self.tmpdir.cleanup()

    def test_orchestrator_config_with_queue(self) -> None:
        """OrchestratorConfig works with WorkQueue."""
        config = OrchestratorConfig(max_workers=4, max_per_host=2, lease_ttl_seconds=120)

        queue_cfg = QueueConfig(path=self.queue_path, wal_mode=True, timeout_sec=10)

        # Create queue with config
        queue = WorkQueue(queue_cfg.path, wal_mode=queue_cfg.wal_mode)

        # Enqueue and verify
        queue.enqueue("doi:test", {"doi": "test"})
        jobs = queue.lease("worker-1", 1, config.lease_ttl_seconds)
        self.assertEqual(len(jobs), 1)

    def test_keyed_limiter_with_orchestrator_config(self) -> None:
        """KeyedLimiter uses OrchestratorConfig settings."""
        config = OrchestratorConfig(
            max_per_resolver={"resolver-a": 2, "resolver-b": 4}, max_per_host=3
        )

        # Create limiter with config
        limiter = KeyedLimiter(default_limit=config.max_per_host, per_key=config.max_per_resolver)

        # Verify config applied
        self.assertEqual(limiter.get_limit("resolver-a"), 2)
        self.assertEqual(limiter.get_limit("resolver-b"), 4)
        self.assertEqual(limiter.get_limit("unknown"), 3)  # default


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and edge cases."""

    def setUp(self) -> None:
        """Create temporary queue."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.queue_path = str(Path(self.tmpdir.name) / "test.sqlite")

    def tearDown(self) -> None:
        """Clean up."""
        self.tmpdir.cleanup()

    def test_lease_nonexistent_job(self) -> None:
        """Leasing non-existent job ID is safe."""
        queue = WorkQueue(self.queue_path)

        # Try to ack non-existent job
        queue.ack(9999, "done")  # Should not raise

        stats = queue.stats()
        self.assertEqual(stats["total"], 0)

    def test_multiple_format_conversions(self) -> None:
        """JSON serialization/deserialization is consistent."""
        queue = WorkQueue(self.queue_path)

        artifact = {
            "doi": "10.1234/test",
            "title": "Test Document",
            "nested": {"key": "value", "list": [1, 2, 3]},
        }

        queue.enqueue("doi:test", artifact)
        jobs = queue.lease("worker-1", 1, 600)

        # Verify artifact can be deserialized
        import json

        rehydrated = json.loads(jobs[0]["artifact_json"])
        self.assertEqual(rehydrated, artifact)

    def test_empty_queue_operations(self) -> None:
        """Operations on empty queue are safe."""
        queue = WorkQueue(self.queue_path)

        stats = queue.stats()
        self.assertEqual(stats["total"], 0)

        jobs = queue.lease("worker-1", 10, 600)
        self.assertEqual(len(jobs), 0)

        queue.heartbeat("worker-1")  # Should not raise


if __name__ == "__main__":
    unittest.main()
