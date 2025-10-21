# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.test_tokenbucket_threadsafety",
#   "purpose": "Thread-safety tests for TokenBucket rate limiter",
#   "sections": [
#     {"id": "concurrent_acquisitions", "name": "test_concurrent_acquisitions", "anchor": "#function-test-concurrent-acquisitions", "kind": "test"},
#     {"id": "concurrent_refill", "name": "test_concurrent_refill", "anchor": "#function-test-concurrent-refill", "kind": "test"},
#     {"id": "stress_test", "name": "test_concurrent_stress", "anchor": "#function-test-concurrent-stress", "kind": "test"}
#   ]
# }
# === /NAVMAP ===

"""Thread-safety tests for TokenBucket rate limiter.

Tests that TokenBucket is safe for concurrent access by multiple threads
in a multi-worker environment. Verifies:
- No race conditions during acquisitions
- Correct token accounting under concurrency
- No deadlocks
- Proper synchronization
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from DocsToKG.ContentDownload.resolver_http_client import TokenBucket


class TestTokenBucketThreadSafety(unittest.TestCase):
    """Thread-safety tests for TokenBucket."""

    def test_concurrent_acquisitions_no_data_race(self) -> None:
        """Multiple threads acquiring tokens simultaneously don't cause data races."""
        bucket = TokenBucket(capacity=100.0, refill_per_sec=10.0)
        results: list[Any] = []
        errors: list[Exception] = []

        def acquire_token() -> None:
            try:
                result = bucket.acquire(tokens=1.0, timeout_s=1.0)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Spawn 20 threads all trying to acquire at once
        threads = [threading.Thread(target=acquire_token) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        # All should succeed with 0 sleep (plenty of tokens available)
        assert all(r == 0.0 for r in results)

    def test_concurrent_acquisitions_respect_capacity(self) -> None:
        """Concurrent acquisitions respect token bucket capacity."""
        bucket = TokenBucket(capacity=5.0, refill_per_sec=0.0)  # No refill
        acquired = []
        errors: list[Exception] = []

        def acquire_token() -> None:
            try:
                result = bucket.acquire(tokens=1.0, timeout_s=0.5)
                acquired.append(result)
            except TimeoutError:
                # Expected for threads that can't get tokens
                pass
            except Exception as e:
                errors.append(e)

        # Try to acquire 10 tokens with only 5 available
        threads = [threading.Thread(target=acquire_token) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # At most 5 should succeed immediately
        assert len(acquired) <= 5

    def test_concurrent_acquisitions_with_refill(self) -> None:
        """Tokens refill correctly under concurrent acquisition."""
        bucket = TokenBucket(capacity=10.0, refill_per_sec=5.0)
        acquired = []
        errors: list[Exception] = []

        def acquire_token(delay: float) -> None:
            try:
                time.sleep(delay)
                result = bucket.acquire(tokens=1.0, timeout_s=2.0)
                acquired.append(result)
            except Exception as e:
                errors.append(e)

        # Wave 1: 5 threads acquire immediately (should succeed)
        # Wave 2: 5 threads acquire after 0.2s (tokens should have refilled)
        threads: list[threading.Thread] = []
        for i in range(5):
            threads.append(threading.Thread(target=acquire_token, args=(0.0,)))
        for i in range(5):
            threads.append(threading.Thread(target=acquire_token, args=(0.2,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 10+ acquisitions succeed (wave1 + refilled tokens)
        assert len(acquired) >= 8

    def test_high_concurrency_stress(self) -> None:
        """TokenBucket handles high concurrency without corruption."""
        bucket = TokenBucket(capacity=100.0, refill_per_sec=50.0)
        acquired: list[float] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def acquire_many() -> None:
            try:
                for _ in range(10):
                    result = bucket.acquire(tokens=0.5, timeout_s=1.0)
                    with lock:
                        acquired.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        # 50 threads each making 10 acquisitions = 500 total acquisitions
        threads = [threading.Thread(target=acquire_many) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(acquired) == 500

    def test_no_deadlock_on_timeout(self) -> None:
        """Timeouts don't cause deadlocks under concurrent load."""
        bucket = TokenBucket(capacity=1.0, refill_per_sec=0.1)
        completed = []
        lock = threading.Lock()

        def try_acquire_with_timeout() -> None:
            try:
                bucket.acquire(tokens=100.0, timeout_s=0.1)
            except TimeoutError:
                pass  # Expected
            finally:
                with lock:
                    completed.append(True)

        threads = [threading.Thread(target=try_acquire_with_timeout) for _ in range(10)]
        for t in threads:
            t.start()

        # Should complete within reasonable time (no deadlock)
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread deadlocked"

        assert len(completed) == 10

    def test_parallel_burst_acquisitions(self) -> None:
        """Multiple threads acquiring burst allowance works correctly."""
        bucket = TokenBucket(capacity=10.0, refill_per_sec=1.0, burst=20.0)
        results: list[tuple[float, float]] = []
        lock = threading.Lock()

        def acquire_burst() -> None:
            try:
                result = bucket.acquire(tokens=5.0, timeout_s=2.0)
                with lock:
                    results.append((time.time(), result))
            except Exception:
                pass

        # Multiple threads acquiring burst
        threads = [threading.Thread(target=acquire_burst) for _ in range(6)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        # All should eventually succeed (burst allows up to capacity + burst = 30 tokens)
        assert len(results) >= 4  # At least 4 should succeed
        assert elapsed < 10.0  # Should complete quickly

    def test_alternating_acquire_release_pattern(self) -> None:
        """Simulates alternating acquire/release with thread-safe state."""
        bucket = TokenBucket(capacity=5.0, refill_per_sec=2.0)
        iterations = 50
        success_count = []
        lock = threading.Lock()

        def alternating_acquire() -> None:
            try:
                for i in range(iterations):
                    # Try to acquire 1 token
                    bucket.acquire(tokens=1.0, timeout_s=1.0)
                    with lock:
                        success_count.append(1)
                    # "Release" by letting time pass for refill
                    time.sleep(0.05)
            except TimeoutError:
                pass  # Some will timeout, that's ok

        # Multiple threads alternating
        threads = [threading.Thread(target=alternating_acquire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With high contention and limited capacity, we should get at least some successes
        # Key is that there are NO data races or deadlocks
        assert len(success_count) >= 20, f"Expected at least 20 successes, got {len(success_count)}"


if __name__ == "__main__":
    unittest.main()
