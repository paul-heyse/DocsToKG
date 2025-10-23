"""Concurrency regression tests for :mod:`resolver_http_client.TokenBucket`."""

from __future__ import annotations

import threading
import time

from DocsToKG.ContentDownload.resolver_http_client import TokenBucket


def test_token_bucket_threaded_acquire_and_refund() -> None:
    """Ensure concurrent acquire/refund cycles never sleep negatively and still rate limit."""

    bucket = TokenBucket(capacity=1.0, refill_per_sec=5.0, burst=0.0)
    start_barrier = threading.Barrier(parties=4)
    waits: list[float] = []
    errors: list[BaseException] = []
    iterations = 50
    work_time_s = 0.02

    def worker() -> None:
        try:
            start_barrier.wait()
            for _ in range(iterations):
                wait = bucket.acquire(tokens=1.0, timeout_s=5.0)
                waits.append(wait)
                time.sleep(work_time_s)
                bucket.refund(tokens=1.0)
        except BaseException as exc:  # pragma: no cover - captured for assertions
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(start_barrier.parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"token bucket raised unexpected errors: {errors}"
    assert len(waits) == iterations * len(threads)
    assert all(wait >= 0.0 for wait in waits)
    assert max(waits) >= work_time_s - 0.005
