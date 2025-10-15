"""Utility helpers supporting OntologyDownload components."""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

__all__ = ["retry_with_backoff"]


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retryable: Callable[[Exception], bool],
    max_attempts: int,
    backoff_base: float,
    jitter: float = 0.0,
    callback: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """Execute ``func`` with exponential backoff until success or exhaustion.

    Args:
        func: Zero-argument callable to invoke.
        retryable: Predicate returning ``True`` when the raised exception should
            trigger a retry instead of immediate propagation.
        max_attempts: Maximum number of attempts including the initial call.
        backoff_base: Base delay in seconds used for the exponential schedule.
        jitter: Optional jitter in seconds added to the computed sleep duration.
        callback: Optional hook invoked before sleeping with the attempt number,
            exception instance, and computed delay.

    Returns:
        Result returned by ``func`` when it succeeds.

    Raises:
        Exception: Propagates the most recent exception when retries are exhausted
            or when :func:`retryable` returns ``False``.
    """

    attempts = max(1, int(max_attempts))
    base_delay = max(0.0, float(backoff_base))
    jitter = max(0.0, float(jitter))

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= attempts or not retryable(exc):
                raise
            sleep_time = base_delay * (2 ** (attempt - 1))
            if jitter:
                sleep_time += random.uniform(0.0, jitter)
            if callback is not None:
                try:
                    callback(attempt, exc, sleep_time)
                except Exception:
                    pass
            if sleep_time > 0:
                time.sleep(sleep_time)

    raise RuntimeError("retry_with_backoff exhausted without returning")  # pragma: no cover
