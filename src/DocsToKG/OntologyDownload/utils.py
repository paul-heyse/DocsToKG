<<<<<<< HEAD
"""Utility helpers supporting OntologyDownload components."""
=======
"""Utility helpers shared across ontology download modules.

This module centralizes retry orchestration to provide consistent
exponential backoff behaviour across resolvers and download components.
"""
>>>>>>> 9b35f42188d4e1aa83a450f8ffa471e6683bfdc8

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

__all__ = ["retry_with_backoff"]


def retry_with_backoff(
    func: Callable[[], T],
    *,
<<<<<<< HEAD
    retryable: Callable[[Exception], bool],
=======
    retryable: Callable[[BaseException], bool],
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.5,
    callback: Optional[Callable[[int, BaseException, float], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute ``func`` with exponential backoff until it succeeds.

    Args:
        func: Callable that performs the operation to retry.
        retryable: Predicate returning ``True`` when an exception should trigger
            another attempt.
        max_attempts: Maximum number of attempts including the initial call.
        backoff_base: Base delay in seconds used for the exponential schedule.
        jitter: Maximum random jitter (uniform) added to each delay.
        callback: Optional hook invoked before sleeping with
            ``(attempt_number, error, delay_seconds)``.
        sleep: Sleep function, overridable for tests.

    Returns:
        Result returned by ``func`` when successful.

    Raises:
        ValueError: When ``max_attempts`` is less than one.
        BaseException: Re-raises the last exception from ``func`` when retries
            are exhausted or the predicate determines it is not retryable.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except BaseException as exc:  # pylint: disable=broad-except
            should_retry = attempt < max_attempts and retryable(exc)
            if not should_retry:
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter > 0:
                delay += random.uniform(0, jitter)
            if callback is not None:
                callback(attempt, exc, delay)
            sleep(max(delay, 0.0))


__all__ = ["retry_with_backoff"]

>>>>>>> 9b35f42188d4e1aa83a450f8ffa471e6683bfdc8
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
