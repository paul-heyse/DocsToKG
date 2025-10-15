"""Utility helpers shared across ontology download modules.

This module centralizes retry orchestration to provide consistent
exponential backoff behaviour across resolvers and download components.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
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

    max_attempts: int,
    backoff_base: float,
    jitter: float = 0.0,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> T:
    """Execute ``func`` with exponential backoff until it succeeds or exhausts attempts."""

    attempt = 1
    while True:
        try:
            return func()
        except BaseException as exc:  # pragma: no cover - deliberate broad catch
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter > 0:
                delay += random.random() * jitter
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(delay)
            attempt += 1
    attempts: int = 3,
    backoff_base: float = 0.5,
    jitter: float = 0.1,
    callback: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Execute ``func`` retrying retryable exceptions with exponential backoff."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001 - propagate non-standard errors
            if not retryable(exc) or attempt == attempts:
                raise
            sleep_time = backoff_base * (2 ** (attempt - 1))
            if jitter:
                sleep_time += random.uniform(0.0, jitter)
            if callback:
                callback(attempt, exc, sleep_time)
            time.sleep(sleep_time)
    raise RuntimeError("retry_with_backoff reached unreachable state")


__all__ = ["retry_with_backoff"]
