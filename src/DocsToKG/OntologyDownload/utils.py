"""Utility helpers shared across ontology download modules.

Currently hosts the unified exponential backoff helper that underpins resolver
planning and download retries in the refactored ontology downloader.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

__all__ = ["retry_with_backoff"]


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
        func: Zero-argument callable to invoke.
        retryable: Predicate returning ``True`` when the raised exception should
            trigger another attempt.
        max_attempts: Maximum number of attempts including the initial call.
        backoff_base: Base delay in seconds used for the exponential schedule.
        jitter: Maximum random jitter (uniform) added to each delay.
        callback: Optional hook invoked before sleeping with
            ``(attempt_number, error, delay_seconds)``.
        sleep: Sleep function, overridable for deterministic tests.

    Returns:
        The result produced by ``func`` when it succeeds.

    Raises:
        ValueError: If ``max_attempts`` is less than one.
        BaseException: Re-raises the last exception from ``func`` when retries
            are exhausted or the predicate indicates it is not retryable.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except BaseException as exc:  # pragma: no cover - behaviour verified via callers
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            if callback is not None:
                try:
                    callback(attempt, exc, delay)
                except Exception:  # pragma: no cover - defensive against callbacks
                    pass
            sleep(max(delay, 0.0))
