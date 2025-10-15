"""Shared utility helpers for the ontology downloader package."""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retryable: Callable[[BaseException], bool],
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
