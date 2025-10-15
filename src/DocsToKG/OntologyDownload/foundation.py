"""Cross-cutting utilities shared across the ontology downloader package.

This module intentionally groups helpers that are required by multiple layers
of the pipeline—retry orchestration, safe filename generation, and correlation
id creation—so that higher-level modules such as networking, logging, and the
CLI can depend on a single lightweight utility surface without introducing
import cycles.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
import uuid
from typing import Callable, Dict, Optional, TypeVar

T = TypeVar("T")

__all__ = [
    "retry_with_backoff",
    "sanitize_filename",
    "generate_correlation_id",
    "mask_sensitive_data",
]


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


def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to prevent directory traversal and unsafe characters.

    Args:
        filename: Candidate filename provided by an upstream service.

    Returns:
        Safe filename compatible with local filesystem storage.
    """

    original = filename
    safe = filename.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe)
    safe = safe.strip("._") or "ontology"
    if len(safe) > 255:
        safe = safe[:255]
    if safe != original:
        logging.getLogger("DocsToKG.OntologyDownload").warning(
            "sanitized unsafe filename",
            extra={"stage": "sanitize", "original": original, "sanitized": safe},
        )
    return safe


def generate_correlation_id() -> str:
    """Create a short-lived identifier that links related log entries."""

    return uuid.uuid4().hex[:12]


def mask_sensitive_data(payload: Dict[str, object]) -> Dict[str, object]:
    """Remove secrets from structured payloads prior to logging."""

    sensitive_keys = {"authorization", "api_key", "apikey", "token", "secret", "password"}
    masked: Dict[str, object] = {}
    for key, value in payload.items():
        lower = key.lower()
        if lower in sensitive_keys:
            masked[key] = "***masked***"
        elif isinstance(value, str) and "apikey" in value.lower():
            masked[key] = "***masked***"
        else:
            masked[key] = value
    return masked

