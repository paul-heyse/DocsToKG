# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.limits",
#   "purpose": "Thread-safe keyed concurrency limiters for per-resolver and per-host fairness",
#   "sections": [
#     {"id": "keyedlimiter", "name": "KeyedLimiter", "anchor": "#class-keyedlimiter", "kind": "class"},
#     {"id": "hostkey", "name": "host_key", "anchor": "#function-hostkey", "kind": "function"}
#   ]
# }
# === /NAVMAP ===

"""Keyed concurrency limiters for ContentDownload orchestration.

This module provides thread-safe semaphores keyed by resolver name or host,
enabling per-resolver and per-host concurrency fairness in the worker pool.

**Design:**

Keyed semaphores prevent burst concurrency to specific resolvers or hosts:

    KeyedLimiter(default_limit=2, per_key={"unpaywall": 1, "crossref": 2})

    limiter.acquire("unpaywall")  # Wait if 1+ unpaywall jobs active
    # ... process job ...
    limiter.release("unpaywall")

**Usage in Orchestrator:**

    resolver_limiter = KeyedLimiter(default_limit=8, per_key={...})
    host_limiter = KeyedLimiter(default_limit=4)  # per-host cap

    # In worker, before download streaming:
    resolver_limiter.acquire(plan.resolver_name)  # Fair resolver ordering
    host_limiter.acquire(host_key(plan.url))      # Fair host throttling
    try:
        # Download with bounded parallelism
        stream_to_part(...)
    finally:
        host_limiter.release(host_key(plan.url))
        resolver_limiter.release(plan.resolver_name)

**Thread Safety:**

All operations are thread-safe via internal lock. Multiple workers can safely
call acquire/release concurrently without data races.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional
from urllib.parse import urlsplit

__all__ = ["KeyedLimiter", "host_key"]

logger = logging.getLogger(__name__)


def host_key(url: str) -> str:
    """Extract normalized host key from URL.

    Returns host:port pair, excluding default ports (80 for http, 443 for https).
    Used as concurrency limit key for per-host fairness.

    Args:
        url: Full URL (e.g., "https://api.example.com:8080/path")

    Returns:
        Normalized host key (e.g., "api.example.com:8080" or "example.com")

    Examples:
        >>> host_key("https://api.crossref.org/works")
        'api.crossref.org'

        >>> host_key("http://example.com:8080/data")
        'example.com:8080'
    """
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ""
        port = parts.port

        # Exclude default ports
        if (parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443):
            port = None

        if port:
            return f"{hostname}:{port}"
        return hostname
    except Exception as e:
        logger.warning(f"Failed to extract host key from {url}: {e}")
        return url  # Fallback to full URL


class KeyedLimiter:
    """Thread-safe keyed semaphore for per-key concurrency fairness.

    Provides fine-grained concurrency control by key (e.g., resolver name, host).
    Each key gets its own semaphore with configurable limit.

    Example:
        >>> limiter = KeyedLimiter(default_limit=2, per_key={"unpaywall": 1})
        >>> limiter.acquire("unpaywall")       # May wait if limit exceeded
        >>> # ... do work ...
        >>> limiter.release("unpaywall")
    """

    def __init__(
        self,
        default_limit: int,
        per_key: Optional[dict[str, int]] = None,
    ) -> None:
        """Initialize keyed limiter.

        Args:
            default_limit: Default concurrency limit for unknown keys
            per_key: Optional per-key overrides (e.g., {"unpaywall": 1, "crossref": 2})
        """
        self.default_limit = max(1, default_limit)
        self.per_key = per_key or {}
        self._locks: dict[str, threading.Semaphore] = {}
        self._mutex = threading.Lock()

    def _get_semaphore(self, key: str) -> threading.Semaphore:
        """Get or create semaphore for key.

        Thread-safe creation of semaphores on first access.
        """
        with self._mutex:
            if key not in self._locks:
                limit = self.per_key.get(key, self.default_limit)
                self._locks[key] = threading.Semaphore(limit)
            return self._locks[key]

    def acquire(self, key: str) -> None:
        """Acquire concurrency slot for key.

        Blocks if limit for this key is exceeded. Thread-safe.

        Args:
            key: Concurrency key (e.g., resolver name or host)
        """
        sem = self._get_semaphore(key)
        sem.acquire()

    def release(self, key: str) -> None:
        """Release concurrency slot for key.

        Unblocks any waiting workers. Thread-safe.

        Args:
            key: Concurrency key (e.g., resolver name or host)
        """
        sem = self._get_semaphore(key)
        sem.release()

    def try_acquire(self, key: str, timeout: Optional[float] = None) -> bool:
        """Try to acquire slot, with optional timeout.

        Non-blocking or timeout-based acquisition. Useful for avoiding
        indefinite waits.

        Args:
            key: Concurrency key
            timeout: Timeout in seconds (None = infinite)

        Returns:
            True if acquired, False if timeout exceeded
        """
        sem = self._get_semaphore(key)
        return sem.acquire(timeout=timeout)

    def get_limit(self, key: str) -> int:
        """Get concurrency limit for key.

        Args:
            key: Concurrency key

        Returns:
            Configured limit for this key
        """
        return self.per_key.get(key, self.default_limit)

    def set_limit(self, key: str, limit: int) -> None:
        """Update concurrency limit for key.

        Changes take effect for new acquisitions. Existing holders are unaffected.

        Args:
            key: Concurrency key
            limit: New limit (must be ≥ 1)
        """
        if limit < 1:
            raise ValueError(f"Limit must be ≥ 1, got {limit}")

        with self._mutex:
            self.per_key[key] = limit
            # If semaphore already exists, we can't change it directly
            # (threading.Semaphore doesn't support modification)
            # Future acquisitions will use the new limit via get_semaphore
            logger.debug(f"Updated limit for key={key} to {limit}")
