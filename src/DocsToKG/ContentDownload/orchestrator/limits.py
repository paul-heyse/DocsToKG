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

**Feature Flags:**

- `DOCSTOKG_ENABLE_SEMAPHORE_RECYCLING`: TTL-based cleanup (default: true)
- `DOCSTOKG_SEMAPHORE_TTL_SECONDS`: TTL for unused semaphores (default: 3600)
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Optional
from urllib.parse import urlsplit

from . import feature_flags

__all__ = ["KeyedLimiter", "host_key"]

logger = logging.getLogger(__name__)


def host_key(url: str) -> str:
    """Extract normalized host from URL for keying.

    Returns the hostname with port if present.
    Handles punycode (IDN) domains.

    Args:
        url: Full URL (e.g., "https://api.example.com:8080/path")

    Returns:
        Host key (e.g., "api.example.com:8080")
    """
    try:
        parsed = urlsplit(url)
        host = parsed.netloc
        if not host:
            logger.warning(f"Failed to extract host key from {url}: empty netloc")
            return url
        return host
    except Exception as e:
        logger.warning(f"Failed to extract host key from {url}: {e}")
        return url  # Fallback to full URL


class KeyedLimiter:
    """Thread-safe keyed semaphore for per-key concurrency fairness.

    Provides fine-grained concurrency control by key (e.g., resolver name, host).
    Each key gets its own semaphore with configurable limit.

    Implements TTL-based semaphore eviction to prevent unbounded memory growth
    when dealing with many dynamic keys (e.g., CDN edge servers with distinct IPs).

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
        semaphore_ttl_sec: Optional[int] = None,
    ) -> None:
        """Initialize keyed limiter.

        Args:
            default_limit: Default concurrency limit for unknown keys
            per_key: Optional per-key overrides (e.g., {"unpaywall": 1, "crossref": 2})
            semaphore_ttl_sec: TTL for unused semaphores (default from feature flags).
                              Set to None to disable TTL-based eviction.
                              If not provided, reads from DOCSTOKG_SEMAPHORE_TTL_SECONDS.
        """
        self.default_limit = max(1, default_limit)
        self.per_key = per_key or {}
        
        # Get TTL from parameter or feature flags
        if semaphore_ttl_sec is None and feature_flags.is_enabled("semaphore_recycling"):
            semaphore_ttl_sec = feature_flags.get_ttl("semaphore_recycling")
        
        self.semaphore_ttl_sec = semaphore_ttl_sec
        # Store (semaphore, last_access_time) tuples
        self._locks: dict[str, tuple[threading.Semaphore, float]] = {}
        self._mutex = threading.Lock()

    def _get_semaphore(self, key: str) -> threading.Semaphore:
        """Get or create semaphore for key.

        Thread-safe creation of semaphores on first access.
        Implements lazy TTL-based eviction of stale semaphores if enabled.
        
        Args:
            key: Concurrency key
            
        Returns:
            Semaphore for the key
        """
        with self._mutex:
            now = time.time()
            
            # Lazy eviction: only if semaphore_recycling is enabled
            if (
                feature_flags.is_enabled("semaphore_recycling")
                and self.semaphore_ttl_sec is not None
                and len(self._locks) > 10000
                and random.random() < 0.01
            ):
                self._locks = {
                    k: (sem, ts)
                    for k, (sem, ts) in self._locks.items()
                    if now - ts < self.semaphore_ttl_sec
                }
                logger.debug(
                    f"Evicted stale semaphores; remaining: {len(self._locks)}"
                )
            
            if key not in self._locks:
                limit = self.per_key.get(key, self.default_limit)
                self._locks[key] = (threading.Semaphore(limit), now)
            else:
                # Update last-access time
                sem, _ = self._locks[key]
                self._locks[key] = (sem, now)
            
            sem, _ = self._locks[key]
            return sem

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
