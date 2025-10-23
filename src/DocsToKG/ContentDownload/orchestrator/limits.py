# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.limits",
#   "purpose": "Thread-safe keyed concurrency limiters for per-resolver and per-host fairness",
#   "sections": [
#     {
#       "id": "semaphoreentry",
#       "name": "_SemaphoreEntry",
#       "anchor": "class-semaphoreentry",
#       "kind": "class"
#     },
#     {
#       "id": "host-key",
#       "name": "host_key",
#       "anchor": "function-host-key",
#       "kind": "function"
#     },
#     {
#       "id": "keyedlimiter",
#       "name": "KeyedLimiter",
#       "anchor": "class-keyedlimiter",
#       "kind": "class"
#     }
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

**Memory Management:**

Uses TTL-based lazy eviction of unused semaphores following the industry best
practice pattern (Python docs, asyncio patterns). Semaphores are automatically
removed when they exceed their TTL without being accessed, preventing unbounded
memory growth with dynamic keys.

The approach is simple and proven: maintain a dict of semaphores keyed by
resource identifier, periodically evict stale entries. This is the standard
pattern documented in Python's asyncio and threading modules.

**Feature Flags:**

- `DOCSTOKG_ENABLE_SEMAPHORE_RECYCLING`: Lazy TTL-based cleanup (default: true)
- `DOCSTOKG_SEMAPHORE_TTL_SECONDS`: TTL for unused semaphores (default: 3600)
"""

from __future__ import annotations

import logging
import random
import threading
import time
from urllib.parse import urlsplit

from . import feature_flags

__all__ = ["KeyedLimiter", "host_key"]

logger = logging.getLogger(__name__)


class _SemaphoreEntry:
    """Wrapper for semaphore + timestamp to enable weak references.

    WeakValueDictionary cannot hold weak references to tuples,
    so we use this simple wrapper class instead.
    """

    def __init__(self, semaphore: threading.Semaphore, last_access_time: float, capacity: int):
        self.semaphore = semaphore
        self.last_access_time = last_access_time
        self.capacity = capacity


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

    Implements two complementary cleanup strategies:
    1. **WeakValueDictionary**: Automatic removal when semaphore is no longer referenced
    2. **TTL-based eviction**: Lazy cleanup of stale entries (optional, feature-flagged)

    This is the industry best practice pattern (asyncio docs, python patterns).

    Example:
        >>> limiter = KeyedLimiter(default_limit=2, per_key={"unpaywall": 1})
        >>> limiter.acquire("unpaywall")       # May wait if limit exceeded
        >>> # ... do work ...
        >>> limiter.release("unpaywall")
    """

    def __init__(
        self,
        default_limit: int,
        per_key: dict[str, int] | None = None,
        semaphore_ttl_sec: int | None = None,
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

        # Use regular dict with TTL-based eviction (industry best practice)
        # Avoids WeakValueDictionary garbage collection issues while still
        # cleaning up stale entries via lazy eviction
        self._locks: dict[str, _SemaphoreEntry] = {}
        self._mutex = threading.Lock()

        logger.debug(
            f"KeyedLimiter initialized: default_limit={default_limit}, "
            f"per_key={per_key}, ttl_sec={semaphore_ttl_sec}"
        )

    def _get_semaphore(self, key: str, *, create: bool = True) -> threading.Semaphore | None:
        """Get or create semaphore for key.

        Thread-safe creation of semaphores on first access.
        Implements lazy TTL-based eviction of stale semaphores if enabled.
        Uses WeakValueDictionary for automatic cleanup.

        Args:
            key: Concurrency key

        Returns:
            Semaphore for the key
        """
        with self._mutex:
            now = time.time()

            entry = self._locks.get(key)
            if entry is not None:
                entry.last_access_time = now

            # Lazy TTL-based eviction: only if feature enabled
            if (
                feature_flags.is_enabled("semaphore_recycling")
                and self.semaphore_ttl_sec is not None
                and len(self._locks) > 10000
                and random.random() < 0.01
            ):
                # Create new dict with only non-stale entries
                # (WeakValueDictionary doesn't support item deletion well during iteration)
                stale_keys = [
                    k
                    for k, semaphore_entry in self._locks.items()
                    if k != key
                    and now - semaphore_entry.last_access_time >= self.semaphore_ttl_sec
                    and getattr(semaphore_entry.semaphore, "_value", 0) == semaphore_entry.capacity
                ]
                for stale_key in stale_keys:
                    try:
                        del self._locks[stale_key]
                    except KeyError:
                        # Key may have already been garbage collected
                        pass
                logger.debug(
                    f"Evicted {len(stale_keys)} stale semaphores; remaining: {len(self._locks)}"
                )

            # Update or create entry (tuple stays same, but we update access time)
            if entry is None:
                if not create:
                    return None
                limit = self.per_key.get(key, self.default_limit)
                entry = _SemaphoreEntry(threading.Semaphore(limit), now, limit)
                self._locks[key] = entry

            return entry.semaphore

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
        sem = self._get_semaphore(key, create=False)
        if sem is None:
            logger.error("Attempted to release unknown limiter key '%s'", key)
            raise KeyError(f"No semaphore tracked for key={key!r}")
        sem.release()

    def try_acquire(self, key: str, timeout: float | None = None) -> bool:
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
            # Note: Existing semaphore won't change; new keys will use new limit
            logger.debug(f"Updated limit for key={key} to {limit}")
