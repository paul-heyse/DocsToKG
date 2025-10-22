# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.robots",
#   "purpose": "Robots.txt cache and enforcement for respectful web crawling",
#   "sections": [
#     {"id": "robotsCache", "name": "RobotsCache", "anchor": "class-robotscache", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""Robots.txt cache and enforcement (P1 Observability & Integrity).

**Purpose**
-----------
This module provides a production-grade, thread-safe cache for robots.txt policies,
enabling respectful web crawling by checking URL permissions before landing page
fetches. Per P1 scope, robots guards prevent prohibited requests and emit telemetry.

**Responsibilities**
--------------------
- Cache parsed robots.txt files per hostname with configurable TTL
- Respect robots.txt Disallow/Allow rules before landing page fetches
- Provide fail-open semantics (errors don't block requests)
- Thread-safe operations with minimal contention
- Support configurable User-Agent strings for site-specific rules

**Key Classes**
---------------

:class:`RobotsCache`
  Thread-safe per-hostname robots.txt cache with TTL expiration.
  Methods: is_allowed(session, url, user_agent) → bool

**Integration Points**
----------------------
- Called from resolvers.landing_page.py::LandingPageResolver.iter_urls()
- Pre-fetch check before GET requests to landing URLs
- Emits telemetry via ROBOTS_DISALLOWED when request is blocked
- Uses request_with_retries() for HTTP operations

**Safety & Reliability**
------------------------
- **Fail-open**: If robots.txt fetch fails, request is allowed (conservative)
- **Parser isolation**: Separate RobotFileParser per host (no cross-contamination)
- **TTL enforcement**: Stale policies are refetched automatically
- **Thread-safe**: Uses monotonic time for cache expiration checks
- **Timeout protection**: 5-second timeout per robots.txt fetch

**Performance**
---------------
- Default 3600s (1 hour) TTL balances freshness and cache hit rate
- LRU-style cache (dict) holds parsed policies per netloc
- Lazy parsing: RobotFileParser created only on fetch
- Low memory: ~1-2 KB per cached policy

**Design Pattern**
------------------
This module follows P1 principles:
- **Non-breaking**: Pluggable into existing resolvers
- **Observable**: Emits ROBOTS_DISALLOWED events for telemetry
- **Configurable**: TTL, User-Agent, and enabled/disabled via config
- **Simple interface**: Single is_allowed() method for URL checking

**Fail-Open Semantics**
-----------------------
If any error occurs (network, parsing, timeout):
- Return True (allowed)
- Log warning with error details
- Continue request processing

This conservative approach ensures operational resilience.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib import robotparser
from urllib.parse import urlsplit, urlunsplit

from DocsToKG.ContentDownload.networking import request_with_retries

__all__ = ["RobotsCache"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CacheEntry:
    """Cached robots.txt policy metadata."""

    fetched_at: float
    parser: robotparser.RobotFileParser
    status_code: Optional[int]
    elapsed_ms: Optional[int]


class RobotsCache:
    """Thread-safe cache of robots.txt policies per hostname.

    This class fetches and caches parsed robots.txt files, one per hostname,
    with configurable TTL to ensure freshness. It respects robots.txt Disallow
    and Allow rules before landing-page fetches in the ContentDownload pipeline.

    **Caching behavior:**
    - Policies are cached per netloc (e.g., "example.com:443")
    - Each cached entry includes (fetch_time, RobotFileParser)
    - Expired entries are refetched on next access
    - Cache is never pre-cleared; only TTL determines freshness

    **Fail-open semantics:**
    - If robots.txt unavailable (404): assume allowed (empty disallow)
    - If fetch timeout or error: assume allowed (conservative)
    - If parsing error: assume allowed (preserve functionality)
    - These choices prioritize functionality over strict compliance

    Attributes:
        ttl_sec: Cache TTL in seconds (default 3600 = 1 hour).
                Higher values reduce network requests; lower values ensure fresher policies.
    """

    def __init__(self, ttl_sec: int = 3600) -> None:
        """Initialize the robots cache.

        Args:
            ttl_sec: Time-to-live for cached policies in seconds.
                    Default 3600 (1 hour) balances freshness and cache hit rate.
                    Set lower for frequently-changing robots.txt;
                    set higher for stable content sources.
        """
        self.ttl_sec = ttl_sec
        self._cache: Dict[str, _CacheEntry] = {}
        self._cache_lock = threading.RLock()
        self._host_locks: Dict[str, threading.Lock] = {}

    def is_allowed(
        self,
        session: object,  # httpx.Client or similar
        url: str,
        user_agent: str,
        *,
        telemetry: Optional[Any] = None,
        run_id: Optional[str] = None,
        resolver: Optional[str] = None,
    ) -> bool:
        """Check if the URL is allowed by the robots.txt policy.

        Fetches and caches robots.txt for the URL's hostname. Returns True
        if robots.txt allows the request, or if robots.txt is unavailable
        (fail-open behavior).

        **Flow:**
        1. Extract hostname from URL
        2. Check cache (TTL validation)
        3. If expired/missing, fetch robots.txt for hostname
        4. Parse robots.txt into RobotFileParser
        5. Check URL against policy for user_agent
        6. Return allow/disallow decision

        **Error handling:**
        - 404 responses: treated as empty robots.txt (allow all)
        - Network errors: treated as unavailable (allow all)
        - Parsing errors: treated as invalid (allow all)
        - Timeout: treated as unavailable (allow all)

        Args:
            session: HTTP client with a `get()` method (e.g., httpx.Client).
                    Must support timeout parameter.
            url: Full URL (with scheme and path) to check, e.g.,
                 "https://example.com/news/article.html"
            user_agent: User-Agent string to use when checking the policy.
                       Must match robots.txt User-agent directives exactly
                       (case-insensitive per RFC, but provided as-is to robotparser).

        Returns:
            True if allowed or if robots.txt unavailable (fail-open).
            False if robots.txt explicitly disallows the URL for user_agent.

        Examples:
            >>> cache = RobotsCache(ttl_sec=3600)
            >>> import httpx
            >>> with httpx.Client() as client:
            ...     # Check if we can fetch landing page
            ...     if cache.is_allowed(client, "https://example.com/article", "MyBot/1.0"):
            ...         # Safe to fetch landing page
            ...         resp = client.get("https://example.com/article")
            ...     else:
            ...         # Blocked by robots.txt
            ...         logger.info("Skipped URL due to robots.txt")
        """
        parts = urlsplit(url)
        netloc = parts.netloc

        # Build robots.txt URL for this host
        robots_url = urlunsplit((parts.scheme, netloc, "/robots.txt", "", ""))

        entry, cache_hit = self._lookup_entry(session, netloc, robots_url)
        if entry is None:
            return True
        parser_to_use = entry.parser

        # Check if user_agent can fetch this URL
        try:
            allowed = parser_to_use.can_fetch(user_agent, url)
        except Exception as e:
            logger.warning(f"Error checking robots policy for {url}: {e}")
            # Fail open
            return True

        if not allowed:
            self._emit_disallowed_telemetry(
                url=url,
                robots_url=robots_url,
                user_agent=user_agent,
                telemetry=telemetry,
                run_id=run_id,
                resolver=resolver,
                http_status=entry.status_code,
                elapsed_ms=entry.elapsed_ms,
                cache_hit=cache_hit,
            )

        return allowed

    def _lookup_entry(
        self,
        session: object,
        netloc: str,
        robots_url: str,
    ) -> Tuple[Optional[_CacheEntry], bool]:
        """Return a cached entry, fetching when expired."""

        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(netloc)
        if entry is not None and self._is_entry_fresh(entry, now):
            return entry, True

        host_lock = self._get_host_lock(netloc)
        with host_lock:
            refreshed_now = time.monotonic()
            with self._cache_lock:
                entry = self._cache.get(netloc)
            if entry is not None and self._is_entry_fresh(entry, refreshed_now):
                return entry, True

            fetched_entry = self._fetch_entry(session, robots_url)
            if fetched_entry is None:
                with self._cache_lock:
                    self._cache.pop(netloc, None)
                return None, False

            with self._cache_lock:
                self._cache[netloc] = fetched_entry
            return fetched_entry, False

    def _is_entry_fresh(self, entry: _CacheEntry, now: float) -> bool:
        if self.ttl_sec <= 0:
            return False
        return (now - entry.fetched_at) <= self.ttl_sec

    def _get_host_lock(self, netloc: str) -> threading.Lock:
        with self._cache_lock:
            lock = self._host_locks.get(netloc)
            if lock is None:
                lock = threading.Lock()
                self._host_locks[netloc] = lock
            return lock

    def _fetch_entry(
        self,
        session: object,
        robots_url: str,
    ) -> Optional[_CacheEntry]:
        start = time.perf_counter()
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)

        try:
            response_cm = request_with_retries(
                session,
                "GET",
                robots_url,
                role="metadata",
                timeout=5.0,
                allow_redirects=True,
                max_retries=1,
                max_retry_duration=5.0,
                backoff_max=5.0,
                retry_after_cap=5.0,
                original_url=robots_url,
            )
        except Exception as exc:  # pragma: no cover - fail-open path exercised elsewhere
            logger.warning(f"Failed to fetch robots.txt from {robots_url}: {exc}")
            return None

        if hasattr(response_cm, "__enter__") and hasattr(response_cm, "__exit__"):
            response_ctx = response_cm  # type: ignore[assignment]
        else:
            response_ctx = contextlib.nullcontext(response_cm)

        status_code: Optional[int] = None
        try:
            with response_ctx as response:
                status_code = getattr(response, "status_code", None)
                text = getattr(response, "text", "") or ""

                if status_code == 200:
                    lines = text.splitlines()
                elif status_code == 404:
                    lines = []
                elif status_code is not None and status_code >= 400:
                    lines = []
                else:
                    lines = text.splitlines()

                try:
                    parser.parse(lines)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse robots.txt from %s: %s", robots_url, exc
                    )
                    parser.parse([])
        except Exception as exc:
            logger.warning(f"Failed to fetch robots.txt from {robots_url}: {exc}")
            return None

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return _CacheEntry(
            fetched_at=time.monotonic(),
            parser=parser,
            status_code=status_code,
            elapsed_ms=elapsed_ms,
        )

    def _emit_disallowed_telemetry(
        self,
        *,
        url: str,
        robots_url: str,
        user_agent: str,
        telemetry: Optional[Any],
        run_id: Optional[str],
        resolver: Optional[str],
        http_status: Optional[int],
        elapsed_ms: Optional[int],
        cache_hit: bool,
    ) -> None:
        from DocsToKG.ContentDownload.telemetry import (
            ATTEMPT_REASON_ROBOTS,
            ATTEMPT_STATUS_ROBOTS_DISALLOWED,
            SimplifiedAttemptRecord,
        )

        if telemetry is None or not hasattr(telemetry, "log_io_attempt"):
            return

        extra = {
            "robots_url": robots_url,
            "user_agent": user_agent,
            "cache": "hit" if cache_hit else "miss",
        }

        record = SimplifiedAttemptRecord(
            ts=datetime.now(timezone.utc),
            run_id=run_id,
            resolver=resolver,
            url=url,
            verb="ROBOTS",
            status=ATTEMPT_STATUS_ROBOTS_DISALLOWED,
            http_status=http_status,
            content_type=None,
            reason=ATTEMPT_REASON_ROBOTS,
            elapsed_ms=elapsed_ms,
            bytes_written=None,
            content_length_hdr=None,
            extra=extra,
        )

        try:
            telemetry.log_io_attempt(record)
        except Exception:  # pragma: no cover - telemetry failures shouldn't crash
            logger.exception("Failed to emit robots telemetry for %s", url)
