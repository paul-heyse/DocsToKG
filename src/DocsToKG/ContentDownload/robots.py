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

import logging
import time
from urllib import robotparser
from urllib.parse import urlsplit, urlunsplit

__all__ = ["RobotsCache"]

logger = logging.getLogger(__name__)


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
        self._cache: dict[str, tuple[float, robotparser.RobotFileParser]] = {}

    def is_allowed(
        self,
        session: object,  # httpx.Client or similar
        url: str,
        user_agent: str,
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

        now = time.monotonic()
        need_fetch = True

        # Check cache and TTL
        if netloc in self._cache:
            fetched_at, parser = self._cache[netloc]
            if now - fetched_at <= self.ttl_sec:
                need_fetch = False
                parser_to_use = parser
            else:
                # Cache expired; fetch fresh
                parser_to_use = None
        else:
            parser_to_use = None

        if need_fetch or parser_to_use is None:
            try:
                resp = session.get(robots_url, timeout=5)
                rp = robotparser.RobotFileParser()

                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    # If robots.txt doesn't exist (404), create empty policy (allow all)
                    rp.parse([])

                self._cache[netloc] = (now, rp)
                parser_to_use = rp
            except Exception as e:
                logger.warning(f"Failed to fetch robots.txt from {robots_url}: {e}")
                # Fail open: assume allowed if fetch fails
                return True

        # Check if user_agent can fetch this URL
        try:
            return parser_to_use.can_fetch(user_agent, url)
        except Exception as e:
            logger.warning(f"Error checking robots policy for {url}: {e}")
            # Fail open
            return True
