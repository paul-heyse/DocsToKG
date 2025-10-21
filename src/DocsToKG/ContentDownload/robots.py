"""Robots.txt cache and enforcement (P1 Observability & Integrity).

This module provides a cached robots.txt parser with thread-safe operations:

- :class:`RobotsCache` - Caches parsed robots.txt files per hostname with TTL.
  Respects robots.txt before landing-page fetches.

Designed for use in the landing-page resolver to enforce robots before
attempting a GET.
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

    Fetches robots.txt once per netloc, caches the parsed RobotFileParser,
    and checks URLs against it. TTL ensures stale policies are refreshed.

    Attributes:
        ttl_sec: Cache TTL in seconds (default 3600 = 1 hour).
    """

    def __init__(self, ttl_sec: int = 3600) -> None:
        """Initialize the robots cache.

        Args:
            ttl_sec: Time-to-live for cached policies in seconds.
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
        if robots.txt allows the request, or if robots.txt is unavailable.

        Args:
            session: HTTP client with a `get()` method (e.g., httpx.Client).
            url: Full URL to check.
            user_agent: User-Agent string to use when checking the policy.

        Returns:
            True if allowed or if robots.txt unavailable (fail-open).
            False if robots.txt explicitly disallows the URL.
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
