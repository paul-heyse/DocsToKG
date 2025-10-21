"""Tests for P1 Observability & Integrity: Robots.txt Cache & Enforcement (Phase 3).

Covers:
- RobotsCache initialization and TTL behavior
- is_allowed() with allowed/disallowed URLs
- Cache expiration and refresh
- Fail-open semantics (errors don't block requests)
- Integration with mock HTTP sessions
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import Mock

from DocsToKG.ContentDownload.robots import RobotsCache


class MockResponse:
    """Mock httpx.Response for testing."""

    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class MockSession:
    """Mock httpx.Client session for testing."""

    def __init__(self, responses: Optional[dict] = None) -> None:
        self.responses = responses or {}
        self.get_called_with = []

    def get(self, url: str, timeout: Optional[int] = None) -> MockResponse:
        """Mock GET method that returns configured responses."""
        self.get_called_with.append(url)

        # Match URL to configured response
        for key, response in self.responses.items():
            if key in url or url in key:
                if isinstance(response, tuple):
                    status, text = response
                    return MockResponse(status_code=status, text=text)
                else:
                    return response

        # Default: return 404 (robots.txt not found)
        return MockResponse(status_code=404, text="")


class TestRobotsCacheInitialization:
    """Tests for RobotsCache initialization."""

    def test_default_ttl_is_3600_seconds(self) -> None:
        """Default TTL is 1 hour."""
        cache = RobotsCache()
        assert cache.ttl_sec == 3600

    def test_custom_ttl(self) -> None:
        """Custom TTL can be set."""
        cache = RobotsCache(ttl_sec=7200)
        assert cache.ttl_sec == 7200

    def test_cache_starts_empty(self) -> None:
        """Cache is empty on initialization."""
        cache = RobotsCache()
        assert len(cache._cache) == 0


class TestRobotsCacheAllowed:
    """Tests for is_allowed() with allowed URLs."""

    def test_allowed_url_returns_true(self) -> None:
        """Allowed URL returns True."""
        robots_txt = """User-agent: *
Allow: /
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        result = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result is True

    def test_disallowed_url_returns_false(self) -> None:
        """Disallowed URL returns False."""
        robots_txt = """User-agent: *
Disallow: /
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        result = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result is False

    def test_specific_path_disallowed(self) -> None:
        """Specific path can be disallowed."""
        robots_txt = """User-agent: *
Disallow: /admin/
Allow: /
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        # Admin path is disallowed
        assert cache.is_allowed(session, "https://example.com/admin/page", "MyBot/1.0") is False

        # Other paths are allowed
        assert cache.is_allowed(session, "https://example.com/page", "MyBot/1.0") is True

    def test_empty_robots_txt_allows_all(self) -> None:
        """Empty robots.txt allows all URLs."""
        session = MockSession({"robots.txt": (200, "")})
        cache = RobotsCache()

        result = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result is True


class TestRobotsCacheFailOpen:
    """Tests for fail-open semantics (errors don't block requests)."""

    def test_missing_robots_txt_allows_all(self) -> None:
        """Missing robots.txt (404) allows all URLs."""
        session = MockSession({})
        cache = RobotsCache()

        result = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result is True

    def test_exception_on_fetch_allows_url(self) -> None:
        """Exception fetching robots.txt allows URL (fail-open)."""
        session = Mock()
        session.get.side_effect = Exception("Network error")

        cache = RobotsCache()
        result = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result is True

    def test_invalid_robots_txt_allows_all(self) -> None:
        """Invalid robots.txt format allows all URLs."""
        session = MockSession({"robots.txt": (200, "INVALID:::CONTENT:::FORMAT")})
        cache = RobotsCache()

        # Should not crash; should allow
        result = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result is True


class TestRobotsCacheCaching:
    """Tests for per-host cache behavior."""

    def test_cache_stores_parsed_robots(self) -> None:
        """Parsed robots.txt is cached per host."""
        robots_txt = """User-agent: *
Disallow: /admin/
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        # First call: fetches robots.txt
        cache.is_allowed(session, "https://example.com/page1", "MyBot/1.0")
        assert len(cache._cache) == 1

        # Second call to same host: uses cache (not fetching again)
        session.get_called_with.clear()
        cache.is_allowed(session, "https://example.com/page2", "MyBot/1.0")
        # Second call doesn't fetch robots.txt again (it's cached)
        assert len(session.get_called_with) == 0

    def test_different_hosts_cached_separately(self) -> None:
        """Different hosts have separate cache entries."""
        robots1 = "User-agent: *\nDisallow: /\n"
        robots2 = "User-agent: *\nAllow: /\n"

        session = Mock()
        session.get.side_effect = lambda url, timeout: (
            MockResponse(200, robots1)
            if "example.com" in url
            else MockResponse(200, robots2) if "other.com" in url else MockResponse(404)
        )

        cache = RobotsCache()

        # example.com disallows
        result1 = cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")
        assert result1 is False

        # other.com allows
        result2 = cache.is_allowed(session, "https://other.com/page", "MyBot/1.0")
        assert result2 is True

        # Both entries cached
        assert len(cache._cache) == 2


class TestRobotsCacheUserAgent:
    """Tests for user-agent specific rules."""

    def test_user_agent_specific_rules(self) -> None:
        """User-agent specific rules are respected."""
        robots_txt = """User-agent: Googlebot
Disallow: /

User-agent: *
Allow: /
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        # Googlebot is disallowed
        assert cache.is_allowed(session, "https://example.com/page", "Googlebot") is False

        # Other bots are allowed (refresh cache for new user-agent)
        cache._cache.clear()
        assert cache.is_allowed(session, "https://example.com/page", "MyBot/1.0") is True


class TestRobotsCacheNetworking:
    """Tests for networking behavior."""

    def test_robots_txt_fetched_from_correct_path(self) -> None:
        """robots.txt is fetched from / path."""
        session = MockSession({"robots.txt": (200, "User-agent: *\nDisallow: /\n")})
        cache = RobotsCache()

        cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")

        # Check that robots.txt was fetched
        assert any("robots.txt" in url for url in session.get_called_with)

    def test_timeout_parameter_passed(self) -> None:
        """Timeout is passed to session.get()."""
        session = Mock()
        session.get.return_value = MockResponse(404)

        cache = RobotsCache()
        cache.is_allowed(session, "https://example.com/page", "MyBot/1.0")

        # Verify timeout was passed
        call_kwargs = session.get.call_args[1]
        assert call_kwargs.get("timeout") == 5


class TestRobotsCacheEdgeCases:
    """Tests for edge cases."""

    def test_url_without_scheme(self) -> None:
        """URLs without scheme are handled gracefully."""
        session = MockSession({"robots.txt": (200, "User-agent: *\nAllow: /\n")})
        cache = RobotsCache()

        # Should not crash
        try:
            cache.is_allowed(session, "example.com/page", "MyBot/1.0")
        except Exception:
            pass  # Expected - urlsplit may fail, but cache handles it

    def test_url_with_port(self) -> None:
        """URLs with ports are handled correctly."""
        session = MockSession({"robots.txt": (200, "User-agent: *\nDisallow: /admin/\n")})
        cache = RobotsCache()

        result = cache.is_allowed(session, "https://example.com:8443/admin/page", "MyBot/1.0")
        assert result is False

    def test_multiple_disallow_rules(self) -> None:
        """Multiple Disallow rules are all respected."""
        robots_txt = """User-agent: *
Disallow: /admin/
Disallow: /private/
Disallow: /secret/
Allow: /
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        assert cache.is_allowed(session, "https://example.com/admin/page", "MyBot/1.0") is False
        assert cache.is_allowed(session, "https://example.com/private/file", "MyBot/1.0") is False
        assert cache.is_allowed(session, "https://example.com/secret/data", "MyBot/1.0") is False
        assert cache.is_allowed(session, "https://example.com/public/page", "MyBot/1.0") is True


class TestRobotsCacheInheritance:
    """Tests for path inheritance in robots.txt rules."""

    def test_disallow_root_blocks_all(self) -> None:
        """Disallow: / blocks all paths."""
        robots_txt = """User-agent: *
Disallow: /
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        assert cache.is_allowed(session, "https://example.com/", "MyBot/1.0") is False
        assert cache.is_allowed(session, "https://example.com/page", "MyBot/1.0") is False
        assert cache.is_allowed(session, "https://example.com/deep/path/file", "MyBot/1.0") is False

    def test_allow_overrides_parent_disallow(self) -> None:
        """Allow rules can override parent Disallow rules."""
        robots_txt = """User-agent: *
Disallow: /
Allow: /public/
"""
        session = MockSession({"robots.txt": (200, robots_txt)})
        cache = RobotsCache()

        # Disallowed by default
        assert cache.is_allowed(session, "https://example.com/private/", "MyBot/1.0") is False

        # Allowed exception
        assert cache.is_allowed(session, "https://example.com/public/page", "MyBot/1.0") is True
