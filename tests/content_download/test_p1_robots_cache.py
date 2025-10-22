"""Tests for P1 Observability & Integrity: Robots.txt Cache & Enforcement (Phase 3).

Covers:
- RobotsCache initialization and TTL behavior
- is_allowed() with allowed/disallowed URLs
- Cache expiration and refresh
- Fail-open semantics (errors don't block requests)
- Integration with shared networking + telemetry primitives
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import pytest

from DocsToKG.ContentDownload.robots import RobotsCache
from DocsToKG.ContentDownload.telemetry import (
    ATTEMPT_REASON_ROBOTS,
    ATTEMPT_STATUS_ROBOTS_DISALLOWED,
    SimplifiedAttemptRecord,
)


class MockResponse:
    """Lightweight stand-in for :class:`httpx.Response`."""

    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.headers: Dict[str, str] = {}


class RequestStub:
    """Patch :func:`request_with_retries` and supply canned responses."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._responses: Dict[str, List[Any]] = {}
        self.calls: List[tuple[object, str, str, Dict[str, Any]]] = []
        monkeypatch.setattr(
            "DocsToKG.ContentDownload.robots.request_with_retries", self
        )

    def set_response(self, url: str, response: Union[Any, List[Any]]) -> None:
        if isinstance(response, list):
            self._responses[url] = list(response)
        else:
            self._responses[url] = [response]

    def __call__(self, session: object, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append((session, method, url, kwargs))
        queue = self._responses.get(url)
        if queue:
            resp = queue.pop(0) if queue else None
        else:
            resp = None

        if resp is None:
            resp = MockResponse(status_code=404, text="")

        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            resp = resp()

        return resp


@pytest.fixture
def request_stub(monkeypatch: pytest.MonkeyPatch) -> RequestStub:
    """Provide a request stub patched into the robots cache module."""

    return RequestStub(monkeypatch)


class FakeTelemetry:
    """Collect `SimplifiedAttemptRecord` instances for assertions."""

    def __init__(self) -> None:
        self.records: List[SimplifiedAttemptRecord] = []

    def log_io_attempt(self, record: SimplifiedAttemptRecord) -> None:
        self.records.append(record)


class TestRobotsCacheInitialization:
    """Tests for RobotsCache initialization."""

    def test_default_ttl_is_3600_seconds(self) -> None:
        cache = RobotsCache()
        assert cache.ttl_sec == 3600

    def test_custom_ttl(self) -> None:
        cache = RobotsCache(ttl_sec=7200)
        assert cache.ttl_sec == 7200

    def test_cache_starts_empty(self) -> None:
        cache = RobotsCache()
        assert len(cache._cache) == 0


class TestRobotsCacheAllowed:
    """Tests for is_allowed() with allowed URLs."""

    def test_allowed_url_returns_true(self, request_stub: RequestStub) -> None:
        robots_txt = """User-agent: *\nAllow: /\n"""
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

    def test_disallowed_url_returns_false(self, request_stub: RequestStub) -> None:
        robots_txt = """User-agent: *\nDisallow: /\n"""
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        assert not cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

    def test_specific_path_disallowed(self, request_stub: RequestStub) -> None:
        robots_txt = """User-agent: *\nDisallow: /admin/\nAllow: /\n"""
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        assert not cache.is_allowed(
            object(), "https://example.com/admin/page", "MyBot/1.0"
        )
        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

    def test_empty_robots_txt_allows_all(self, request_stub: RequestStub) -> None:
        request_stub.set_response("https://example.com/robots.txt", MockResponse(200, ""))
        cache = RobotsCache()

        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")


class TestRobotsCacheFailOpen:
    """Tests for fail-open semantics (errors don't block requests)."""

    def test_missing_robots_txt_allows_all(self, request_stub: RequestStub) -> None:
        cache = RobotsCache()

        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

    def test_exception_on_fetch_allows_url(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt", Exception("Network error")
        )
        cache = RobotsCache()

        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

    def test_invalid_robots_txt_allows_all(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt",
            MockResponse(200, "INVALID:::CONTENT:::FORMAT"),
        )
        cache = RobotsCache()

        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")


class TestRobotsCacheCaching:
    """Tests for per-host cache behavior."""

    def test_cache_stores_parsed_robots(self, request_stub: RequestStub) -> None:
        robots_txt = """User-agent: *\nDisallow: /admin/\n"""
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        cache.is_allowed(object(), "https://example.com/page1", "MyBot/1.0")
        assert len(cache._cache) == 1

        cache.is_allowed(object(), "https://example.com/page2", "MyBot/1.0")
        assert len(request_stub.calls) == 1

    def test_different_hosts_cached_separately(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt",
            MockResponse(200, "User-agent: *\nDisallow: /\n"),
        )
        request_stub.set_response(
            "https://other.com/robots.txt",
            MockResponse(200, "User-agent: *\nAllow: /\n"),
        )

        cache = RobotsCache()

        assert not cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")
        assert cache.is_allowed(object(), "https://other.com/page", "MyBot/1.0")
        assert len(cache._cache) == 2


class TestRobotsCacheUserAgent:
    """Tests for user-agent specific rules."""

    def test_user_agent_specific_rules(self, request_stub: RequestStub) -> None:
        robots_txt = (
            "User-agent: Googlebot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
        )
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        assert not cache.is_allowed(object(), "https://example.com/page", "Googlebot")

        cache._cache.clear()
        assert cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")


class TestRobotsCacheNetworking:
    """Tests for networking behavior."""

    def test_robots_txt_fetched_from_correct_path(
        self, request_stub: RequestStub
    ) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt",
            MockResponse(200, "User-agent: *\nDisallow: /\n"),
        )
        cache = RobotsCache()
        cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

        assert any(call[2].endswith("/robots.txt") for call in request_stub.calls)

    def test_timeout_parameter_passed(self, request_stub: RequestStub) -> None:
        cache = RobotsCache()
        cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")

        assert request_stub.calls[0][3]["timeout"] == 5.0


class TestRobotsCacheEdgeCases:
    """Tests for edge cases."""

    def test_url_without_scheme(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, "User-agent: *\nAllow: /\n")
        )
        cache = RobotsCache()
        cache.is_allowed(object(), "example.com/page", "MyBot/1.0")

    def test_url_with_port(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com:8443/robots.txt",
            MockResponse(200, "User-agent: *\nDisallow: /admin/\n"),
        )
        cache = RobotsCache()

        assert not cache.is_allowed(
            object(), "https://example.com:8443/admin/page", "MyBot/1.0"
        )

    def test_multiple_disallow_rules(self, request_stub: RequestStub) -> None:
        robots_txt = """User-agent: *\nDisallow: /admin/\nDisallow: /private/\nDisallow: /secret/\nAllow: /\n"""
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        assert not cache.is_allowed(object(), "https://example.com/admin/page", "MyBot/1.0")
        assert not cache.is_allowed(object(), "https://example.com/private/file", "MyBot/1.0")
        assert not cache.is_allowed(object(), "https://example.com/secret/data", "MyBot/1.0")
        assert cache.is_allowed(object(), "https://example.com/public/page", "MyBot/1.0")


class TestRobotsCacheInheritance:
    """Tests for path inheritance in robots.txt rules."""

    def test_disallow_root_blocks_all(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt",
            MockResponse(200, "User-agent: *\nDisallow: /\n"),
        )
        cache = RobotsCache()

        assert not cache.is_allowed(object(), "https://example.com/", "MyBot/1.0")
        assert not cache.is_allowed(object(), "https://example.com/page", "MyBot/1.0")
        assert not cache.is_allowed(
            object(), "https://example.com/deep/path/file", "MyBot/1.0"
        )

    def test_allow_overrides_parent_disallow(self, request_stub: RequestStub) -> None:
        robots_txt = """User-agent: *\nDisallow: /\nAllow: /public/\n"""
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, robots_txt)
        )
        cache = RobotsCache()

        assert not cache.is_allowed(object(), "https://example.com/private/", "MyBot/1.0")
        assert cache.is_allowed(object(), "https://example.com/public/page", "MyBot/1.0")


class TestRobotsTelemetry:
    """Telemetry emission when robots enforcement blocks a URL."""

    def test_allowed_does_not_emit(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, "User-agent: *\nAllow: /\n")
        )
        cache = RobotsCache()
        telemetry = FakeTelemetry()

        allowed = cache.is_allowed(
            object(),
            "https://example.com/page",
            "MyBot/1.0",
            telemetry=telemetry,
            run_id="run-123",
            resolver="landing",
        )

        assert allowed is True
        assert telemetry.records == []

    def test_blocked_emits_record(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt", MockResponse(200, "User-agent: *\nDisallow: /blocked\n")
        )
        cache = RobotsCache()
        telemetry = FakeTelemetry()

        allowed = cache.is_allowed(
            object(),
            "https://example.com/blocked",
            "MyBot/1.0",
            telemetry=telemetry,
            run_id="run-123",
            resolver="landing",
        )

        assert allowed is False
        assert len(telemetry.records) == 1
        record = telemetry.records[0]
        assert isinstance(record, SimplifiedAttemptRecord)
        assert record.status == ATTEMPT_STATUS_ROBOTS_DISALLOWED
        assert record.reason == ATTEMPT_REASON_ROBOTS
        assert record.url == "https://example.com/blocked"
        assert record.run_id == "run-123"
        assert record.resolver == "landing"
        assert record.extra["robots_url"] == "https://example.com/robots.txt"
        assert record.extra["cache"] == "miss"

        # Second call hits cache and reports cache-hit in telemetry extra metadata
        telemetry.records.clear()
        cache.is_allowed(
            object(),
            "https://example.com/blocked",
            "MyBot/1.0",
            telemetry=telemetry,
            run_id="run-123",
            resolver="landing",
        )
        assert telemetry.records[0].extra["cache"] == "hit"

    def test_fail_open_does_not_emit(self, request_stub: RequestStub) -> None:
        request_stub.set_response(
            "https://example.com/robots.txt", Exception("boom")
        )
        cache = RobotsCache()
        telemetry = FakeTelemetry()

        allowed = cache.is_allowed(
            object(),
            "https://example.com/page",
            "MyBot/1.0",
            telemetry=telemetry,
            run_id="run-123",
            resolver="landing",
        )

        assert allowed is True
        assert telemetry.records == []
