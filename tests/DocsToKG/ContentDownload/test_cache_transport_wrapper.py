"""Tests for role-aware cache transport TTL enforcement."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpcore

from DocsToKG.ContentDownload.cache_policy import CacheDecision
from DocsToKG.ContentDownload.cache_transport_wrapper import RoleAwareCacheTransport


def _make_request() -> httpcore.Request:
    return httpcore.Request(
        method=b"GET",
        url="https://example.org/resource",
        headers=[(b"accept", b"application/json")],
        extensions={},
    )


def _set_date_header(response: httpcore.Response, *, age_seconds: int) -> None:
    """Mutate the response Date header to simulate cached age."""
    target_time = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    http_date = format_datetime(target_time, usegmt=True).encode("ascii")
    filtered = [(name, value) for name, value in response.headers if name.lower() != b"date"]
    filtered.append((b"date", http_date))
    response.headers = filtered


def test_policy_controller_injects_cache_control_and_date() -> None:
    decision = CacheDecision(use_cache=True, ttl_s=60, swrv_s=30)
    controller = RoleAwareCacheTransport._build_controller_for_decision(decision)

    request = _make_request()
    response = httpcore.Response(status=200, headers=[], content=b"{}")

    assert controller.is_cachable(request, response)

    cache_control_values = [
        value.decode("latin-1")
        for name, value in response.headers
        if name.lower() == b"cache-control"
    ]

    assert cache_control_values, "Cache-Control header should be injected"
    cache_control = ", ".join(cache_control_values)
    assert "max-age=60" in cache_control
    assert "stale-while-revalidate=30" in cache_control
    assert any(name.lower() == b"date" for name, _ in response.headers), "Date header should be present"


def test_policy_controller_construct_response_respects_ttl() -> None:
    decision = CacheDecision(use_cache=True, ttl_s=60)
    controller = RoleAwareCacheTransport._build_controller_for_decision(decision)

    request = _make_request()
    response = httpcore.Response(status=200, headers=[], content=b"{}")

    controller.is_cachable(request, response)

    fresh_response = httpcore.Response(status=200, headers=list(response.headers), content=b"{}")
    _set_date_header(fresh_response, age_seconds=30)

    result = controller.construct_response_from_cache(request, fresh_response, request)
    assert isinstance(result, httpcore.Response)

    stale_response = httpcore.Response(status=200, headers=list(response.headers), content=b"{}")
    _set_date_header(stale_response, age_seconds=120)

    stale_result = controller.construct_response_from_cache(request, stale_response, request)
    assert isinstance(stale_result, httpcore.Request)

