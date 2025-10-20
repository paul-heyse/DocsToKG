# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_net",
#   "purpose": "Validates the shared HTTPX client, request/response hooks, and Hishel-backed caching transport helpers.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Validates the shared HTTPX client, request/response hooks, and Hishel-backed caching transport helpers."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from hishel import CacheTransport, FileStorage

from DocsToKG.OntologyDownload import net
from DocsToKG.OntologyDownload.settings import DownloadConfiguration
from DocsToKG.OntologyDownload.testing import use_mock_http_client
from tests.conftest import PatchManager


def _config(user_agent: str = "NetTest/1.0") -> DownloadConfiguration:
    config = DownloadConfiguration()
    config.polite_headers = {"User-Agent": user_agent}
    config.http2_enabled = False  # avoid h2 dependency warnings during tests
    return config


def test_get_http_client_singleton():
    records: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        records.append(request)
        return httpx.Response(200, content=b"ok")

    config = _config()
    transport = httpx.MockTransport(handler)
    event_hooks = {"request": [net._request_hook], "response": [net._response_hook]}
    with use_mock_http_client(
        transport,
        default_config=config,
        event_hooks=event_hooks,
        http2=False,
        trust_env=True,
    ) as client:
        client_a = net.get_http_client()
        client_b = net.get_http_client()
        assert client_a is client_b is client
        assert records == []  # no request issued yet


def test_configure_http_client_swap_and_reset(tmp_path: Path):
    patch_manager = PatchManager()
    patch_manager.setattr(net, "HTTP_CACHE_DIR", tmp_path / "http-cache", raising=False)
    net.reset_http_client()

    custom_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        custom_calls.append(str(request.url))
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    custom_client = httpx.Client(
        transport=transport,
        event_hooks={"request": [net._request_hook], "response": [net._response_hook]},
    )
    config = _config()
    config.polite_headers["X-Test"] = "swap"

    try:
        net.configure_http_client(client=custom_client, default_config=config)
        assert net.get_http_client() is custom_client
        response = custom_client.get("https://example.org/swap")
        response.read()
        response.close()
        assert custom_calls == ["https://example.org/swap"]
    finally:
        net.reset_http_client()
        patch_manager.close()

    rebuilt = net.get_http_client(config)
    try:
        assert rebuilt is not custom_client
        assert rebuilt.timeout.connect == pytest.approx(config.connect_timeout_sec)
    finally:
        net.reset_http_client()


def test_request_hook_applies_polite_headers():
    captured_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = {key: value for key, value in request.headers.items()}
        return httpx.Response(200, content=b"payload")

    config = _config(user_agent="NetTest/2.0")
    transport = httpx.MockTransport(handler)
    event_hooks = {"request": [net._request_hook], "response": [net._response_hook]}
    with use_mock_http_client(
        transport,
        default_config=config,
        event_hooks=event_hooks,
        http2=False,
        trust_env=True,
    ) as client:
        request = client.build_request("GET", "https://example.org/data")
        request.extensions["ontology_headers"] = {
            "config": config,
            "headers": {"X-Correlation-ID": "abc123"},
            "correlation_id": "abc123",
        }
        client.send(request).read()

    assert captured_headers["user-agent"] == "NetTest/2.0"
    assert captured_headers["x-correlation-id"] == "abc123"


def test_http_client_respects_configuration_limits(tmp_path: Path):
    patch_manager = PatchManager()
    patch_manager.setattr(net, "HTTP_CACHE_DIR", tmp_path / "http-cache", raising=False)
    net.reset_http_client()

    config = _config()
    config.connect_timeout_sec = 1.25
    config.pool_timeout_sec = 2.75
    config.timeout_sec = 45
    config.max_httpx_connections = 7
    config.max_keepalive_connections = 3
    config.keepalive_expiry_sec = 14.0

    try:
        client = net.get_http_client(config)
        assert client.timeout.connect == pytest.approx(1.25)
        assert client.timeout.pool == pytest.approx(2.75)
        assert client.timeout.read == 45
        transport = client._transport._transport  # CacheTransport -> HTTPTransport
        pool = transport._pool
        assert pool._max_connections == 7
        assert pool._max_keepalive_connections == 3
        assert pool._keepalive_expiry == pytest.approx(14.0)
    finally:
        net.reset_http_client()
        patch_manager.close()


def test_http_client_records_cache_hits(tmp_path: Path):
    call_log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append(str(request.url))
        return httpx.Response(
            200,
            headers={"Cache-Control": "max-age=60", "ETag": "abc123"},
            content=b"cached-body",
        )

    cache_root = tmp_path / "cache"
    transport = CacheTransport(
        transport=httpx.MockTransport(handler),
        storage=FileStorage(base_path=cache_root),
        controller=net._controller(),
    )
    event_hooks = {"request": [net._request_hook], "response": [net._response_hook]}
    config = _config()

    with use_mock_http_client(
        transport,
        default_config=config,
        event_hooks=event_hooks,
        http2=False,
        trust_env=True,
    ) as client:
        first = client.get("https://example.org/cache")
        first.read()
        first.close()

        second = client.get("https://example.org/cache")
        second.read()
        cache_status = second.extensions.get("ontology_cache_status")
        second.close()

    assert cache_root.exists()
    assert call_log == ["https://example.org/cache", "https://example.org/cache"]
    assert isinstance(cache_status, dict)
