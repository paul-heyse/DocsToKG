from __future__ import annotations

import logging
from pathlib import Path

import httpx

from DocsToKG.OntologyDownload import net
from DocsToKG.OntologyDownload.settings import DownloadConfiguration
from DocsToKG.OntologyDownload.testing import use_mock_http_client


def _config(user_agent: str = "NetTest/1.0") -> DownloadConfiguration:
    config = DownloadConfiguration()
    config.polite_headers = {"User-Agent": user_agent}
    return config


def test_get_http_client_singleton():
    records: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        records.append(request)
        return httpx.Response(200, content=b"ok")

    config = _config()
    transport = httpx.MockTransport(handler)
    with use_mock_http_client(transport, default_config=config) as client:
        client_a = net.get_http_client()
        client_b = net.get_http_client()
        assert client_a is client_b is client
        assert records == []  # no request issued yet


def test_request_hook_applies_polite_headers():
    captured_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = {key: value for key, value in request.headers.items()}
        return httpx.Response(200, content=b"payload")

    config = _config(user_agent="NetTest/2.0")
    transport = httpx.MockTransport(handler)
    with use_mock_http_client(transport, default_config=config) as client:
        request = client.build_request("GET", "https://example.org/data")
        request.extensions["ontology_headers"] = {
            "config": config,
            "headers": {"X-Correlation-ID": "abc123"},
            "correlation_id": "abc123",
        }
        client.send(request).read()

    assert captured_headers["user-agent"] == "NetTest/2.0"
    assert captured_headers["x-correlation-id"] == "abc123"


def test_cache_hits_marked_in_extensions(tmp_path: Path):
    cache_hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        if cache_hits:
            cache_hits.append("revalidate")
            return httpx.Response(304, headers={"ETag": "abc"})
        cache_hits.append("initial")
        return httpx.Response(
            200,
            headers={"ETag": "abc", "Cache-Control": "max-age=60"},
            content=b"cached-body",
        )

    config = _config()
    transport = httpx.MockTransport(handler)
    with use_mock_http_client(transport, default_config=config) as client:
        def _send() -> httpx.Response:
            request = client.build_request("GET", "https://example.org/cache")
            request.extensions["ontology_headers"] = {"config": config, "headers": {}, "correlation_id": None}
            response = client.send(request, stream=True)
            response.read()
            return response

        first = _send()
        second = _send()

    assert cache_hits == ["initial", "revalidate"]
    cache_status = second.extensions.get("ontology_cache_status", {})
    assert cache_status.get("from_cache") is True
    assert cache_status.get("revalidated") in {True, None}
