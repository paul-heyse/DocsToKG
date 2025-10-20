"""HTTPX/Hishel networking regression tests for ContentDownload."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from pyrate_limiter import Duration, Rate

from DocsToKG.ContentDownload import httpx_transport
from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from DocsToKG.ContentDownload.download import (
    DownloadContext,
    DownloadOutcome,
    RobotsCache,
    download_candidate,
)
from DocsToKG.ContentDownload.errors import RateLimitError
from DocsToKG.ContentDownload.networking import ConditionalRequestHelper, request_with_retries
from DocsToKG.ContentDownload.ratelimit import (
    RolePolicy,
    clone_policies,
    configure_rate_limits,
    get_rate_limiter_manager,
)


def _response(
    status: int, *, headers: dict[str, str] | None = None, body: bytes = b""
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.org/resource")
    return httpx.Response(status, headers=headers, content=body, request=request)


def test_conditional_request_builds_headers_when_metadata_complete() -> None:
    helper = ConditionalRequestHelper(
        prior_etag='"etag"',
        prior_last_modified="Wed, 01 May 2024 00:00:00 GMT",
        prior_sha256="sha",
        prior_content_length=1024,
        prior_path="/tmp/cached.pdf",
    )

    headers = helper.build_headers()

    assert headers["If-None-Match"] == '"etag"'
    assert headers["If-Modified-Since"] == "Wed, 01 May 2024 00:00:00 GMT"


def test_conditional_request_requires_complete_metadata(caplog) -> None:
    helper = ConditionalRequestHelper(
        prior_etag='"etag"',
        prior_last_modified="Wed, 01 May 2024 00:00:00 GMT",
        prior_sha256=None,
        prior_content_length=None,
        prior_path=None,
    )

    with caplog.at_level("WARNING"):
        headers = helper.build_headers()

    assert headers == {}
    assert any("resume-metadata-incomplete" in record.message for record in caplog.records)


def test_conditional_request_interprets_cached_response(tmp_path: Path) -> None:
    cached_path = tmp_path / "archive.pdf"
    cached_body = b"cached-pdf"
    cached_path.write_bytes(cached_body)
    helper = ConditionalRequestHelper(
        prior_etag='"etag"',
        prior_last_modified="Wed, 01 May 2024 00:00:00 GMT",
        prior_sha256=hashlib.sha256(cached_body).hexdigest(),
        prior_content_length=len(cached_body),
        prior_path=str(cached_path),
        prior_mtime_ns=os.stat(cached_path).st_mtime_ns,
    )

    result = helper.interpret_response(_response(304))

    assert result.path == str(cached_path)
    assert result.sha256 == hashlib.sha256(cached_body).hexdigest()
    assert result.content_length == len(cached_body)
    assert result.etag == '"etag"'


def test_retry_after_controls_sleep(install_mock_http_client, capture_sleep) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, request=request)

    install_mock_http_client(handler)

    response = request_with_retries(
        None,
        "GET",
        "https://example.org/resource",
        max_retries=2,
        backoff_factor=0.0,
        backoff_max=0.0,
        respect_retry_after=True,
    )

    assert response.status_code == 200
    assert capture_sleep == [2.0]
    assert len(calls) == 2


def test_streaming_download_writes_atomically(tmp_path: Path, install_mock_http_client) -> None:
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()

    artifact = WorkArtifact(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/archive.pdf"],
        open_access_url=None,
        source_display_names=["test"],
        base_stem="example",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        xml_dir=xml_dir,
    )

    chunks = [b"%PDF-1.4\n", b"1 0 obj\n", b"endobj\n", b"%%EOF\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("archive.pdf")
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            stream=iter(chunks),
            request=request,
        )

    client = install_mock_http_client(handler)

    context = DownloadContext()

    stream_result = download_candidate(
        client,
        artifact,
        artifact.pdf_urls[0],
        referer=None,
        timeout=10.0,
        context=context,
    )

    assert isinstance(stream_result, DownloadOutcome)
    assert stream_result.classification is Classification.PDF
    assert stream_result.path is not None

    output_path = Path(stream_result.path)
    assert output_path.exists()
    assert output_path.read_bytes() == b"".join(chunks)
    assert not pdf_dir.joinpath("example.pdf.part").exists()


def test_robots_cache_reuses_shared_client(install_mock_http_client) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = b"User-agent: TestBot\nDisallow: /forbidden"
        return httpx.Response(200, content=body, request=request)

    client = install_mock_http_client(handler)
    cache = RobotsCache(user_agent="TestBot", ttl_seconds=60)

    assert not cache.is_allowed(client, "https://example.org/forbidden", timeout=2.0)
    assert not cache.is_allowed(client, "https://example.org/forbidden", timeout=2.0)
    assert calls == 1


def test_request_with_retries_does_not_retry_rate_limit_error() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise RateLimitError("blocked", host="example.org", role="metadata")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(RateLimitError):
        request_with_retries(
            client,
            "GET",
            "https://example.org/resource",
            max_retries=5,
            backoff_factor=0.0,
            backoff_max=0.0,
        )

    client.close()
    assert attempts == 1


def test_cache_hits_do_not_consume_rate_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_mock_http_client
) -> None:
    manager = get_rate_limiter_manager()
    original_policies = clone_policies(manager.policies())
    original_backend = manager.backend

    policy = RolePolicy(
        rates={"metadata": [Rate(5, Duration.SECOND)]},
        max_delay_ms={"metadata": 0},
        mode={"metadata": "raise"},
        count_head={"metadata": False},
        weight={"metadata": 1},
    )

    monkeypatch.setenv("DOCSTOKG_DATA_ROOT", str(tmp_path))
    httpx_transport.purge_http_cache()
    manager.reset_metrics()
    configure_rate_limits(policies={"example.org": policy})

    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        now = datetime.now(timezone.utc)
        headers = {
            "Cache-Control": "public, max-age=60",
            "Date": format_datetime(now),
            "Expires": format_datetime(now + timedelta(seconds=60)),
        }
        return httpx.Response(
            200,
            headers=headers,
            request=request,
            content=b"payload",
        )

    client = install_mock_http_client(handler)
    url = "https://example.org/resource"

    response1 = client.get(url)
    assert response1.status_code == 200
    assert request_count == 1

    snapshot1 = manager.metrics_snapshot()
    assert snapshot1["example.org"]["metadata"]["acquire_total"] == 1

    response2 = client.get(url)
    assert response2.status_code == 200
    assert request_count == 1  # cache hit should bypass inner transport

    cache_flag = response2.extensions.get("from_cache")
    if cache_flag is None:
        cache_flag = response2.request.extensions.get("docs_network_meta", {}).get("from_cache")
    assert cache_flag is True

    snapshot2 = manager.metrics_snapshot()
    assert snapshot2["example.org"]["metadata"]["acquire_total"] == 1

    configure_rate_limits(policies=original_policies, backend=original_backend)
    manager.reset_metrics()
