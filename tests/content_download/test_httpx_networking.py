"""HTTPX/Hishel networking regression tests for ContentDownload."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx

from DocsToKG.ContentDownload.download import (
    DownloadContext,
    DownloadOutcome,
    RobotsCache,
    download_candidate,
)
from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from DocsToKG.ContentDownload.networking import ConditionalRequestHelper, request_with_retries


def _response(status: int, *, headers: dict[str, str] | None = None, body: bytes = b"") -> httpx.Response:
    request = httpx.Request("GET", "https://example.org/resource")
    return httpx.Response(status, headers=headers, content=body, request=request)


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

    chunks = [b"%PDF-1.4\n", b"1 0 obj\n", b"endobj\n"]

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
        body = b"User-agent: *\nDisallow: /forbidden"
        return httpx.Response(200, content=body, request=request)

    client = install_mock_http_client(handler)
    cache = RobotsCache(user_agent="TestBot", ttl_seconds=60)

    assert not cache.is_allowed(client, "https://example.org/forbidden", timeout=2.0)
    assert not cache.is_allowed(client, "https://example.org/forbidden", timeout=2.0)
    assert calls == 1
