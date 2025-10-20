"""HTTPX MockTransport-based coverage for the streaming downloader."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from DocsToKG.OntologyDownload.errors import PolicyError
from DocsToKG.OntologyDownload.io import download_stream
from DocsToKG.OntologyDownload.settings import DownloadConfiguration
from DocsToKG.OntologyDownload.testing import use_mock_http_client


def _logger(name: str = "ontology-httpx-test") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def test_download_stream_emits_progress_with_mock_transport(tmp_path, caplog):
    """Progress telemetry should surface when bytes cross the configured threshold."""

    payload = b"x" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
                "ETag": '"progress-etag"',
            },
            content=payload,
        )

    transport = httpx.MockTransport(handler)
    config = DownloadConfiguration()
    config.perform_head_precheck = False
    config.progress_log_bytes_threshold = 16

    cache_dir = tmp_path / "cache"
    destination = tmp_path / "progress.owl"
    caplog.set_level(logging.INFO)

    with use_mock_http_client(transport, default_config=config):
        result = download_stream(
            url="https://example.org/progress.owl",
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=cache_dir,
            logger=_logger("ontology-progress"),
            expected_media_type="application/rdf+xml",
            service="test",
        )

    assert result.status in {"fresh", "updated"}
    assert destination.exists()
    progress_logs = [
        record for record in caplog.records if record.message == "download progress"
    ]
    assert progress_logs, "Expected progress telemetry when bytes threshold crossed"
    assert any(
        getattr(record, "progress", {}).get("bytes_downloaded", 0) >= 32
        for record in progress_logs
    )


def test_download_stream_short_circuits_on_304(tmp_path, caplog):
    """Conditional requests should mark downloads as cached without fetching the body."""

    payload = b"<ontology/>"
    url = "https://example.org/cached.owl"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / "cached.owl"
    config = DownloadConfiguration()
    config.perform_head_precheck = False

    etag = '"cached-etag"'
    last_modified = "Wed, 21 Oct 2015 07:28:00 GMT"

    def initial_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
                "ETag": etag,
                "Last-Modified": last_modified,
            },
            content=payload,
        )

    with use_mock_http_client(httpx.MockTransport(initial_handler), default_config=config):
        first_result = download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=None,
            http_config=config,
            cache_dir=cache_dir,
            logger=_logger("ontology-cache-initial"),
            expected_media_type="application/rdf+xml",
            service="test",
        )

    previous_manifest = {
        "etag": first_result.etag,
        "last_modified": first_result.last_modified,
        "content_type": first_result.content_type,
        "content_length": first_result.content_length,
        "sha256": first_result.sha256,
    }

    def cached_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("If-None-Match") == etag
        response_headers = {"ETag": etag}
        if last_modified:
            response_headers["Last-Modified"] = last_modified
        return httpx.Response(304, headers=response_headers, request=request)

    caplog.set_level(logging.INFO)
    with use_mock_http_client(httpx.MockTransport(cached_handler), default_config=config):
        cached_result = download_stream(
            url=url,
            destination=destination,
            headers={},
            previous_manifest=previous_manifest,
            http_config=config,
            cache_dir=cache_dir,
            logger=_logger("ontology-cache-second"),
            expected_media_type="application/rdf+xml",
            service="test",
            expected_hash=f"sha256:{first_result.sha256}",
        )

    assert cached_result.status == "cached"
    assert cached_result.sha256 == first_result.sha256
    assert any(record.message == "cache hit" for record in caplog.records)


def test_download_stream_enforces_size_cap(tmp_path):
    """Streaming should raise PolicyError when content exceeds configured limits."""

    payload = b"x" * 2048

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/rdf+xml",
                "Content-Length": str(len(payload)),
            },
            content=payload,
        )

    transport = httpx.MockTransport(handler)
    config = DownloadConfiguration()
    config.perform_head_precheck = False
    config.max_uncompressed_size_gb = 0.0000005  # ~524 bytes

    cache_dir = tmp_path / "cache"
    destination = tmp_path / "oversized.owl"

    with use_mock_http_client(transport, default_config=config):
        with pytest.raises(PolicyError):
            download_stream(
                url="https://example.org/oversized.owl",
                destination=destination,
                headers={},
                previous_manifest=None,
                http_config=config,
                cache_dir=cache_dir,
                logger=_logger("ontology-oversized"),
                expected_media_type="application/rdf+xml",
                service="test",
            )
