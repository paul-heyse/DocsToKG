# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_download",
#   "purpose": "Pytest coverage for ontology download download scenarios",
#   "sections": [
#     {
#       "id": "dummyresponse",
#       "name": "DummyResponse",
#       "anchor": "class-dummyresponse",
#       "kind": "class"
#     },
#     {
#       "id": "dummysession",
#       "name": "DummySession",
#       "anchor": "class-dummysession",
#       "kind": "class"
#     },
#     {
#       "id": "make-session",
#       "name": "make_session",
#       "anchor": "function-make-session",
#       "kind": "function"
#     },
#     {
#       "id": "clear-token-buckets",
#       "name": "clear_token_buckets",
#       "anchor": "function-clear-token-buckets",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-success",
#       "name": "test_download_stream_success",
#       "anchor": "function-test-download-stream-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-uses-cache-on-304",
#       "name": "test_download_stream_uses_cache_on_304",
#       "anchor": "function-test-download-stream-uses-cache-on-304",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-resumes-from-partial",
#       "name": "test_download_stream_resumes_from_partial",
#       "anchor": "function-test-download-stream-resumes-from-partial",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-retries",
#       "name": "test_download_stream_retries",
#       "anchor": "function-test-download-stream-retries",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-rate-limiting",
#       "name": "test_download_stream_rate_limiting",
#       "anchor": "function-test-download-stream-rate-limiting",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-bucket-service-specific-rate",
#       "name": "test_get_bucket_service_specific_rate",
#       "anchor": "function-test-get-bucket-service-specific-rate",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-bucket-without-service-uses-host-key",
#       "name": "test_get_bucket_without_service_uses_host_key",
#       "anchor": "function-test-get-bucket-without-service-uses-host-key",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-bucket-falls-back-to-host-limit",
#       "name": "test_get_bucket_falls_back_to_host_limit",
#       "anchor": "function-test-get-bucket-falls-back-to-host-limit",
#       "kind": "function"
#     },
#     {
#       "id": "test-get-bucket-independent-keys-for-services",
#       "name": "test_get_bucket_independent_keys_for_services",
#       "anchor": "function-test-get-bucket-independent-keys-for-services",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-check-success",
#       "name": "test_head_check_success",
#       "anchor": "function-test-head-check-success",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-check-graceful-on-405",
#       "name": "test_head_check_graceful_on_405",
#       "anchor": "function-test-head-check-graceful-on-405",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-check-raises-on-oversized",
#       "name": "test_head_check_raises_on_oversized",
#       "anchor": "function-test-head-check-raises-on-oversized",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-media-type-match",
#       "name": "test_validate_media_type_match",
#       "anchor": "function-test-validate-media-type-match",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-media-type-acceptable-variation",
#       "name": "test_validate_media_type_acceptable_variation",
#       "anchor": "function-test-validate-media-type-acceptable-variation",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-media-type-mismatch-logs-warning",
#       "name": "test_validate_media_type_mismatch_logs_warning",
#       "anchor": "function-test-validate-media-type-mismatch-logs-warning",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-media-type-disabled",
#       "name": "test_validate_media_type_disabled",
#       "anchor": "function-test-validate-media-type-disabled",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-zip-safe",
#       "name": "test_extract_zip_safe",
#       "anchor": "function-test-extract-zip-safe",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-zip-rejects-traversal",
#       "name": "test_extract_zip_rejects_traversal",
#       "anchor": "function-test-extract-zip-rejects-traversal",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-zip-rejects-absolute",
#       "name": "test_extract_zip_rejects_absolute",
#       "anchor": "function-test-extract-zip-rejects-absolute",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-zip-detects-compression-bomb",
#       "name": "test_extract_zip_detects_compression_bomb",
#       "anchor": "function-test-extract-zip-detects-compression-bomb",
#       "kind": "function"
#     },
#     {
#       "id": "make-tarfile",
#       "name": "_make_tarfile",
#       "anchor": "function-make-tarfile",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-tar-safe",
#       "name": "test_extract_tar_safe",
#       "anchor": "function-test-extract-tar-safe",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-tar-rejects-traversal",
#       "name": "test_extract_tar_rejects_traversal",
#       "anchor": "function-test-extract-tar-rejects-traversal",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-tar-rejects-absolute",
#       "name": "test_extract_tar_rejects_absolute",
#       "anchor": "function-test-extract-tar-rejects-absolute",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-tar-rejects-symlink",
#       "name": "test_extract_tar_rejects_symlink",
#       "anchor": "function-test-extract-tar-rejects-symlink",
#       "kind": "function"
#     },
#     {
#       "id": "test-extract-tar-detects-compression-bomb",
#       "name": "test_extract_tar_detects_compression_bomb",
#       "anchor": "function-test-extract-tar-detects-compression-bomb",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-http-error",
#       "name": "test_download_stream_http_error",
#       "anchor": "function-test-download-stream-http-error",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-no-space",
#       "name": "test_download_stream_no_space",
#       "anchor": "function-test-download-stream-no-space",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-hash-mismatch-triggers-retry",
#       "name": "test_download_stream_hash_mismatch_triggers_retry",
#       "anchor": "function-test-download-stream-hash-mismatch-triggers-retry",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-rejects-private-ip",
#       "name": "test_validate_url_security_rejects_private_ip",
#       "anchor": "function-test-validate-url-security-rejects-private-ip",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-upgrades-http",
#       "name": "test_validate_url_security_upgrades_http",
#       "anchor": "function-test-validate-url-security-upgrades-http",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-respects-allowlist",
#       "name": "test_validate_url_security_respects_allowlist",
#       "anchor": "function-test-validate-url-security-respects-allowlist",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-blocks-disallowed-host",
#       "name": "test_validate_url_security_blocks_disallowed_host",
#       "anchor": "function-test-validate-url-security-blocks-disallowed-host",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-normalizes-idn",
#       "name": "test_validate_url_security_normalizes_idn",
#       "anchor": "function-test-validate-url-security-normalizes-idn",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-rejects-mixed-script-idn",
#       "name": "test_validate_url_security_rejects_mixed_script_idn",
#       "anchor": "function-test-validate-url-security-rejects-mixed-script-idn",
#       "kind": "function"
#     },
#     {
#       "id": "test-validate-url-security-respects-wildcard-allowlist",
#       "name": "test_validate_url_security_respects_wildcard_allowlist",
#       "anchor": "function-test-validate-url-security-respects-wildcard-allowlist",
#       "kind": "function"
#     },
#     {
#       "id": "test-ensure-license-allowed-normalizes-spdx",
#       "name": "test_ensure_license_allowed_normalizes_spdx",
#       "anchor": "function-test-ensure-license-allowed-normalizes-spdx",
#       "kind": "function"
#     },
#     {
#       "id": "test-sanitize-filename-removes-traversal",
#       "name": "test_sanitize_filename_removes_traversal",
#       "anchor": "function-test-sanitize-filename-removes-traversal",
#       "kind": "function"
#     },
#     {
#       "id": "test-migrate-manifest-sets-default-version",
#       "name": "test_migrate_manifest_sets_default_version",
#       "anchor": "function-test-migrate-manifest-sets-default-version",
#       "kind": "function"
#     },
#     {
#       "id": "test-migrate-manifest-upgrades-old-schema",
#       "name": "test_migrate_manifest_upgrades_old_schema",
#       "anchor": "function-test-migrate-manifest-upgrades-old-schema",
#       "kind": "function"
#     },
#     {
#       "id": "test-migrate-manifest-warns-unknown-version",
#       "name": "test_migrate_manifest_warns_unknown_version",
#       "anchor": "function-test-migrate-manifest-warns-unknown-version",
#       "kind": "function"
#     },
#     {
#       "id": "test-read-manifest-applies-migration",
#       "name": "test_read_manifest_applies_migration",
#       "anchor": "function-test-read-manifest-applies-migration",
#       "kind": "function"
#     },
#     {
#       "id": "test-download-stream-rejects-large-content",
#       "name": "test_download_stream_rejects_large_content",
#       "anchor": "function-test-download-stream-rejects-large-content",
#       "kind": "function"
#     },
#     {
#       "id": "test-version-lock-serializes-concurrent-writers",
#       "name": "test_version_lock_serializes_concurrent_writers",
#       "anchor": "function-test-version-lock-serializes-concurrent-writers",
#       "kind": "function"
#     },
#     {
#       "id": "noop-logger",
#       "name": "_noop_logger",
#       "anchor": "function-noop-logger",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Ontology Download Streaming Tests

This module validates the ontology download subsystem, focusing on HTTP
streaming, caching, resume support, and archive extraction safety for the
DocsToKG ontology ingestion workflow.

Key Scenarios:
- Streams ontology payloads with resumable and cached responses
- Applies retry and rate limiting controls during transfer
- Guards against path traversal when extracting downloaded archives

Dependencies:
- pytest/requests: Network simulation and assertions
- DocsToKG.OntologyDownload.ontology_download: Streaming implementation under test

Usage:
    pytest tests/ontology_download/test_download.py
"""

import io
import json
import logging
import stat
import tarfile
import zipfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

import DocsToKG.OntologyDownload.net as download
import DocsToKG.OntologyDownload.pipeline as pipeline_mod
from DocsToKG.OntologyDownload import io_safe as io_safe_mod
from DocsToKG.OntologyDownload.config import ConfigError, DefaultsConfig, DownloadConfiguration, ResolvedConfig
from DocsToKG.OntologyDownload.errors import DownloadFailure, OntologyDownloadError, PolicyError
from DocsToKG.OntologyDownload.io_safe import sanitize_filename
from DocsToKG.OntologyDownload.pipeline import ConfigurationError, FetchSpec
from DocsToKG.OntologyDownload.resolvers import FetchPlan
from DocsToKG.OntologyDownload.storage import CACHE_DIR


@dataclass
class DummyResponse:
    status_code: int
    content: bytes
    headers: dict
    raise_error: bool = False

    def __post_init__(self):
        self.headers = dict(self.headers)
        self.request_headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - nothing to clean
        return False

    def iter_content(self, chunk_size=1024):
        stream = io.BytesIO(self.content)
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def raise_for_status(self):
        if self.raise_error:
            error = requests.HTTPError(f"status {self.status_code}")
            response = requests.Response()
            response.status_code = self.status_code
            error.response = response
            raise error


class DummySession:
    def __init__(self, queue, head_queue=None):
        self.queue = queue
        self.head_queue = list(head_queue or [])
        self.calls = []
        self.head_calls = []

    def get(self, url, *, headers=None, stream=None, timeout=None, allow_redirects=None):
        response = self.queue.pop(0)
        if isinstance(response, Exception):
            raise response
        response.request_headers = headers or {}
        self.calls.append(response.request_headers)
        return response

    def head(
        self, url, *, headers=None, timeout=None, allow_redirects=None
    ):  # pragma: no cover - exercised via downloader
        response = self.head_queue.pop(0) if self.head_queue else DummyResponse(405, b"", {})
        if isinstance(response, Exception):
            raise response
        response.request_headers = headers or {}
        self.head_calls.append(response.request_headers)
        return response

    def close(self):
        """Mimic requests.Session.close."""
        return None


def make_session(monkeypatch, responses, head_responses=None):
    session = DummySession(list(responses), head_responses)

    def _factory():
        return session

    monkeypatch.setattr(requests, "Session", _factory)
    return session


@pytest.fixture(autouse=True)
def clear_token_buckets():
    """Reset token bucket cache between tests to avoid leakage."""

    download._TOKEN_BUCKETS.clear()
    yield
    download._TOKEN_BUCKETS.clear()


# --- Test Cases ---


def test_download_stream_success(monkeypatch, tmp_path):
    content = b"ontology"
    response = DummyResponse(200, content, {"ETag": "abc", "Last-Modified": "yesterday"})
    make_session(monkeypatch, [response])
    destination = tmp_path / "file.owl"
    result = download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=DownloadConfiguration(),
        cache_dir=tmp_path / "cache",
        logger=_noop_logger(),
    )
    assert destination.read_bytes() == content
    assert result.status == "fresh"
    assert result.etag == "abc"


def test_download_stream_uses_cache_on_304(monkeypatch, tmp_path):
    destination = tmp_path / "file.owl"
    destination.write_bytes(b"cached")
    response = DummyResponse(304, b"", {"ETag": "abc"})
    session = make_session(monkeypatch, [response])
    result = download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest={"etag": "abc"},
        http_config=DownloadConfiguration(),
        cache_dir=tmp_path / "cache",
        logger=_noop_logger(),
    )
    assert result.status == "cached"
    assert session.calls[0]["If-None-Match"] == "abc"


def test_download_stream_resumes_from_partial(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    part = cache_dir / "file.owl.part"
    part.write_bytes(b"hello")
    response = DummyResponse(206, b" world", {"ETag": "abc"})
    make_session(monkeypatch, [response])
    destination = tmp_path / "file.owl"
    result = download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=DownloadConfiguration(),
        cache_dir=cache_dir,
        logger=_noop_logger(),
    )
    data = destination.read_bytes()
    assert data.endswith(b" world")
    assert result.status == "updated"


def test_streaming_downloader_handles_cached_response(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_file = cache_dir / "cache_entry"
    part_file = Path(str(output_file) + ".part")
    initial = b"partial"
    part_file.write_bytes(initial)
    destination = tmp_path / "destination.owl"
    destination.write_bytes(b"existing")
    previous_manifest = {
        "etag": "old-tag",
        "last_modified": "Wed, 01 Jan 2024 00:00:00 GMT",
    }
    response = DummyResponse(
        304,
        b"",
        {"ETag": "new-tag", "Last-Modified": "Thu, 02 Jan 2024 00:00:00 GMT"},
    )
    session = make_session(monkeypatch, [response])
    downloader = download.StreamingDownloader(
        destination=destination,
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=previous_manifest,
        logger=_noop_logger(),
    )
    downloader(
        "https://example.org/file.owl",
        output_file.as_posix(),
        logging.getLogger("pooch-test"),
    )

    assert downloader.status == "cached"
    assert downloader.response_etag == "new-tag"
    assert downloader.response_last_modified == "Thu, 02 Jan 2024 00:00:00 GMT"
    assert not part_file.exists()
    assert session.calls[0]["If-None-Match"] == "old-tag"
    assert session.calls[0]["Range"] == f"bytes={len(initial)}-"


def test_streaming_downloader_resumes_range_request(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_file = cache_dir / "resume_entry"
    part_file = Path(str(output_file) + ".part")
    initial = b"partial "
    part_file.write_bytes(initial)
    response = DummyResponse(
        206,
        b"data",
        {"ETag": "etag-206", "Last-Modified": "Fri, 03 Jan 2024 00:00:00 GMT"},
    )
    session = make_session(monkeypatch, [response])
    downloader = download.StreamingDownloader(
        destination=tmp_path / "destination.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=_noop_logger(),
    )
    downloader(
        "https://example.org/file.owl",
        output_file.as_posix(),
        logging.getLogger("pooch-test"),
    )

    assert downloader.status == "updated"
    assert downloader.response_etag == "etag-206"
    assert downloader.response_last_modified == "Fri, 03 Jan 2024 00:00:00 GMT"
    assert Path(output_file).read_bytes() == initial + b"data"
    assert not part_file.exists()
    assert session.calls[0]["Range"] == f"bytes={len(initial)}-"


def test_streaming_downloader_records_fresh_response_metadata(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_file = cache_dir / "fresh_entry"
    response = DummyResponse(
        200,
        b"fresh-data",
        {"ETag": "fresh-tag", "Last-Modified": "Sat, 04 Jan 2024 00:00:00 GMT"},
    )
    session = make_session(monkeypatch, [response])
    downloader = download.StreamingDownloader(
        destination=tmp_path / "destination.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=_noop_logger(),
    )
    downloader(
        "https://example.org/file.owl",
        output_file.as_posix(),
        logging.getLogger("pooch-test"),
    )

    assert downloader.status == "fresh"
    assert downloader.response_etag == "fresh-tag"
    assert downloader.response_last_modified == "Sat, 04 Jan 2024 00:00:00 GMT"
    assert Path(output_file).read_bytes() == b"fresh-data"
    assert session.calls[0].get("Range") is None
    assert session.calls[0].get("If-None-Match") is None


def test_download_stream_retries(monkeypatch, tmp_path):
    error_response = DummyResponse(500, b"", {}, raise_error=True)
    success_response = DummyResponse(200, b"data", {})
    make_session(monkeypatch, [error_response, success_response])
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    destination = tmp_path / "file.owl"
    result = download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=DownloadConfiguration(max_retries=2, backoff_factor=0.1),
        cache_dir=tmp_path / "cache",
        logger=_noop_logger(),
    )
    assert result.status == "fresh"
    assert destination.read_bytes() == b"data"


def test_download_stream_rate_limiting(monkeypatch, tmp_path):
    response = DummyResponse(200, b"data", {})
    make_session(monkeypatch, [response])
    consumed = []

    class StubBucket:
        def consume(self):
            consumed.append(True)

    monkeypatch.setattr(download, "_get_bucket", lambda host, config, service=None: StubBucket())
    destination = tmp_path / "file.owl"
    download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=DownloadConfiguration(),
        cache_dir=tmp_path / "cache",
        logger=_noop_logger(),
    )
    assert consumed


def test_get_bucket_service_specific_rate():
    config = DownloadConfiguration(rate_limits={"ols": "2/second"})

    bucket = download._get_bucket("ols.example.org", config, "ols")

    assert bucket.rate == pytest.approx(2.0)
    assert download._TOKEN_BUCKETS["ols:ols.example.org"] is bucket


def test_get_bucket_without_service_uses_host_key():
    config = DownloadConfiguration(per_host_rate_limit="4/second")

    bucket = download._get_bucket("example.org", config)

    assert download._TOKEN_BUCKETS["example.org"] is bucket
    assert bucket.rate == pytest.approx(config.rate_limit_per_second())


def test_get_bucket_falls_back_to_host_limit():
    config = DownloadConfiguration(per_host_rate_limit="6/second")

    bucket = download._get_bucket("obo.org", config, "unknown")

    assert bucket.rate == pytest.approx(config.rate_limit_per_second())
    assert download._TOKEN_BUCKETS["unknown:obo.org"].rate == bucket.rate


def test_get_bucket_independent_keys_for_services():
    config = DownloadConfiguration(rate_limits={"ols": "2/second", "bioportal": "1/second"})

    ols_bucket = download._get_bucket("api.example.org", config, "ols")
    bioportal_bucket = download._get_bucket("api.example.org", config, "bioportal")

    assert ols_bucket is not bioportal_bucket
    assert ols_bucket.rate == pytest.approx(2.0)
    assert bioportal_bucket.rate == pytest.approx(1.0)


def test_head_check_success(monkeypatch, tmp_path):
    head_response = DummyResponse(
        200,
        b"",
        {"Content-Type": "application/rdf+xml", "Content-Length": "1024"},
    )
    session = make_session(monkeypatch, [], head_responses=[head_response])
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=_noop_logger(),
        expected_media_type="application/rdf+xml",
    )
    content_type, content_length = downloader._preliminary_head_check(
        "https://example.org/ont.owl", session
    )

    assert content_type == "application/rdf+xml"
    assert content_length == 1024


def test_head_check_graceful_on_405(monkeypatch, tmp_path):
    head_response = DummyResponse(405, b"", {})
    session = make_session(monkeypatch, [], head_responses=[head_response])
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=_noop_logger(),
    )
    content_type, content_length = downloader._preliminary_head_check(
        "https://example.org/ont.owl", session
    )

    assert content_type is None
    assert content_length is None


def test_head_check_raises_on_oversized(monkeypatch, tmp_path):
    oversize_bytes = 6 * 1024 * 1024 * 1024
    head_response = DummyResponse(
        200,
        b"",
        {"Content-Length": str(oversize_bytes)},
    )
    session = make_session(monkeypatch, [], head_responses=[head_response])
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(max_download_size_gb=5.0),
        previous_manifest=None,
        logger=_noop_logger(),
    )
    with pytest.raises(PolicyError) as exc_info:
        downloader._preliminary_head_check("https://example.org/huge.owl", session)

    assert "exceeds limit" in str(exc_info.value)


def test_validate_media_type_match(tmp_path):
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=_noop_logger(),
        expected_media_type="application/rdf+xml",
    )

    downloader._validate_media_type(
        "application/rdf+xml",
        downloader.expected_media_type,
        "https://example.org/ont.owl",
    )


def test_validate_media_type_acceptable_variation(tmp_path, caplog):
    logger = logging.getLogger("test-download-info")
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=logger,
        expected_media_type="application/rdf+xml",
    )

    with caplog.at_level(logging.INFO, logger="test-download-info"):
        downloader._validate_media_type(
            "application/xml",
            downloader.expected_media_type,
            "https://example.org/ont.owl",
        )

    assert any("acceptable media type variation" in record.message for record in caplog.records)


def test_validate_media_type_mismatch_logs_warning(tmp_path, caplog):
    logger = logging.getLogger("test-download-warning")
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(),
        previous_manifest=None,
        logger=logger,
        expected_media_type="application/rdf+xml",
    )

    with caplog.at_level(logging.WARNING, logger="test-download-warning"):
        downloader._validate_media_type(
            "text/html",
            downloader.expected_media_type,
            "https://example.org/ont.owl",
        )

    assert any("media type mismatch" in record.message for record in caplog.records)
    assert any(getattr(record, "override_hint", None) for record in caplog.records)


def test_validate_media_type_disabled(tmp_path, caplog):
    logger = logging.getLogger("test-download-disabled")
    downloader = download.StreamingDownloader(
        destination=tmp_path / "file.owl",
        headers={},
        http_config=DownloadConfiguration(validate_media_type=False),
        previous_manifest=None,
        logger=logger,
        expected_media_type="application/rdf+xml",
    )

    with caplog.at_level(logging.WARNING, logger="test-download-disabled"):
        downloader._validate_media_type(
            "text/plain",
            downloader.expected_media_type,
            "https://example.org/ont.owl",
        )

    assert not caplog.records


def test_extract_zip_safe(tmp_path):
    archive = tmp_path / "archive.zip"
    safe_dir = tmp_path / "safe"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("folder/file.txt", "data")
    extracted = io_safe_mod.extract_zip_safe(archive, safe_dir)
    assert (safe_dir / "folder" / "file.txt").read_text() == "data"
    assert extracted


def test_extract_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "data")
    with pytest.raises(ConfigError):
        io_safe_mod.extract_zip_safe(archive, tmp_path / "out")


def test_extract_zip_rejects_absolute(tmp_path):
    archive = tmp_path / "bad_abs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/absolute/path.txt", "data")
    with pytest.raises(ConfigError):
        io_safe_mod.extract_zip_safe(archive, tmp_path / "out")


def test_extract_zip_detects_compression_bomb(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("large.txt", b"0" * (11 * 1024 * 1024))

    with pytest.raises(ConfigError) as exc_info:
        io_safe_mod.extract_zip_safe(archive, tmp_path / "zip_out")

    assert "compression ratio" in str(exc_info.value)


def test_extract_zip_rejects_symlink(tmp_path):
    archive = tmp_path / "bad_symlink.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.create_system = 3  # POSIX
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "target")

    with pytest.raises(ConfigError):
        io_safe_mod.extract_zip_safe(archive, tmp_path / "out")


# --- Helper Functions ---


def _make_tarfile(path: Path, entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for member, data in entries:
            fileobj = io.BytesIO(data) if data is not None else None
            tf.addfile(member, fileobj)


def test_extract_tar_safe(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    info = tarfile.TarInfo("folder/file.txt")
    data = b"payload"
    info.size = len(data)
    _make_tarfile(archive, [(info, data)])

    extracted = io_safe_mod.extract_tar_safe(archive, tmp_path / "tar_out")

    assert (tmp_path / "tar_out" / "folder" / "file.txt").read_bytes() == data
    assert extracted


def test_extract_tar_rejects_traversal(tmp_path):
    archive = tmp_path / "bad_traversal.tar.gz"
    info = tarfile.TarInfo("../evil.txt")
    payload = b"evil"
    info.size = len(payload)
    _make_tarfile(archive, [(info, payload)])

    with pytest.raises(ConfigError):
        io_safe_mod.extract_tar_safe(archive, tmp_path / "out")


def test_extract_tar_rejects_absolute(tmp_path):
    archive = tmp_path / "bad_absolute.tar.gz"
    info = tarfile.TarInfo("/abs/path.txt")
    data = b"absolute"
    info.size = len(data)
    _make_tarfile(archive, [(info, data)])

    with pytest.raises(ConfigError):
        io_safe_mod.extract_tar_safe(archive, tmp_path / "out")


def test_extract_tar_rejects_symlink(tmp_path):
    archive = tmp_path / "bad_symlink.tar.gz"
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = "target"
    info.size = 0
    _make_tarfile(archive, [(info, None)])

    with pytest.raises(ConfigError):
        io_safe_mod.extract_tar_safe(archive, tmp_path / "out")


def test_extract_tar_detects_compression_bomb(tmp_path):
    archive = tmp_path / "bomb.tar.gz"
    info = tarfile.TarInfo("large.bin")
    payload = b"0" * (11 * 1024 * 1024)
    info.size = len(payload)
    _make_tarfile(archive, [(info, payload)])

    with pytest.raises(ConfigError) as exc_info:
        io_safe_mod.extract_tar_safe(archive, tmp_path / "out")

    assert "compression ratio" in str(exc_info.value)


def test_download_stream_http_error(monkeypatch, tmp_path):
    response = DummyResponse(500, b"", {}, raise_error=True)
    make_session(monkeypatch, [response])
    with pytest.raises(DownloadFailure):
        download.download_stream(
            url="https://example.org/file.owl",
            destination=tmp_path / "file.owl",
            headers={},
            previous_manifest=None,
            http_config=DownloadConfiguration(max_retries=0),
            cache_dir=tmp_path / "cache",
            logger=_noop_logger(),
        )


def test_download_stream_no_space(monkeypatch, tmp_path):
    response = DummyResponse(200, b"data", {})
    make_session(monkeypatch, [response])
    original_open = Path.open

    def failing_open(self, mode="r", *args, **kwargs):
        if self.suffix == ".part":
            raise OSError("No space left on device")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OntologyDownloadError):
        download.download_stream(
            url="https://example.org/file.owl",
            destination=tmp_path / "file.owl",
            headers={},
            previous_manifest=None,
            http_config=DownloadConfiguration(),
            cache_dir=tmp_path / "cache",
            logger=_noop_logger(),
        )


def test_download_stream_hash_mismatch_triggers_retry(monkeypatch, tmp_path):
    responses = [
        DummyResponse(200, b"first", {}),
        DummyResponse(200, b"second", {}),
    ]
    session = make_session(monkeypatch, responses)
    destination = tmp_path / "file.owl"
    result = download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest={"sha256": "mismatch"},
        http_config=DownloadConfiguration(),
        cache_dir=tmp_path / "cache",
        logger=_noop_logger(),
    )
    assert len(session.calls) == 2
    assert destination.read_bytes() == b"second"
    assert result.status == "fresh"


def test_validate_url_security_rejects_private_ip():
    with pytest.raises(ConfigError):
        download.validate_url_security("https://127.0.0.1/ontology.owl")


def test_validate_url_security_upgrades_http(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()
    monkeypatch.setattr(
        io_safe_mod.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )
    secure_url = download.validate_url_security("http://example.org/ontology.owl")
    assert secure_url.startswith("https://example.org")


def test_validate_url_security_respects_allowlist(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()
    looked_up = {}

    def fake_getaddrinfo(host, *_args, **_kwargs):
        looked_up["host"] = host
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(io_safe_mod.socket, "getaddrinfo", fake_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["example.org", "purl.obolibrary.org"])

    secure_url = download.validate_url_security("https://purl.obolibrary.org/ontology.owl", config)

    assert looked_up["host"] == "purl.obolibrary.org"
    assert secure_url.startswith("https://purl.obolibrary.org")


def test_validate_url_security_blocks_disallowed_host():
    config = DownloadConfiguration(allowed_hosts=["example.org"])

    with pytest.raises(ConfigError):
        download.validate_url_security("https://malicious.com/evil.owl", config)


def test_validate_url_security_normalizes_idn(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()
    looked_up = {}

    def fake_getaddrinfo(host, *_args, **_kwargs):
        looked_up["host"] = host
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(io_safe_mod.socket, "getaddrinfo", fake_getaddrinfo)

    config = DownloadConfiguration()
    secure_url = download.validate_url_security("https://münchen.example.org/ontology.owl", config)

    assert looked_up["host"] == "xn--mnchen-3ya.example.org"
    assert secure_url.startswith("https://xn--mnchen-3ya.example.org")


def test_validate_url_security_rejects_mixed_script_idn():
    with pytest.raises(ConfigError):
        download.validate_url_security("https://раураl.com/ontology.owl")


def test_validate_url_security_respects_wildcard_allowlist(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()
    def fake_getaddrinfo(host, *_args, **_kwargs):
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(io_safe_mod.socket, "getaddrinfo", fake_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["*.example.org"])

    secure_url = download.validate_url_security("https://sub.example.org/ontology.owl", config)

    assert secure_url.startswith("https://sub.example.org")


def test_validate_url_security_rejects_userinfo() -> None:
    with pytest.raises(ConfigError):
        download.validate_url_security("https://user:secret@example.org/resource.owl")


def test_validate_url_security_enforces_default_ports() -> None:
    with pytest.raises(ConfigError):
        download.validate_url_security("https://example.org:8443/ontology.owl")


def test_validate_url_security_allows_configured_port(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()

    monkeypatch.setattr(
        io_safe_mod.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )

    config = DownloadConfiguration(allowed_ports=[8443])
    secure_url = download.validate_url_security("https://example.org:8443/ontology.owl", config)

    assert secure_url.startswith("https://example.org:8443")


def test_validate_url_security_allows_host_specific_port(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()

    monkeypatch.setattr(
        io_safe_mod.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )

    config = DownloadConfiguration(allowed_hosts=["example.org:8443"])
    secure_url = download.validate_url_security("https://example.org:8443/ontology.owl", config)

    assert secure_url.startswith("https://example.org:8443")


def test_validate_url_security_dns_lookup_cached(monkeypatch):
    io_safe_mod._DNS_CACHE.clear()
    calls = {"count": 0}

    def fake_getaddrinfo(host, *_args, **_kwargs):
        calls["count"] += 1
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(io_safe_mod.socket, "getaddrinfo", fake_getaddrinfo)

    config = DownloadConfiguration()
    url = "https://example.org/ontology.owl"

    download.validate_url_security(url, config)
    download.validate_url_security(url, config)

    assert calls["count"] == 1


def test_ensure_license_allowed_normalizes_spdx() -> None:
    config = ResolvedConfig(
        defaults=DefaultsConfig(accept_licenses=["CC-BY-4.0"]),
        specs=[],
    )
    spec = FetchSpec(id="hp", resolver="obo", extras={}, target_formats=["owl"])
    plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version=None,
        license="Creative Commons Attribution 4.0",
        media_type="application/rdf+xml",
        service="obo",
    )

    pipeline_mod._ensure_license_allowed(plan, config, spec)

    disallowed_plan = FetchPlan(
        url="https://example.org/hp.owl",
        headers={},
        filename_hint=None,
        version=None,
        license="GPL",
        media_type="application/rdf+xml",
        service="obo",
    )

    with pytest.raises(ConfigurationError):
        pipeline_mod._ensure_license_allowed(disallowed_plan, config, spec)


def test_sanitize_filename_removes_traversal():
    sanitized = sanitize_filename("../evil.owl")
    assert ".." not in sanitized
    assert sanitized.endswith("evil.owl")


def test_migrate_manifest_sets_default_version():
    payload = {}
    pipeline_mod._migrate_manifest_inplace(payload)
    assert payload["schema_version"] == "1.0"


def test_migrate_manifest_upgrades_old_schema():
    payload = {"schema_version": "0.9"}
    pipeline_mod._migrate_manifest_inplace(payload)
    assert payload["schema_version"] == "1.0"
    assert payload["resolver_attempts"] == []


def test_migrate_manifest_warns_unknown_version(caplog):
    caplog.set_level(logging.WARNING)
    payload = {"schema_version": "2.0"}
    pipeline_mod._migrate_manifest_inplace(payload)
    assert any("unknown manifest schema version" in record.message for record in caplog.records)


def test_read_manifest_applies_migration(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "0.9"}))

    observed = {}

    def _validate(payload, source=None):  # pragma: no cover - minimal validation stub
        observed["schema_version"] = payload.get("schema_version")
        observed["resolver_attempts"] = payload.get("resolver_attempts")

    monkeypatch.setattr(download, "validate_manifest_dict", _validate)

    payload = pipeline_mod._read_manifest(manifest_path)
    assert payload["schema_version"] == "1.0"
    assert payload["resolver_attempts"] == []
    assert observed == {"schema_version": "1.0", "resolver_attempts": []}


def test_download_stream_rejects_large_content(monkeypatch, tmp_path):
    response = DummyResponse(200, b"data", {"Content-Length": str(10 * 1024 * 1024 * 1024)})
    make_session(monkeypatch, [response])
    with pytest.raises(OntologyDownloadError):
        download.download_stream(
            url="https://example.org/file.owl",
            destination=tmp_path / "file.owl",
            headers={},
            previous_manifest=None,
            http_config=DownloadConfiguration(max_download_size_gb=1),
            cache_dir=tmp_path / "cache",
            logger=_noop_logger(),
        )


def test_version_lock_serializes_concurrent_writers(tmp_path):
    barrier = threading.Barrier(2)
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    completed = []

    def _worker(name: str):
        barrier.wait()
        with pipeline_mod._version_lock("hp", "2024-01-01"):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
                completed.append(name)

    threads = [threading.Thread(target=_worker, args=(f"worker-{idx}",)) for idx in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max_active"] == 1
    assert len(completed) == 2
    lock_path = CACHE_DIR / "locks" / "hp__2024-01-01.lock"
    assert lock_path.exists()
    assert lock_path.stat().st_size > 0


def _noop_logger():
    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):  # pragma: no cover - debug logging ignored in tests
            pass

    return _Logger()
