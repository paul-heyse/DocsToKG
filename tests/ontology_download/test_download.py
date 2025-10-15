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
- DocsToKG.OntologyDownload.download: Streaming implementation under test

Usage:
    pytest tests/ontology_download/test_download.py
"""

import io
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import download
from DocsToKG.OntologyDownload.config import ConfigError, DownloadConfiguration


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
    with pytest.raises(ConfigError) as exc_info:
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
    with download.zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("folder/file.txt", "data")
    extracted = download.extract_zip_safe(archive, safe_dir)
    assert (safe_dir / "folder" / "file.txt").read_text() == "data"
    assert extracted


def test_extract_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with download.zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "data")
    with pytest.raises(download.ConfigError):
        download.extract_zip_safe(archive, tmp_path / "out")


def test_extract_zip_rejects_absolute(tmp_path):
    archive = tmp_path / "bad_abs.zip"
    with download.zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/absolute/path.txt", "data")
    with pytest.raises(download.ConfigError):
        download.extract_zip_safe(archive, tmp_path / "out")


def test_extract_zip_detects_compression_bomb(tmp_path):
    archive = tmp_path / "bomb.zip"
    with download.zipfile.ZipFile(archive, "w", compression=download.zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("large.txt", b"0" * (11 * 1024 * 1024))

    with pytest.raises(download.ConfigError) as exc_info:
        download.extract_zip_safe(archive, tmp_path / "zip_out")

    assert "compression ratio" in str(exc_info.value)


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

    extracted = download.extract_tar_safe(archive, tmp_path / "tar_out")

    assert (tmp_path / "tar_out" / "folder" / "file.txt").read_bytes() == data
    assert extracted


def test_extract_tar_rejects_traversal(tmp_path):
    archive = tmp_path / "bad_traversal.tar.gz"
    info = tarfile.TarInfo("../evil.txt")
    payload = b"evil"
    info.size = len(payload)
    _make_tarfile(archive, [(info, payload)])

    with pytest.raises(download.ConfigError):
        download.extract_tar_safe(archive, tmp_path / "out")


def test_extract_tar_rejects_absolute(tmp_path):
    archive = tmp_path / "bad_absolute.tar.gz"
    info = tarfile.TarInfo("/abs/path.txt")
    data = b"absolute"
    info.size = len(data)
    _make_tarfile(archive, [(info, data)])

    with pytest.raises(download.ConfigError):
        download.extract_tar_safe(archive, tmp_path / "out")


def test_extract_tar_rejects_symlink(tmp_path):
    archive = tmp_path / "bad_symlink.tar.gz"
    info = tarfile.TarInfo("link")
    info.type = tarfile.SYMTYPE
    info.linkname = "target"
    info.size = 0
    _make_tarfile(archive, [(info, None)])

    with pytest.raises(download.ConfigError):
        download.extract_tar_safe(archive, tmp_path / "out")


def test_extract_tar_detects_compression_bomb(tmp_path):
    archive = tmp_path / "bomb.tar.gz"
    info = tarfile.TarInfo("large.bin")
    payload = b"0" * (11 * 1024 * 1024)
    info.size = len(payload)
    _make_tarfile(archive, [(info, payload)])

    with pytest.raises(download.ConfigError) as exc_info:
        download.extract_tar_safe(archive, tmp_path / "out")

    assert "compression ratio" in str(exc_info.value)


def test_download_stream_http_error(monkeypatch, tmp_path):
    response = DummyResponse(500, b"", {}, raise_error=True)
    make_session(monkeypatch, [response])
    with pytest.raises(ConfigError):
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
    with pytest.raises(ConfigError):
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
    make_session(monkeypatch, responses)
    destination = tmp_path / "file.owl"
    download.download_stream(
        url="https://example.org/file.owl",
        destination=destination,
        headers={},
        previous_manifest={"sha256": "mismatch"},
        http_config=DownloadConfiguration(),
        cache_dir=tmp_path / "cache",
        logger=_noop_logger(),
    )


def test_validate_url_security_rejects_private_ip():
    with pytest.raises(ConfigError):
        download.validate_url_security("https://127.0.0.1/ontology.owl")


def test_validate_url_security_upgrades_http(monkeypatch):
    monkeypatch.setattr(
        download.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )
    secure_url = download.validate_url_security("http://example.org/ontology.owl")
    assert secure_url.startswith("https://example.org")


def test_validate_url_security_respects_allowlist(monkeypatch):
    looked_up = {}

    def fake_getaddrinfo(host, *_args, **_kwargs):
        looked_up["host"] = host
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(download.socket, "getaddrinfo", fake_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["example.org", "purl.obolibrary.org"])

    secure_url = download.validate_url_security("https://purl.obolibrary.org/ontology.owl", config)

    assert looked_up["host"] == "purl.obolibrary.org"
    assert secure_url.startswith("https://purl.obolibrary.org")


def test_validate_url_security_blocks_disallowed_host():
    config = DownloadConfiguration(allowed_hosts=["example.org"])

    with pytest.raises(ConfigError):
        download.validate_url_security("https://malicious.com/evil.owl", config)


def test_validate_url_security_normalizes_idn(monkeypatch):
    looked_up = {}

    def fake_getaddrinfo(host, *_args, **_kwargs):
        looked_up["host"] = host
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(download.socket, "getaddrinfo", fake_getaddrinfo)

    config = DownloadConfiguration()
    secure_url = download.validate_url_security("https://münchen.example.org/ontology.owl", config)

    assert looked_up["host"] == "xn--mnchen-3ya.example.org"
    assert secure_url.startswith("https://xn--mnchen-3ya.example.org")


def test_validate_url_security_rejects_mixed_script_idn():
    with pytest.raises(ConfigError):
        download.validate_url_security("https://раураl.com/ontology.owl")


def test_validate_url_security_respects_wildcard_allowlist(monkeypatch):
    def fake_getaddrinfo(host, *_args, **_kwargs):
        return [(None, None, None, None, ("93.184.216.34", 0))]

    monkeypatch.setattr(download.socket, "getaddrinfo", fake_getaddrinfo)
    config = DownloadConfiguration(allowed_hosts=["*.example.org"])

    secure_url = download.validate_url_security("https://sub.example.org/ontology.owl", config)

    assert secure_url.startswith("https://sub.example.org")


def test_sanitize_filename_removes_traversal():
    sanitized = download.sanitize_filename("../evil.owl")
    assert ".." not in sanitized
    assert sanitized.endswith("evil.owl")


def test_download_stream_rejects_large_content(monkeypatch, tmp_path):
    response = DummyResponse(200, b"data", {"Content-Length": str(10 * 1024 * 1024 * 1024)})
    make_session(monkeypatch, [response])
    with pytest.raises(ConfigError):
        download.download_stream(
            url="https://example.org/file.owl",
            destination=tmp_path / "file.owl",
            headers={},
            previous_manifest=None,
            http_config=DownloadConfiguration(max_download_size_gb=1),
            cache_dir=tmp_path / "cache",
            logger=_noop_logger(),
        )


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
