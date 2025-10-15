import io
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

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
    def __init__(self, queue):
        self.queue = queue
        self.calls = []

    def get(self, url, *, headers=None, stream=None, timeout=None, allow_redirects=None):
        response = self.queue.pop(0)
        if isinstance(response, Exception):
            raise response
        response.request_headers = headers or {}
        self.calls.append(response.request_headers)
        return response


def make_session(monkeypatch, responses):
    session = DummySession(list(responses))

    def _factory():
        return session

    monkeypatch.setattr(requests, "Session", _factory)
    return session


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
    session = make_session(monkeypatch, [response])
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
        http_config=DownloadConfiguration(max_retries=2, backoff_factor=0),
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

    monkeypatch.setattr(download, "_get_bucket", lambda host, config: StubBucket())
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

    return _Logger()
