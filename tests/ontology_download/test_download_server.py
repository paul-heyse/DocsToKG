# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_download_server",
#   "purpose": "Pytest coverage for ontology download download server scenarios",
#   "sections": [
#     {
#       "id": "serverstate",
#       "name": "_ServerState",
#       "anchor": "class-serverstate",
#       "kind": "class"
#     },
#     {
#       "id": "statefulserver",
#       "name": "_StatefulServer",
#       "anchor": "class-statefulserver",
#       "kind": "class"
#     },
#     {
#       "id": "handler",
#       "name": "_Handler",
#       "anchor": "class-handler",
#       "kind": "class"
#     },
#     {
#       "id": "http-server",
#       "name": "http_server",
#       "anchor": "function-http-server",
#       "kind": "function"
#     },
#     {
#       "id": "reset-token-buckets",
#       "name": "_reset_token_buckets",
#       "anchor": "function-reset-token-buckets",
#       "kind": "function"
#     },
#     {
#       "id": "allow-local-addresses",
#       "name": "_allow_local_addresses",
#       "anchor": "function-allow-local-addresses",
#       "kind": "function"
#     },
#     {
#       "id": "download-to-tmp",
#       "name": "_download_to_tmp",
#       "anchor": "function-download-to-tmp",
#       "kind": "function"
#     },
#     {
#       "id": "stublogger",
#       "name": "_StubLogger",
#       "anchor": "class-stublogger",
#       "kind": "class"
#     },
#     {
#       "id": "test-flaky-server-retry",
#       "name": "test_flaky_server_retry",
#       "anchor": "function-test-flaky-server-retry",
#       "kind": "function"
#     },
#     {
#       "id": "test-error-endpoint-raises",
#       "name": "test_error_endpoint_raises",
#       "anchor": "function-test-error-endpoint-raises",
#       "kind": "function"
#     },
#     {
#       "id": "test-head-mismatch-logs-warning",
#       "name": "test_head_mismatch_logs_warning",
#       "anchor": "function-test-head-mismatch-logs-warning",
#       "kind": "function"
#     },
#     {
#       "id": "test-cache-hit-uses-304",
#       "name": "test_cache_hit_uses_304",
#       "anchor": "function-test-cache-hit-uses-304",
#       "kind": "function"
#     },
#     {
#       "id": "test-partial-resume-handles-range",
#       "name": "test_partial_resume_handles_range",
#       "anchor": "function-test-partial-resume-handles-range",
#       "kind": "function"
#     },
#     {
#       "id": "test-etag-flip-updates-etag",
#       "name": "test_etag_flip_updates_etag",
#       "anchor": "function-test-etag-flip-updates-etag",
#       "kind": "function"
#     },
#     {
#       "id": "test-token-bucket-limits-concurrency",
#       "name": "test_token_bucket_limits_concurrency",
#       "anchor": "function-test-token-bucket-limits-concurrency",
#       "kind": "function"
#     },
#     {
#       "id": "test-concurrent-hosts-do-not-block",
#       "name": "test_concurrent_hosts_do_not_block",
#       "anchor": "function-test-concurrent-hosts-do-not-block",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Integration tests exercising the streaming downloader against a local server."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import pytest
import requests

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import DownloadConfiguration
from DocsToKG.OntologyDownload import ontology_download as download


@dataclass
class _ServerState:
    flaky_calls: int = 0
    etag_version: int = 0
    cache_etag: str = '"etag-cache"'
    payload: bytes = b"ontology-stream"
    large_payload: bytes = b"0123456789" * 1024
    media_type: str = "application/rdf+xml"
    head_media_type: str = "text/plain"
    lock: threading.Lock = field(default_factory=threading.Lock)
    concurrent_active: int = 0
    concurrent_max: int = 0


class _StatefulServer(ThreadingHTTPServer):
    def __init__(self, address, handler, state: _ServerState):
        super().__init__(address, handler)
        self.state = state


class _Handler(BaseHTTPRequestHandler):
    server: _StatefulServer  # type: ignore[assignment]

    def log_message(self, format: str, *args):  # noqa: D401 - silence server logs
        """Suppress default HTTP server logging."""

    def _write(self, status: int, headers: Dict[str, str], body: bytes = b"") -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: D401
        state = self.server.state
        parsed = urlparse(self.path)
        if parsed.path == "/head-mismatch":
            headers = {
                "Content-Type": state.head_media_type,
                "Content-Length": str(len(state.payload)),
            }
            self._write(200, headers)
            return
        if parsed.path == "/delay":
            params = parse_qs(parsed.query)
            delay_ms = int(params.get("ms", ["100"])[0])
            time.sleep(delay_ms / 1000)
        self._write(
            200, {"Content-Type": state.media_type, "Content-Length": str(len(state.payload))}
        )

    def do_GET(self) -> None:  # noqa: D401
        state = self.server.state
        parsed = urlparse(self.path)
        if parsed.path == "/flaky":
            with state.lock:
                state.flaky_calls += 1
                call = state.flaky_calls
            if call == 1:
                self._write(503, {"Content-Type": "text/plain"}, b"retry later")
            else:
                self._write(200, {"Content-Type": state.media_type}, state.payload)
            return
        if parsed.path == "/error":
            params = parse_qs(parsed.query)
            status = int(params.get("status", ["500"])[0])
            self._write(status, {"Content-Type": "text/plain"}, b"error")
            return
        if parsed.path == "/head-mismatch":
            headers = {"Content-Type": state.media_type, "Content-Length": str(len(state.payload))}
            self._write(200, headers, state.payload)
            return
        if parsed.path == "/cache":
            etag = state.cache_etag
            if self.headers.get("If-None-Match") == etag:
                self._write(304, {"ETag": etag})
            else:
                headers = {"ETag": etag, "Content-Type": state.media_type}
                self._write(200, headers, state.payload)
            return
        if parsed.path == "/etag-flip":
            with state.lock:
                state.etag_version += 1
                etag = f'"etag-{state.etag_version}"'
            headers = {"ETag": etag, "Content-Type": state.media_type}
            self._write(200, headers, state.payload)
            return
        if parsed.path == "/partial":
            etag = '"partial-etag"'
            data = state.large_payload
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                start = int(range_header.split("=", 1)[1].split("-", 1)[0])
                body = data[start:]
                headers = {
                    "Content-Type": state.media_type,
                    "ETag": etag,
                    "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{len(data)-1}/{len(data)}",
                }
                self._write(206, headers, body)
            else:
                headers = {
                    "Content-Type": state.media_type,
                    "Content-Length": str(len(data)),
                    "ETag": etag,
                }
                self._write(200, headers, data)
            return
        if parsed.path == "/concurrent":
            with state.lock:
                state.concurrent_active += 1
                state.concurrent_max = max(state.concurrent_max, state.concurrent_active)
            time.sleep(0.05)
            with state.lock:
                state.concurrent_active -= 1
            self._write(200, {"Content-Type": state.media_type}, state.payload)
            return
        if parsed.path == "/content-type":
            params = parse_qs(parsed.query)
            override = params.get("type", [state.media_type])[0]
            headers = {"Content-Type": override}
            self._write(200, headers, state.payload)
            return
        headers = {"Content-Type": state.media_type}
        self._write(200, headers, state.payload)


@pytest.fixture(scope="module")
def http_server():
    state = _ServerState()
    server = _StatefulServer(("127.0.0.1", 0), _Handler, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    yield base_url, state
    server.shutdown()
    thread.join()


@pytest.fixture(autouse=True)
# --- Helper Functions ---

def _reset_token_buckets():
    download._TOKEN_BUCKETS.clear()
    yield
    download._TOKEN_BUCKETS.clear()


@pytest.fixture(autouse=True)
def _allow_local_addresses(monkeypatch):
    from types import SimpleNamespace
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    def _validate(url: str, http_config=None):  # noqa: ANN001 - test helper
        return url

    monkeypatch.setattr(download, "validate_url_security", _validate)

    class _LocalResponse:
        def __init__(self, url: str, method: str, headers: Dict[str, str], timeout: Optional[int]):
            req = urllib_request.Request(url, method=method, headers=headers)
            try:
                handle = urllib_request.urlopen(req, timeout=timeout)
            except urllib_error.HTTPError as exc:
                if exc.code < 400:
                    handle = exc
                else:
                    error = requests.HTTPError(f"status {exc.code}")
                    error.response = SimpleNamespace(status_code=exc.code)
                    raise error from exc
            except urllib_error.URLError as exc:
                raise requests.ConnectionError(str(exc)) from exc
            self._handle = handle
            self.status_code = self._handle.getcode()
            self.headers = dict(self._handle.getheaders())
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._handle.close()
            return False

        def iter_content(self, chunk_size=1 << 20):
            while True:
                chunk = self._handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        def raise_for_status(self):
            if self.status_code >= 400:
                error = requests.HTTPError(f"status {self.status_code}")
                error.response = SimpleNamespace(status_code=self.status_code)
                raise error

    class _LocalSession:
        def __init__(self):
            self.headers: Dict[str, str] = {}

        def head(self, url, *, headers=None, timeout=None, allow_redirects=None):  # noqa: D401
            merged = {**self.headers, **(headers or {})}
            response = _LocalResponse(url, "HEAD", merged, timeout)
            response.__exit__(None, None, None)  # close immediately
            return response

        def get(self, url, *, headers=None, stream=None, timeout=None, allow_redirects=None):
            merged = {**self.headers, **(headers or {})}
            return _LocalResponse(url, "GET", merged, timeout)

    monkeypatch.setattr(download.requests, "Session", _LocalSession)


def _download_to_tmp(
    url: str, tmp_path: Path, *, previous_manifest=None, validate_media: bool = False
):
    destination = tmp_path / f"download-{hash(url)}.owl"
    http_config = DownloadConfiguration(
        max_retries=2,
        validate_media_type=validate_media,
        allowed_hosts=["127.0.0.1", "localhost"],
    )
    result = download.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=previous_manifest,
        http_config=http_config,
        cache_dir=tmp_path / "cache",
        logger=_StubLogger(),
    )
    return destination, result


class _StubLogger:
    def __init__(self):
        self.warnings: List[Dict[str, str]] = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, message, *, extra=None):
        if extra:
            self.warnings.append(extra)

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass
# --- Test Cases ---


def test_flaky_server_retry(http_server, tmp_path):
    base_url, state = http_server
    state.flaky_calls = 0
    destination, result = _download_to_tmp(f"{base_url}/flaky", tmp_path)
    assert result.status == "fresh"
    assert destination.read_bytes() == state.payload
    assert state.flaky_calls == 2


def test_error_endpoint_raises(http_server, tmp_path):
    base_url, _ = http_server
    http_config = DownloadConfiguration(allowed_hosts=["127.0.0.1", "localhost"])
    with pytest.raises(download.DownloadFailure):
        download.download_stream(
            url=f"{base_url}/error?status=502",
            destination=tmp_path / "error.owl",
            headers={},
            previous_manifest=None,
            http_config=http_config,
            cache_dir=tmp_path / "cache",
            logger=_StubLogger(),
        )


def test_head_mismatch_logs_warning(http_server, tmp_path):
    base_url, state = http_server
    state.head_media_type = "text/plain"
    logger = _StubLogger()
    destination = tmp_path / "head.owl"
    http_config = DownloadConfiguration(
        validate_media_type=True, allowed_hosts=["127.0.0.1", "localhost"]
    )
    download.download_stream(
        url=f"{base_url}/head-mismatch",
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=http_config,
        cache_dir=tmp_path / "cache",
        logger=logger,
        expected_media_type="application/rdf+xml",
    )
    assert any(entry.get("expected_media_type") for entry in logger.warnings)


def test_cache_hit_uses_304(http_server, tmp_path):
    base_url, state = http_server
    destination, result = _download_to_tmp(f"{base_url}/cache", tmp_path)
    assert result.status == "fresh"
    manifest = {
        "etag": state.cache_etag,
        "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "sha256": download.sha256_file(destination),
    }
    destination.write_bytes(b"cached")
    _, cached = _download_to_tmp(f"{base_url}/cache", tmp_path, previous_manifest=manifest)
    assert cached.status == "cached"


def test_partial_resume_handles_range(http_server, tmp_path):
    base_url, state = http_server
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    part_path = cache_dir / "partial.part"
    part_path.write_bytes(state.large_payload[:100])
    destination = tmp_path / "partial.owl"
    result = download.download_stream(
        url=f"{base_url}/partial",
        destination=destination,
        headers={},
        previous_manifest=None,
        http_config=DownloadConfiguration(allowed_hosts=["127.0.0.1", "localhost"]),
        cache_dir=cache_dir,
        logger=_StubLogger(),
    )
    assert result.status in {"fresh", "updated"}
    assert destination.read_bytes().startswith(state.large_payload)


def test_etag_flip_updates_etag(http_server, tmp_path):
    base_url, _ = http_server
    first_dest, first_result = _download_to_tmp(f"{base_url}/etag-flip", tmp_path)
    manifest = {
        "etag": first_result.etag,
        "sha256": download.sha256_file(first_dest),
    }
    second_dest, second_result = _download_to_tmp(
        f"{base_url}/etag-flip", tmp_path, previous_manifest=manifest
    )
    assert second_result.etag != first_result.etag
    assert second_dest.read_bytes() == first_dest.read_bytes()


def test_token_bucket_limits_concurrency(monkeypatch, http_server, tmp_path):
    base_url, state = http_server
    state.concurrent_max = 0

    class _StubBucket:
        def __init__(self, limit: int = 1):
            self.limit = limit
            self.active = 0
            self.max_active = 0
            self.cond = threading.Condition()

        def consume(self, tokens: float = 1.0) -> None:  # noqa: ARG002 - signature parity
            with self.cond:
                while self.active >= self.limit:
                    self.cond.wait()
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.cond:
                self.active -= 1
                self.cond.notify_all()

    bucket = _StubBucket()

    def _get_bucket(host, http_config, service=None):  # noqa: ARG001
        return bucket

    monkeypatch.setattr(download, "_get_bucket", _get_bucket)

    def _run(idx: int) -> Path:
        path = tmp_path / f"concurrent-{idx}.owl"
        download.download_stream(
            url=f"{base_url}/concurrent",
            destination=path,
            headers={},
            previous_manifest=None,
            http_config=DownloadConfiguration(allowed_hosts=["127.0.0.1", "localhost"]),
            cache_dir=tmp_path / "cache",
            logger=_StubLogger(),
        )
        return path

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert bucket.max_active == 1


def test_concurrent_hosts_do_not_block(monkeypatch, http_server, tmp_path):
    base_url, _ = http_server
    bucket_calls: Dict[str, int] = {}

    class _PassthroughBucket:
        def __init__(self, host: str):
            self.host = host

        def consume(self, tokens: float = 1.0) -> None:  # noqa: ARG002
            bucket_calls[self.host] = bucket_calls.get(self.host, 0) + 1

    def _bucket_for_host(host, http_config, service=None):  # noqa: ARG001
        return _PassthroughBucket(host)

    monkeypatch.setattr(download, "_get_bucket", _bucket_for_host)

    def _run(url: str, output: Path) -> None:
        download.download_stream(
            url=url,
            destination=output,
            headers={},
            previous_manifest=None,
            http_config=DownloadConfiguration(allowed_hosts=["127.0.0.1", "localhost"]),
            cache_dir=tmp_path / "cache",
            logger=_StubLogger(),
        )

    threads = [
        threading.Thread(target=_run, args=(f"{base_url}/concurrent", tmp_path / "same.owl")),
        threading.Thread(
            target=_run,
            args=(
                f"{base_url.replace('127.0.0.1', 'localhost')}/concurrent",
                tmp_path / "other.owl",
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(bucket_calls) == 2
    assert all(count == 1 for count in bucket_calls.values())
