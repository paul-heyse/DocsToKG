"""Integration tests exercising streaming downloader concurrency and retries."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from DocsToKG.OntologyDownload import download
from DocsToKG.OntologyDownload.cli import _results_to_dict
from DocsToKG.OntologyDownload.cli_utils import format_results_table
from DocsToKG.OntologyDownload.config import DownloadConfiguration
from DocsToKG.OntologyDownload.core import FetchResult, FetchSpec


class _TestHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with shared state for deterministic responses."""

    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass):  # type: ignore[override]
        self.state: Dict[str, object] = {
            "flaky_count": 0,
            "head_requests": 0,
            "etag_flip_count": 0,
            "conditional_etag": '"stable"',
            "range_header": None,
            "active_requests": 0,
            "max_concurrent": 0,
            "content_type_history": [],
        }
        super().__init__(server_address, RequestHandlerClass)


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "DocsToKGTestServer/1.0"

    def log_message(self, format, *args):  # noqa: D401 - silence default logging
        return

    def _increment_concurrency(self):
        state = self.server.state
        state["active_requests"] += 1
        state["max_concurrent"] = max(state["max_concurrent"], state["active_requests"])

    def _decrement_concurrency(self):
        self.server.state["active_requests"] -= 1

    def do_HEAD(self):  # noqa: D401 - standard handler signature
        self._increment_concurrency()
        try:
            self._handle_request("HEAD")
        finally:
            self._decrement_concurrency()

    def do_GET(self):  # noqa: D401 - standard handler signature
        self._increment_concurrency()
        try:
            self._handle_request("GET")
        finally:
            self._decrement_concurrency()

    def _handle_request(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        path = parsed.path
        state = self.server.state

        if path == "/delay":
            delay_ms = int(params.get("ms", ["0"])[0])
            time.sleep(delay_ms / 1000.0)
            self._send_response(HTTPStatus.OK, b"delayed")
            return

        if path == "/flaky":
            if method != "HEAD":
                state["flaky_count"] += 1
                if state["flaky_count"] == 1:
                    self._send_response(HTTPStatus.SERVICE_UNAVAILABLE, b"temporary")
                else:
                    self._send_response(HTTPStatus.OK, b"recovered", etag='"flaky"')
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Length", "0")
                self.end_headers()
            return

        if path == "/head-mismatch":
            state["head_requests"] += 1
            if method == "HEAD":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send_response(
                    HTTPStatus.OK,
                    b"rdf-data",
                    content_type="application/rdf+xml",
                    etag='"mismatch"',
                )
            return

        if path == "/etag-flip":
            if method == "HEAD":
                self.send_response(HTTPStatus.OK)
                self.send_header("ETag", '"v1"')
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                state["etag_flip_count"] += 1
                version = "v1" if state["etag_flip_count"] == 1 else "v2"
                body = f"etag-{version}".encode("utf-8")
                self._send_response(HTTPStatus.OK, body, etag=f'"{version}"')
            return

        if path == "/conditional":
            etag = state["conditional_etag"]
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                self.end_headers()
            else:
                payload = b"fresh-data"
                self._send_response(HTTPStatus.OK, payload, etag=etag)
            return

        if path == "/error":
            try:
                status_code = int(
                    params.get("status", [str(int(HTTPStatus.INTERNAL_SERVER_ERROR))])[0]
                )
                status = HTTPStatus(status_code)
            except (ValueError, KeyError):
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            body = params.get("body", ["error"])[0].encode("utf-8")
            self._send_response(status, body)
            return

        if path == "/partial":
            content = b"abcdefghijklmnopqrstuvwxyz"
            range_header = self.headers.get("Range")
            if method == "HEAD":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
            else:
                state["range_header"] = range_header
                if range_header and range_header.startswith("bytes="):
                    start = int(range_header.split("=", 1)[1].split("-", 1)[0])
                    chunk = content[start:]
                    end = len(content) - 1
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self._send_response(
                        HTTPStatus.OK,
                        content,
                        content_type="application/octet-stream",
                    )
            return

        if path == "/content-type":
            requested = params.get("value", ["application/octet-stream"])[0]
            state["content_type_history"].append((method, requested))
            self._send_response(
                HTTPStatus.OK,
                b"body",
                content_type=requested,
            )
            return

        self._send_response(HTTPStatus.NOT_FOUND, b"unknown")

    def _send_response(
        self,
        status: HTTPStatus,
        body: bytes,
        *,
        content_type: str = "text/plain",
        etag: Optional[str] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


@pytest.fixture()
def http_server():
    server = _TestHTTPServer(("127.0.0.1", 0), _RequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    try:
        yield server, base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _make_http_config(**overrides) -> DownloadConfiguration:
    config = DownloadConfiguration()
    return config.model_copy(update=overrides)


@pytest.fixture(autouse=True)
def _allow_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "validate_url_security", lambda url, http_config=None: url)


def _download(
    url: str,
    *,
    destination: Path,
    http_config: DownloadConfiguration,
    previous_manifest: Optional[Dict[str, object]] = None,
    expected_media_type: Optional[str] = None,
    service: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
):
    cache = cache_dir or destination.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    log = logger or logging.getLogger("ontology-download-test")
    return download.download_stream(
        url=url,
        destination=destination,
        headers={},
        previous_manifest=previous_manifest,
        http_config=http_config,
        cache_dir=cache,
        logger=log,
        expected_media_type=expected_media_type,
        service=service,
    )


def test_retry_after_transient_error(
    http_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server, base_url = http_server
    config = _make_http_config()
    destination = tmp_path / "flaky.bin"
    caplog.set_level(logging.WARNING, logger="ontology-download-test")
    result = _download(f"{base_url}/flaky", destination=destination, http_config=config)
    assert result.status == "fresh"
    assert destination.read_bytes() == b"recovered"
    # First attempt fails with 503, second succeeds
    assert server.state["flaky_count"] == 2
    assert any("download retry" in record.message for record in caplog.records)


def test_head_mismatch_logs_warning(
    http_server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, base_url = http_server
    config = _make_http_config()
    destination = tmp_path / "mismatch.bin"
    caplog.set_level(logging.WARNING)
    _download(
        f"{base_url}/head-mismatch",
        destination=destination,
        http_config=config,
        expected_media_type="application/rdf+xml",
    )
    warnings = [
        record for record in caplog.records if "media type mismatch detected" in record.message
    ]
    assert warnings, "expected media type mismatch warning"


def test_conditional_requests_return_cached(http_server, tmp_path: Path) -> None:
    server, base_url = http_server
    config = _make_http_config()
    destination = tmp_path / "conditional.bin"
    first = _download(f"{base_url}/conditional", destination=destination, http_config=config)
    assert first.status == "fresh"
    previous_manifest = {"etag": first.etag, "last_modified": first.last_modified}
    second = _download(
        f"{base_url}/conditional",
        destination=destination,
        http_config=config,
        previous_manifest=previous_manifest,
    )
    assert second.status == "cached"
    assert destination.read_bytes() == b"fresh-data"
    assert server.state["conditional_etag"] == '"stable"'


def test_partial_content_resume(http_server, tmp_path: Path) -> None:
    server, base_url = http_server
    config = _make_http_config()
    destination = tmp_path / "partial.bin"
    partial_path = Path(str(destination) + ".part")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_bytes(b"abcde")
    result = _download(f"{base_url}/partial", destination=destination, http_config=config)
    assert result.status == "updated"
    assert destination.read_bytes() == b"abcdefghijklmnopqrstuvwxyz"
    assert server.state["range_header"] == "bytes=5-"


def test_etag_change_triggers_full_download(http_server, tmp_path: Path) -> None:
    server, base_url = http_server
    config = _make_http_config()
    destination = tmp_path / "etag.bin"
    first = _download(f"{base_url}/etag-flip", destination=destination, http_config=config)
    assert first.etag == '"v1"'
    previous_manifest = {"etag": first.etag}
    second = _download(
        f"{base_url}/etag-flip",
        destination=destination,
        http_config=config,
        previous_manifest=previous_manifest,
    )
    assert second.etag == '"v2"'
    assert destination.read_bytes() == b"etag-v2"
    assert server.state["etag_flip_count"] == 2


def test_token_bucket_limits_same_service(http_server, tmp_path: Path) -> None:
    server, base_url = http_server
    server.state["max_concurrent"] = 0
    server.state["active_requests"] = 0
    download._TOKEN_BUCKETS.clear()
    config = _make_http_config(per_host_rate_limit="1/second")
    urls = [f"{base_url}/delay?ms=100" for _ in range(3)]
    destinations = [tmp_path / f"same-{idx}.bin" for idx in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                _download,
                url,
                destination=dest,
                http_config=config,
            )
            for url, dest in zip(urls, destinations)
        ]
        for future in futures:
            future.result(timeout=5)
    assert server.state["max_concurrent"] == 1


def test_token_bucket_allows_parallel_services(http_server, tmp_path: Path) -> None:
    server, base_url = http_server
    server.state["max_concurrent"] = 0
    server.state["active_requests"] = 0
    download._TOKEN_BUCKETS.clear()
    config = _make_http_config(per_host_rate_limit="1/second")
    urls = [f"{base_url}/delay?ms=100" for _ in range(3)]
    destinations = [tmp_path / f"diff-{idx}.bin" for idx in range(3)]
    services = [f"svc-{idx}" for idx in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                _download,
                url,
                destination=dest,
                http_config=config,
                service=svc,
            )
            for url, dest, svc in zip(urls, destinations, services)
        ]
        for future in futures:
            future.result(timeout=5)
    assert server.state["max_concurrent"] >= 2


def test_cli_results_to_dict_contains_expected_fields(tmp_path: Path) -> None:
    spec = FetchSpec(id="HP", resolver="obo", extras={}, target_formats=("owl",))
    result = FetchResult(
        spec=spec,
        local_path=tmp_path / "hp.owl",
        status="success",
        sha256="deadbeef",
        manifest_path=tmp_path / "manifest.json",
        artifacts=(str(tmp_path / "hp.ttl"),),
    )
    payload = _results_to_dict(result)
    assert payload["id"] == "HP"
    assert payload["resolver"] == "obo"
    assert payload["status"] == "success"
    assert payload["manifest"].endswith("manifest.json")
    assert payload["artifacts"] == [str(tmp_path / "hp.ttl")]


def test_cli_results_table_includes_status_and_sha(tmp_path: Path) -> None:
    spec = FetchSpec(id="CHEBI", resolver="ols", extras={}, target_formats=("obo",))
    result = FetchResult(
        spec=spec,
        local_path=tmp_path / "chebi.owl",
        status="cached",
        sha256="cafebabe",
        manifest_path=tmp_path / "manifest.json",
        artifacts=(),
    )
    table = format_results_table([result])
    assert "CHEBI" in table
    assert "cached" in table
    assert "cafebabe" in table
