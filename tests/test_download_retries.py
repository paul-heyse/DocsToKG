import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

from typing import Any, Dict, Tuple

import requests

from DocsToKG.ContentDownload.download_pyalex_pdfs import (
    WorkArtifact,
    _make_session,
    download_candidate,
)
from DocsToKG.ContentDownload.resolvers import DownloadOutcome


class _SequencedHandler(BaseHTTPRequestHandler):
    statuses: list[int] = []
    retry_after: int | None = None
    calls: list[int] = []
    content: bytes = b"%PDF-1.4\n%%EOF"

    def do_HEAD(self) -> None:  # noqa: D401 - HTTP handler signature
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: D401 - HTTP handler signature
        if not self.__class__.statuses:
            self.send_response(500)
            self.end_headers()
            return
        status = self.__class__.statuses.pop(0)
        self.__class__.calls.append(status)
        self.send_response(status)
        if status == 429 and self.__class__.retry_after is not None:
            self.send_header("Retry-After", str(self.__class__.retry_after))
        self.send_header("Content-Type", "application/pdf")
        self.end_headers()
        if status == 200:
            self.wfile.write(self.__class__.content)

    def log_message(self, format: str, *args: object) -> None:  # noqa: D401
        return


@pytest.fixture
def http_server():
    handler = _SequencedHandler
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield handler, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _make_artifact(base_dir: Path) -> WorkArtifact:
    pdf_dir = base_dir / "pdfs"
    html_dir = base_dir / "html"
    pdf_dir.mkdir()
    html_dir.mkdir()
    return WorkArtifact(
        work_id="W1",
        title="Test",
        publication_year=2024,
        doi="10.1234/test",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="test",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
    )


def _download(
    url: str, tmp_path: Path
) -> Tuple[WorkArtifact, requests.Session, Dict[str, Any], DownloadOutcome]:
    artifact = _make_artifact(tmp_path)
    context = {"dry_run": False, "extract_html_text": False, "previous": {}}
    session = _make_session({})
    return artifact, session, context, download_candidate(
        session,
        artifact,
        url,
        referer=None,
        timeout=5.0,
        context=context,
    )


@pytest.mark.parametrize("statuses", [[503, 503, 200]])
def test_download_candidate_retries_on_transient_errors(http_server, tmp_path, statuses):
    handler, server = http_server
    handler.statuses = list(statuses)
    handler.calls = []
    handler.retry_after = None
    url = f"http://127.0.0.1:{server.server_address[1]}/test.pdf"

    artifact, session, context, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification == "pdf"
        assert outcome.path is not None
        assert handler.calls == [503, 503, 200]
        assert Path(outcome.path).exists()
    finally:
        session.close()


def test_retry_after_header_respected(monkeypatch, http_server, tmp_path):
    handler, server = http_server
    handler.statuses = [429, 200]
    handler.retry_after = 2
    handler.calls = []
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    url = f"http://127.0.0.1:{server.server_address[1]}/test.pdf"

    artifact, session, context, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification == "pdf"
        assert handler.calls == [429, 200]
        assert sleep_calls and sleep_calls[0] >= handler.retry_after
    finally:
        session.close()


def test_non_retryable_errors_do_not_retry(http_server, tmp_path):
    handler, server = http_server
    handler.statuses = [404]
    handler.retry_after = None
    handler.calls = []
    url = f"http://127.0.0.1:{server.server_address[1]}/test.pdf"

    artifact = _make_artifact(tmp_path)
    context = {"dry_run": False, "extract_html_text": False, "previous": {}}
    session = _make_session({})
    try:
        outcome = download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context=context,
        )
    finally:
        session.close()
    assert outcome.classification == "http_error"
    assert handler.calls == [404]
