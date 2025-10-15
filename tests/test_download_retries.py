"""
Download Retry Behaviour Tests

This module simulates HTTP responses to verify how the content download
pipeline handles transient failures, rate limit guidance, and fatal
errors when retrieving OpenAlex artifacts.

Key Scenarios:
- Ensures exponential retry loop continues through recoverable statuses
- Confirms Retry-After headers gate subsequent attempts
- Verifies non-retryable HTTP errors short-circuit additional requests

Dependencies:
- pytest: Fixture orchestration and assertions
- DocsToKG.ContentDownload.download_pyalex_pdfs: Download routines under test

Usage:
    pytest tests/test_download_retries.py
"""

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
from DocsToKG.ContentDownload.resolvers.types import DownloadOutcome


class _SequencedHandler(BaseHTTPRequestHandler):
    statuses: list[int] = []
    retry_after: int | None = None
    calls: list[int] = []
    head_calls: int = 0
    request_times: list[float] = []
    content: bytes = b"%PDF-1.4\n" + (b"0" * 2048) + b"\n%%EOF"

    def do_HEAD(self) -> None:  # noqa: D401 - HTTP handler signature
        self.__class__.head_calls += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: D401 - HTTP handler signature
        if not self.__class__.statuses:
            self.send_response(500)
            self.end_headers()
            return
        status = self.__class__.statuses.pop(0)
        self.__class__.request_times.append(time.monotonic())
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
        handler.calls = []
        handler.statuses = []
        handler.retry_after = None
        handler.head_calls = 0
        handler.request_times = []
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
    return (
        artifact,
        session,
        context,
        download_candidate(
            session,
            artifact,
            url,
            referer=None,
            timeout=5.0,
            context=context,
        ),
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


def test_download_candidate_avoids_per_request_head(http_server, tmp_path):
    """Ensure download path relies solely on GET without redundant HEAD calls."""

    handler, server = http_server
    handler.statuses = [200]
    handler.content = b"%PDF-1.4\n" + (b"1" * 4096) + b"\n%%EOF"
    url = f"http://127.0.0.1:{server.server_address[1]}/asset.pdf"

    _, session, _, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification == "pdf"
        assert handler.head_calls == 0
        assert handler.calls == [200]
    finally:
        session.close()


def test_retry_determinism_matches_request_with_retries(monkeypatch, http_server, tmp_path):
    """Verify retry budget and timing are governed exclusively by the helper."""

    handler, server = http_server
    handler.statuses = [429, 429, 200]
    url = f"http://127.0.0.1:{server.server_address[1]}/rate-limited.pdf"

    monkeypatch.setattr("DocsToKG.ContentDownload.http.random.random", lambda: 0.0)

    sleep_durations: list[float] = []

    def _capture_sleep(delay: float) -> None:
        sleep_durations.append(delay)

    monkeypatch.setattr("DocsToKG.ContentDownload.http.time.sleep", _capture_sleep)

    _, session, _, outcome = _download(url, tmp_path)
    try:
        assert outcome.classification == "pdf"
        assert handler.calls == [429, 429, 200]
        assert handler.head_calls == 0
        # Ensure exactly max_retries + 1 attempts were issued (default helper budget)
        assert len(handler.request_times) == 3
        assert sleep_durations == [0.75, 1.5]
    finally:
        session.close()
