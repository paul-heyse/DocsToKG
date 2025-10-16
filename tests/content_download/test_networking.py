"""Consolidated content download networking tests."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import Mock, call, patch

import pytest
import requests

from DocsToKG.ContentDownload import download_pyalex_pdfs as downloader
from DocsToKG.ContentDownload.download_pyalex_pdfs import (
    JsonlLogger,
    WorkArtifact,
    _make_session,
    download_candidate,
    process_one_work,
)
from DocsToKG.ContentDownload.network import (
    CachedResult,
    ConditionalRequestHelper,
    ModifiedResult,
    head_precheck,
    parse_retry_after_header,
    request_with_retries,
)
from DocsToKG.ContentDownload.resolvers import (
    DownloadOutcome,
    OpenAlexResolver,
    ResolverConfig,
    ResolverMetrics,
    ResolverPipeline,
    ResolverResult,
    WaybackResolver,
)
from DocsToKG.ContentDownload.utils import dedupe, normalize_doi, normalize_pmcid, strip_prefix

# ---- test_conditional_requests.py -----------------------------
try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

# ---- test_conditional_requests.py -----------------------------
HAS_REQUESTS = find_spec("requests") is not None

# ---- test_conditional_requests.py -----------------------------
HAS_PYALEX = find_spec("pyalex") is not None

# ---- test_conditional_requests.py -----------------------------
given = hypothesis.given

# ---- test_conditional_requests.py -----------------------------
if HAS_REQUESTS and HAS_PYALEX:
    from DocsToKG.ContentDownload.download_pyalex_pdfs import (
        ManifestEntry,
        WorkArtifact,
        build_manifest_entry,
        download_candidate,
    )
    from DocsToKG.ContentDownload.resolvers import DownloadOutcome

    class _DummyResponse:
        def __init__(self, status_code: int, headers: Dict[str, str]) -> None:
            self.status_code = status_code
            self.headers = headers

        def __enter__(self) -> "_DummyResponse":  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
            return None

        def iter_content(self, chunk_size: int):  # pragma: no cover - not needed for 304 path
            return iter(())

    class _DummyHead:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}

        def close(self) -> None:  # pragma: no cover - nothing to release
            return

    class _DummySession:
        def __init__(self, response: _DummyResponse) -> None:
            self._response = response

        def head(self, url: str, **kwargs: Any) -> _DummyHead:  # noqa: D401
            return _DummyHead()

        def get(self, url: str, **kwargs: Any) -> _DummyResponse:  # noqa: D401
            return self._response

    def _make_artifact(tmp_path: Path) -> WorkArtifact:
        pdf_dir = tmp_path / "pdfs"
        html_dir = tmp_path / "html"
        pdf_dir.mkdir()
        html_dir.mkdir()
        return WorkArtifact(
            work_id="W-cond",
            title="Conditional",
            publication_year=2024,
            doi="10.1234/cond",
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="conditional",
            pdf_dir=pdf_dir,
            html_dir=html_dir,
        )

    def test_download_candidate_returns_cached(tmp_path: Path) -> None:
        artifact = _make_artifact(tmp_path)
        previous_path = str(artifact.pdf_dir / "conditional.pdf")
        previous = {
            "etag": '"etag"',
            "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "path": previous_path,
            "sha256": "abc",
            "content_length": 123,
        }
        response = _DummyResponse(
            304,
            {"Content-Type": "application/pdf"},
        )
        session = _DummySession(response)
        outcome = download_candidate(
            session,
            artifact,
            "https://example.org/test.pdf",
            referer=None,
            timeout=5.0,
            context={"previous": {"https://example.org/test.pdf": previous}},
        )

        assert outcome.classification == "cached"
        assert outcome.path == previous_path
        assert outcome.sha256 == "abc"
        assert outcome.last_modified == previous["last_modified"]
        assert not (artifact.pdf_dir / "conditional.pdf").exists()

    def test_manifest_entry_preserves_conditional_headers() -> None:
        outcome = DownloadOutcome(
            classification="pdf",
            path="/tmp/file.pdf",
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=12.3,
            sha256="deadbeef",
            content_length=42,
            etag='"tag"',
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        )
        artifact = WorkArtifact(
            work_id="W-cond",
            title="Conditional",
            publication_year=2024,
            doi="10.1234/cond",
            pmid=None,
            pmcid=None,
            arxiv_id=None,
            landing_urls=[],
            pdf_urls=[],
            open_access_url=None,
            source_display_names=[],
            base_stem="conditional",
            pdf_dir=Path("/tmp"),
            html_dir=Path("/tmp"),
        )
        entry = build_manifest_entry(
            artifact, "resolver", "https://example.org", outcome, [], dry_run=False
        )
        assert isinstance(entry, ManifestEntry)
        assert entry.etag == '"tag"'
        assert entry.last_modified == "Mon, 01 Jan 2024 00:00:00 GMT"


# ---- test_conditional_requests.py -----------------------------
@dataclass
class _HelperResponse:
    status_code: int
    headers: Dict[str, str]


# ---- test_conditional_requests.py -----------------------------
def _make_helper_response(
    status_code: int, headers: Optional[Dict[str, str]] = None
) -> _HelperResponse:
    return _HelperResponse(status_code=status_code, headers=headers or {})


# ---- test_conditional_requests.py -----------------------------
def test_build_headers_empty_metadata() -> None:
    helper = ConditionalRequestHelper()

    assert helper.build_headers() == {}


# ---- test_conditional_requests.py -----------------------------
def test_build_headers_etag_only() -> None:
    helper = ConditionalRequestHelper(prior_etag="abc123")

    assert helper.build_headers() == {"If-None-Match": "abc123"}


# ---- test_conditional_requests.py -----------------------------
def test_build_headers_last_modified_only() -> None:
    helper = ConditionalRequestHelper(prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT")

    assert helper.build_headers() == {"If-Modified-Since": "Wed, 21 Oct 2015 07:28:00 GMT"}


# ---- test_conditional_requests.py -----------------------------
def test_build_headers_with_both_headers() -> None:
    helper = ConditionalRequestHelper(
        prior_etag="abc123",
        prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
    )

    assert helper.build_headers() == {
        "If-None-Match": "abc123",
        "If-Modified-Since": "Wed, 21 Oct 2015 07:28:00 GMT",
    }


# ---- test_conditional_requests.py -----------------------------
def test_interpret_response_cached_returns_cached_result() -> None:
    helper = ConditionalRequestHelper(
        prior_etag="abc123",
        prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
        prior_sha256="deadbeef",
        prior_content_length=1024,
        prior_path="/tmp/file.pdf",
    )
    response = _make_helper_response(304)

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, CachedResult)
    assert result.path == "/tmp/file.pdf"
    assert result.sha256 == "deadbeef"
    assert result.content_length == 1024
    assert result.etag == "abc123"
    assert result.last_modified == "Wed, 21 Oct 2015 07:28:00 GMT"


# ---- test_conditional_requests.py -----------------------------
def test_interpret_response_cached_missing_metadata_raises() -> None:
    helper = ConditionalRequestHelper(prior_etag="abc123")
    response = _make_helper_response(304)

    with pytest.raises(ValueError):
        helper.interpret_response(response)  # type: ignore[arg-type]


# ---- test_conditional_requests.py -----------------------------
def test_interpret_response_modified_returns_modified_result() -> None:
    helper = ConditionalRequestHelper()
    response = _make_helper_response(200)

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, ModifiedResult)
    assert result.etag is None
    assert result.last_modified is None


# ---- test_conditional_requests.py -----------------------------
def test_interpret_response_modified_extracts_headers() -> None:
    helper = ConditionalRequestHelper()
    response = _make_helper_response(
        200,
        {
            "ETag": '"xyz"',
            "Last-Modified": "Thu, 01 Jan 1970 00:00:00 GMT",
        },
    )

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, ModifiedResult)
    assert result.etag == '"xyz"'
    assert result.last_modified == "Thu, 01 Jan 1970 00:00:00 GMT"


# ---- test_conditional_requests.py -----------------------------
def test_interpret_response_missing_metadata_lists_fields() -> None:
    helper = ConditionalRequestHelper(prior_etag="tag-only")
    response = _make_helper_response(304)

    with pytest.raises(ValueError) as excinfo:
        helper.interpret_response(response)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "path" in message
    assert "sha256" in message
    assert "content_length" in message


# ---- test_conditional_requests.py -----------------------------
@given(
    etag=st.one_of(st.none(), st.text(min_size=1)),
    last_modified=st.one_of(st.none(), st.text(min_size=1)),
)
def test_build_headers_property(etag: Optional[str], last_modified: Optional[str]) -> None:
    helper = ConditionalRequestHelper(prior_etag=etag, prior_last_modified=last_modified)
    headers = helper.build_headers()

    if etag:
        assert headers["If-None-Match"] == etag
    else:
        assert "If-None-Match" not in headers

    if last_modified:
        assert headers["If-Modified-Since"] == last_modified
    else:
        assert "If-Modified-Since" not in headers


# ---- test_conditional_requests.py -----------------------------
@given(
    path=st.text(min_size=1),
    sha=st.text(min_size=1),
    size=st.integers(min_value=1, max_value=10_000),
)
def test_interpret_response_cached_property(path: str, sha: str, size: int) -> None:
    helper = ConditionalRequestHelper(
        prior_path=path,
        prior_sha256=sha,
        prior_content_length=size,
        prior_etag="etag",
        prior_last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )
    response = _make_helper_response(304)

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, CachedResult)
    assert result.path == path
    assert result.sha256 == sha
    assert result.content_length == size


# ---- test_conditional_requests.py -----------------------------
def test_conditional_helper_rejects_negative_length() -> None:
    with pytest.raises(ValueError):
        ConditionalRequestHelper(prior_content_length=-1)


# ---- test_conditional_requests.py -----------------------------
def test_interpret_response_requires_response_shape() -> None:
    helper = ConditionalRequestHelper()

    with pytest.raises(TypeError):
        helper.interpret_response(object())  # type: ignore[arg-type]


# ---- test_download_retries.py -----------------------------
pytest.importorskip("requests")

# ---- test_download_retries.py -----------------------------
pytest.importorskip("pyalex")


# ---- test_download_retries.py -----------------------------
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


# ---- test_download_retries.py -----------------------------
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


# ---- test_download_retries.py -----------------------------
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


# ---- test_download_retries.py -----------------------------
def _download(
    url: str, tmp_path: Path
) -> Tuple[WorkArtifact, requests.Session, Dict[str, Any], DownloadOutcome]:
    artifact = _make_artifact(tmp_path)
    context = {
        "dry_run": False,
        "extract_html_text": False,
        "previous": {},
        "skip_head_precheck": True,
    }
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


# ---- test_download_retries.py -----------------------------
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


# ---- test_download_retries.py -----------------------------
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


# ---- test_download_retries.py -----------------------------
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


# ---- test_download_retries.py -----------------------------
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


# ---- test_head_precheck.py -----------------------------
def test_head_precheck_allows_pdf(monkeypatch):
    head_response = Mock(status_code=200, headers={"Content-Type": "application/pdf"})
    head_response.close = Mock()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries",
        lambda *args, **kwargs: head_response,
    )

    assert head_precheck(Mock(), "https://example.org/file.pdf", timeout=10.0)
    head_response.close.assert_called_once()


def test_head_precheck_rejects_html(monkeypatch):
    head_response = Mock(status_code=200, headers={"Content-Type": "text/html"})
    head_response.close = Mock()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries",
        lambda *args, **kwargs: head_response,
    )

    assert not head_precheck(Mock(), "https://example.org/page", timeout=10.0)
    head_response.close.assert_called_once()


@pytest.mark.parametrize("status", [405, 501])
def test_head_precheck_degrades_to_get_pdf(monkeypatch, status):
    class _StreamResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "application/pdf"}
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def iter_content(self, chunk_size: int = 1024):
            yield b"%PDF"

        def close(self) -> None:
            self.closed = True

    head_response = Mock(status_code=status, headers={})
    head_response.close = Mock()
    stream_response = _StreamResponse()

    responses = [head_response, stream_response]

    def fake_request(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    assert head_precheck(Mock(), "https://example.org/pdf", timeout=10.0)
    assert stream_response.closed is True


@pytest.mark.parametrize("status", [405, 501])
def test_head_precheck_degrades_to_get_html(monkeypatch, status):
    class _StreamResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def iter_content(self, chunk_size: int = 1024):
            yield b"<html></html>"

        def close(self) -> None:
            return None

    head_response = Mock(status_code=status, headers={})
    head_response.close = Mock()
    stream_response = _StreamResponse()

    responses = [head_response, stream_response]

    def fake_request(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries",
        fake_request,
    )

    assert not head_precheck(Mock(), "https://example.org/html", timeout=10.0)


@pytest.mark.parametrize(
    "exc",
    [requests.Timeout("boom"), requests.ConnectionError("boom")],
)
def test_head_precheck_returns_true_on_exception(monkeypatch, exc):
    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries",
        Mock(side_effect=exc),
    )

    assert head_precheck(Mock(), "https://example.org/err", timeout=5.0)


# ---- test_download_retries.py -----------------------------
def test_retry_determinism_matches_request_with_retries(monkeypatch, http_server, tmp_path):
    """Verify retry budget and timing are governed exclusively by the helper."""

    handler, server = http_server
    handler.statuses = [429, 429, 200]
    url = f"http://127.0.0.1:{server.server_address[1]}/rate-limited.pdf"

    monkeypatch.setattr("DocsToKG.ContentDownload.network.random.random", lambda: 0.0)

    sleep_durations: list[float] = []

    def _capture_sleep(delay: float) -> None:
        sleep_durations.append(delay)

    monkeypatch.setattr("DocsToKG.ContentDownload.network.time.sleep", _capture_sleep)

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


# ---- test_http_retry.py -----------------------------
try:  # pragma: no cover - dependency optional in CI
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - skip if requests missing
    requests = pytest.importorskip("requests")  # type: ignore

# ---- test_http_retry.py -----------------------------
try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

# ---- test_http_retry.py -----------------------------
given = hypothesis.given


# ---- test_http_retry.py -----------------------------
def _mock_response(status: int, headers: Optional[Dict[str, str]] = None) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    return response


# ---- test_http_retry.py -----------------------------
def test_successful_request_no_retries():
    """Verify successful request completes immediately without retries."""

    session = Mock(spec=requests.Session)
    response = _mock_response(200)
    session.request.return_value = response

    result = request_with_retries(session, "GET", "https://example.org/test")

    assert result is response
    session.request.assert_called_once_with(method="GET", url="https://example.org/test")


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_transient_503_with_exponential_backoff(mock_sleep: Mock, _: Mock) -> None:
    """Verify exponential backoff timing for transient 503 errors."""

    session = Mock(spec=requests.Session)
    response_503 = _mock_response(503, headers={})
    response_200 = _mock_response(200)
    session.request.side_effect = [response_503, response_503, response_200]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        max_retries=3,
        backoff_factor=0.5,
    )

    assert result is response_200
    assert session.request.call_count == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0)]


# ---- test_http_retry.py -----------------------------
def test_parse_retry_after_header_integer() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "5"}

    assert parse_retry_after_header(response) == 5.0


# ---- test_http_retry.py -----------------------------
def test_parse_retry_after_header_http_date() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    header_value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response = requests.Response()
    response.headers = {"Retry-After": header_value}

    wait = parse_retry_after_header(response)
    assert wait is not None
    assert 0.0 <= wait <= 30.0


# ---- test_http_retry.py -----------------------------
def test_parse_retry_after_header_invalid_date() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Thu, 32 Foo 2024 00:00:00 GMT"}

    assert parse_retry_after_header(response) is None


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.random.random", return_value=0.0)
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_retry_after_header_overrides_backoff(mock_sleep: Mock, _: Mock) -> None:
    session = Mock(spec=requests.Session)
    retry_headers = {"Retry-After": "10"}
    response_retry = _mock_response(429, headers=retry_headers)
    response_success = _mock_response(200)
    session.request.side_effect = [response_retry, response_success]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        backoff_factor=0.1,
        max_retries=2,
    )

    assert result is response_success
    assert mock_sleep.call_args_list == [call(10.0)]


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_request_exception_raises_after_retries(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    error = requests.RequestException("boom")
    session.request.side_effect = error

    with pytest.raises(requests.RequestException):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=1)

    assert mock_sleep.call_count == 1
    assert session.request.call_count == 2


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_timeout_retry_handling(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    session.request.side_effect = [requests.Timeout("slow"), _mock_response(200)]

    result = request_with_retries(session, "GET", "https://example.org/timeout", max_retries=1)

    assert result.status_code == 200
    assert mock_sleep.call_count == 1


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_connection_error_retry_handling(mock_sleep: Mock) -> None:
    session = Mock(spec=requests.Session)
    session.request.side_effect = [requests.ConnectionError("down"), _mock_response(200)]

    result = request_with_retries(session, "GET", "https://example.org/conn", max_retries=1)

    assert result.status_code == 200
    assert mock_sleep.call_count == 1


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_timeout_raises_after_exhaustion(mock_sleep: Mock) -> None:
    """Ensure timeout retries raise after exhausting the retry budget."""

    session = Mock(spec=requests.Session)
    session.request.side_effect = requests.Timeout("slow")

    with pytest.raises(requests.Timeout):
        request_with_retries(session, "GET", "https://example.org/timeout", max_retries=1)

    # Only the non-terminal attempt sleeps before re-raising on the final attempt.
    assert mock_sleep.call_count == 1


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_connection_error_raises_after_exhaustion(mock_sleep: Mock) -> None:
    """Ensure connection errors propagate when retries are exhausted."""

    session = Mock(spec=requests.Session)
    session.request.side_effect = requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        request_with_retries(session, "GET", "https://example.org/conn", max_retries=1)

    assert mock_sleep.call_count == 1


# ---- test_http_retry.py -----------------------------
@given(st.text())
def test_parse_retry_after_header_property(value: str) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": value}

    result = parse_retry_after_header(response)

    if result is not None:
        assert result >= 0.0 or math.isnan(result)


# ---- test_http_retry.py -----------------------------
def test_request_with_custom_retry_statuses() -> None:
    session = Mock(spec=requests.Session)
    failing = _mock_response(404)
    success = _mock_response(200)
    session.request.side_effect = [failing, success]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        retry_statuses={404},
        max_retries=1,
    )

    assert result is success
    assert session.request.call_count == 2


# ---- test_http_retry.py -----------------------------
def test_request_returns_after_exhausting_single_attempt() -> None:
    session = Mock(spec=requests.Session)
    retry_response = _mock_response(503)
    session.request.return_value = retry_response

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/test",
        max_retries=0,
    )

    assert result is retry_response


# ---- test_http_retry.py -----------------------------
def test_request_with_retries_rejects_negative_retries() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "https://example.org/test", max_retries=-1)


# ---- test_http_retry.py -----------------------------
def test_request_with_retries_rejects_negative_backoff() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "https://example.org/test", backoff_factor=-0.1)


# ---- test_http_retry.py -----------------------------
def test_request_with_retries_requires_method_and_url() -> None:
    session = Mock(spec=requests.Session)

    with pytest.raises(ValueError):
        request_with_retries(session, "", "https://example.org/test")

    with pytest.raises(ValueError):
        request_with_retries(session, "GET", "")


# ---- test_http_retry.py -----------------------------
def test_request_with_retries_uses_method_fallback() -> None:
    class _MinimalSession:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def get(self, url: str, **kwargs: Any):  # noqa: D401
            self.calls.append(url)
            response = Mock(spec=requests.Response)
            response.status_code = 200
            response.headers = {}
            return response

    session = _MinimalSession()

    response = request_with_retries(session, "GET", "https://example.org/fallback")

    assert response.status_code == 200
    assert session.calls == ["https://example.org/fallback"]


# ---- test_http_retry.py -----------------------------
def test_request_with_retries_errors_when_no_callable_available() -> None:
    class _MinimalSession:
        pass

    with pytest.raises(AttributeError):
        request_with_retries(_MinimalSession(), "GET", "https://example.org/fail")


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
def test_retry_after_header_prefers_longer_delay(mock_sleep: Mock) -> None:
    """Verify Retry-After header longer than backoff takes precedence."""

    session = Mock(spec=requests.Session)

    retry_response = requests.Response()
    retry_response.status_code = 429
    retry_response.headers = {"Retry-After": "4"}

    success_response = requests.Response()
    success_response.status_code = 200
    success_response.headers = {}

    session.request.side_effect = [retry_response, success_response]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/with-retry-after",
        backoff_factor=0.1,
        max_retries=2,
    )

    assert result.status_code == 200
    mock_sleep.assert_called_once()
    sleep_arg = mock_sleep.call_args[0][0]
    assert pytest.approx(sleep_arg, rel=0.01) == 4.0


# ---- test_http_retry.py -----------------------------
@patch("DocsToKG.ContentDownload.network.time.sleep")
@patch("DocsToKG.ContentDownload.network.parse_retry_after_header")
def test_respect_retry_after_false_skips_header(mock_parse: Mock, mock_sleep: Mock) -> None:
    """Ensure disabling respect_retry_after bypasses header parsing."""

    session = Mock(spec=requests.Session)
    retry_response = _mock_response(503)
    success_response = _mock_response(200)
    session.request.side_effect = [retry_response, success_response]

    result = request_with_retries(
        session,
        "GET",
        "https://example.org/no-retry-after",
        respect_retry_after=False,
        max_retries=1,
        backoff_factor=0.1,
    )

    assert result is success_response
    mock_parse.assert_not_called()
    mock_sleep.assert_called_once()


# ---- test_http_retry.py -----------------------------
def test_parse_retry_after_header_naive_datetime() -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00"}

    wait = parse_retry_after_header(response)
    assert wait is not None
    assert wait >= 0.0


# ---- test_http_retry.py -----------------------------
def test_parse_retry_after_header_handles_parse_errors(monkeypatch) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00 GMT"}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.parsedate_to_datetime",
        Mock(side_effect=TypeError("boom")),
    )

    assert parse_retry_after_header(response) is None


# ---- test_http_retry.py -----------------------------
def test_parse_retry_after_header_returns_none_when_parser_returns_none(monkeypatch) -> None:
    response = requests.Response()
    response.headers = {"Retry-After": "Mon, 01 Jan 2024 00:00:00 GMT"}

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.parsedate_to_datetime",
        Mock(return_value=None),
    )

    assert parse_retry_after_header(response) is None


# ---- test_download_outcomes.py -----------------------------
if "pyalex" not in sys.modules:
    pyalex_stub = types.ModuleType("pyalex")
    pyalex_stub.Topics = object
    pyalex_stub.Works = object
    config_stub = types.ModuleType("pyalex.config")
    config_stub.mailto = None
    pyalex_stub.config = config_stub
    sys.modules["pyalex"] = pyalex_stub
    sys.modules["pyalex.config"] = config_stub


# ---- test_download_outcomes.py -----------------------------
class FakeResponse:
    def __init__(self, status_code: int, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def iter_content(self, chunk_size: int):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        pass


# ---- test_download_outcomes.py -----------------------------
def make_artifact(tmp_path: Path) -> downloader.WorkArtifact:
    artifact = downloader.WorkArtifact(
        work_id="W-DOWNLOAD",
        title="Outcome Example",
        publication_year=2024,
        doi="10.1000/example",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="outcome-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    return artifact


# ---- test_download_outcomes.py -----------------------------
def stub_requests(
    monkeypatch, mapping: Dict[Tuple[str, str], Callable[[], FakeResponse] | FakeResponse]
):
    def fake_request(session, method, url, **kwargs):
        key = (method.upper(), url)
        if key not in mapping:
            raise AssertionError(f"Unexpected request {key}")
        response = mapping[key]
        return response() if callable(response) else response

    monkeypatch.setattr(downloader, "request_with_retries", fake_request)


# ---- test_download_outcomes.py -----------------------------
def test_successful_pdf_download_populates_metadata(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    pdf_bytes = b"%PDF-1.4\n" + (b"x" * 2048) + b"\n%%EOF"
    expected_sha = hashlib.sha256(pdf_bytes).hexdigest()

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"etag-123"',
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            },
            chunks=[pdf_bytes],
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"skip_head_precheck": True},
    )

    assert outcome.classification == "pdf"
    assert outcome.path is not None
    stored = Path(outcome.path)
    assert stored.exists()
    assert outcome.sha256 == expected_sha
    assert outcome.content_length == stored.stat().st_size
    assert outcome.etag == '"etag-123"'
    assert outcome.last_modified == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert outcome.error is None
    assert outcome.extracted_text_path is None
    rehashed = hashlib.sha256(stored.read_bytes()).hexdigest()
    assert rehashed == expected_sha


# ---- test_download_outcomes.py -----------------------------
def test_cached_response_preserves_prior_metadata(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    cached_path = str(artifact.pdf_dir / "cached.pdf")
    context = {
        "previous": {
            url: {
                "path": cached_path,
                "sha256": "cached-sha",
                "content_length": 1024,
                "etag": '"etag-cached"',
                "last_modified": "Tue, 02 Jan 2024 00:00:00 GMT",
            }
        }
    }

    mapping = {
        ("GET", url): lambda: FakeResponse(
            304,
            headers={"Content-Type": "application/pdf"},
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session, artifact, url, None, timeout=10.0, context=context
    )

    assert outcome.classification == "cached"
    assert outcome.path == cached_path
    assert outcome.sha256 == "cached-sha"
    assert outcome.content_length == 1024
    assert outcome.etag == '"etag-cached"'
    assert outcome.last_modified == "Tue, 02 Jan 2024 00:00:00 GMT"
    assert outcome.error is None


# ---- test_download_outcomes.py -----------------------------
def test_http_error_sets_metadata_to_none(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"

    mapping = {
        ("GET", url): lambda: FakeResponse(404, headers={"Content-Type": "text/html"}),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0)

    assert outcome.classification == "http_error"
    assert outcome.path is None
    assert outcome.sha256 is None
    assert outcome.content_length is None
    assert outcome.etag is None
    assert outcome.last_modified is None
    assert outcome.error is None


# ---- test_download_outcomes.py -----------------------------
def test_html_download_with_text_extraction(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/page.html"
    html_bytes = b"<!DOCTYPE html><html><body><p>Hello</p></body></html>"

    html_extractor = types.SimpleNamespace(extract=lambda text: "Hello")
    monkeypatch.setitem(sys.modules, "trafilatura", html_extractor)

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "text/html",
                "ETag": '"etag-html"',
                "Last-Modified": "Wed, 03 Jan 2024 00:00:00 GMT",
            },
            chunks=[html_bytes],
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"extract_html_text": True},
    )

    assert outcome.classification == "html"
    assert outcome.path is not None and outcome.path.endswith(".html")
    assert outcome.extracted_text_path is not None
    extracted = Path(outcome.extracted_text_path)
    assert extracted.exists()
    assert extracted.read_text(encoding="utf-8") == "Hello"
    assert outcome.sha256 is not None
    assert outcome.etag == '"etag-html"'
    assert outcome.last_modified == "Wed, 03 Jan 2024 00:00:00 GMT"


# ---- test_download_outcomes.py -----------------------------
def test_dry_run_preserves_metadata_without_files(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/paper.pdf"
    pdf_bytes = b"%PDF-1.4\n" + (b"y" * 2048) + b"\n%%EOF"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"etag-dry"',
                "Last-Modified": "Thu, 04 Jan 2024 00:00:00 GMT",
            },
            chunks=[pdf_bytes],
        ),
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"dry_run": True},
    )

    assert outcome.classification == "pdf"
    assert outcome.path is None
    assert outcome.sha256 is None
    assert outcome.content_length is None
    assert outcome.extracted_text_path is None
    assert outcome.etag == '"etag-dry"'
    assert outcome.last_modified == "Thu, 04 Jan 2024 00:00:00 GMT"


# ---- test_download_outcomes.py -----------------------------
def test_small_pdf_detected_as_corrupt(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/tiny.pdf"
    tiny_pdf = b"%PDF-1.4\n1 0 obj<<>>\nendobj\n%%EOF"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={"Content-Type": "application/pdf"},
            chunks=[tiny_pdf],
        )
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(
        session,
        artifact,
        url,
        None,
        timeout=10.0,
        context={"skip_head_precheck": True},
    )

    assert outcome.classification == "pdf_corrupt"
    assert outcome.path is None
    assert not any(artifact.pdf_dir.glob("*.pdf"))


# ---- test_download_outcomes.py -----------------------------
def test_html_tail_in_pdf_marks_corruption(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/error.pdf"
    payload = b"%PDF-1.4\nstream\n<html>Error page</html>"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={"Content-Type": "application/pdf"},
            chunks=[payload],
        )
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0)

    assert outcome.classification == "pdf_corrupt"
    assert outcome.path is None
    assert not any(artifact.pdf_dir.glob("*.pdf"))


# ---- test_download_outcomes.py -----------------------------
def test_build_manifest_entry_includes_download_metadata(tmp_path):
    artifact = make_artifact(tmp_path)
    download_path = str(artifact.pdf_dir / "saved.pdf")
    outcome = DownloadOutcome(
        classification="pdf",
        path=download_path,
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=150.0,
        error=None,
        sha256="abc123",
        content_length=4096,
        etag='"etag-manifest"',
        last_modified="Fri, 05 Jan 2024 00:00:00 GMT",
        extracted_text_path=str(artifact.html_dir / "saved.txt"),
    )

    entry = downloader.build_manifest_entry(
        artifact,
        "figshare",
        "https://example.org/paper.pdf",
        outcome,
        html_paths=["/tmp/example.html"],
        dry_run=False,
    )

    assert entry.sha256 == "abc123"
    assert entry.content_length == 4096
    assert entry.etag == '"etag-manifest"'
    assert entry.last_modified == "Fri, 05 Jan 2024 00:00:00 GMT"
    assert entry.extracted_text_path == str(artifact.html_dir / "saved.txt")


# ---- test_download_outcomes.py -----------------------------
def test_rfc5987_filename_suffix(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/no-extension"
    pdf_bytes = b"%PDF-1.4\n" + (b"z" * 2048) + b"\n%%EOF"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": "attachment; filename*=UTF-8''paper%E2%82%AC.PDF",
            },
            chunks=[pdf_bytes],
        )
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0)

    assert outcome.classification == "pdf"
    assert outcome.path is not None
    assert outcome.path.endswith(".pdf")


# ---- test_download_outcomes.py -----------------------------
def test_html_filename_suffix_from_disposition(tmp_path, monkeypatch):
    artifact = make_artifact(tmp_path)
    url = "https://example.org/content"
    html_bytes = b"<html><body>Hi</body></html>"

    mapping = {
        ("GET", url): lambda: FakeResponse(
            200,
            headers={
                "Content-Type": "application/xhtml+xml",
                "Content-Disposition": "inline; filename=landing.xhtml",
            },
            chunks=[html_bytes],
        )
    }
    stub_requests(monkeypatch, mapping)

    session = requests.Session()
    outcome = downloader.download_candidate(session, artifact, url, None, timeout=10.0)

    assert outcome.classification == "html"
    assert outcome.path is not None
    assert outcome.path.endswith(".xhtml")


# ---- test_download_utils.py -----------------------------
pytest.importorskip("pyalex")


# ---- test_download_utils.py -----------------------------
def test_slugify_truncates_and_normalises():
    assert downloader.slugify("Hello, World!", keep=8) == "Hello_Wo"
    assert downloader.slugify("   ", keep=10) == "untitled"
    assert downloader.slugify("Study: B-cells & growth", keep=40) == "Study_Bcells_growth"


# ---- test_download_utils.py -----------------------------
@pytest.mark.parametrize(
    "payload,ctype,url,expected",
    [
        (b"%PDF-sample", "application/pdf", "https://example.org/file.pdf", "pdf"),
        (b"   %PDF-1.4", "text/plain", "https://example.org/file.bin", "pdf"),
        (b"<html><head></head>", "text/html", "https://example.org", "html"),
        (b"", "application/pdf", "https://example.org/file.pdf", "pdf"),
        (b"", "text/plain", "https://example.org/foo.pdf", "pdf"),
    ],
)
def test_classify_payload_variants(payload, ctype, url, expected):
    assert downloader.classify_payload(payload, ctype, url) == expected


# ---- test_download_utils.py -----------------------------
def test_collect_location_urls_dedupes_and_tracks_sources():
    work = {
        "best_oa_location": {
            "landing_page_url": "https://host.example/landing",
            "pdf_url": "https://host.example/paper.pdf",
            "source": {"display_name": "Host"},
        },
        "primary_location": {
            "landing_page_url": "https://host.example/landing",
            "pdf_url": "https://cdn.example/paper.pdf",
            "source": {"display_name": "Mirror"},
        },
        "locations": [
            {
                "landing_page_url": "https://mirror.example/landing",
                "pdf_url": "https://cdn.example/paper.pdf",
                "source": {"display_name": "Mirror"},
            }
        ],
        "open_access": {"oa_url": "https://oa.example/paper.pdf"},
    }
    collected = downloader._collect_location_urls(work)
    assert collected["landing"] == [
        "https://host.example/landing",
        "https://mirror.example/landing",
    ]
    assert collected["pdf"] == [
        "https://host.example/paper.pdf",
        "https://cdn.example/paper.pdf",
        "https://oa.example/paper.pdf",
    ]
    assert collected["sources"] == ["Host", "Mirror"]


# ---- test_download_utils.py -----------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://doi.org/10.1000/foo", "10.1000/foo"),
        (" 10.1000/bar ", "10.1000/bar"),
        (None, None),
    ],
)
def test_normalize_doi(value, expected):
    assert normalize_doi(value) == expected


# ---- test_download_utils.py -----------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("PMID:123456", "123456"),
        ("https://pubmed.ncbi.nlm.nih.gov/98765/", "98765"),
        (None, None),
    ],
)
def test_normalize_pmid(value, expected):
    assert downloader._normalize_pmid(value) == expected


# ---- test_download_utils.py -----------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("PMC12345", "PMC12345"),
        ("pmc9876", "PMC9876"),
        ("9876", "PMC9876"),
        (None, None),
    ],
)
def test_normalize_pmcid(value, expected):
    assert normalize_pmcid(value) == expected


# ---- test_download_utils.py -----------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("arXiv:2101.12345", "2101.12345"),
        ("https://arxiv.org/abs/2010.00001", "2010.00001"),
        ("2101.99999", "2101.99999"),
    ],
)
def test_normalize_arxiv(value, expected):
    assert downloader._normalize_arxiv(value) == expected


# ---- test_content_download_utils.py -----------------------------
try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

# ---- test_content_download_utils.py -----------------------------
given = hypothesis.given


# ---- test_content_download_utils.py -----------------------------
def test_normalize_doi_with_https_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"


# ---- test_content_download_utils.py -----------------------------
def test_normalize_doi_without_prefix() -> None:
    assert normalize_doi("10.1234/abc") == "10.1234/abc"


# ---- test_content_download_utils.py -----------------------------
def test_normalize_doi_with_whitespace() -> None:
    assert normalize_doi("  10.1234/abc  ") == "10.1234/abc"


# ---- test_content_download_utils.py -----------------------------
def test_normalize_doi_none() -> None:
    assert normalize_doi(None) is None


# ---- test_content_download_utils.py -----------------------------
@pytest.mark.parametrize(
    "prefix",
    [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
        "DOI:",
    ],
)
def test_normalize_doi_prefix_variants(prefix: str) -> None:
    canonical = "10.1234/Example"
    assert normalize_doi(f"{prefix}{canonical}") == canonical


# ---- test_content_download_utils.py -----------------------------
def test_normalize_pmcid_with_pmc_prefix() -> None:
    assert normalize_pmcid("PMC123456") == "PMC123456"


# ---- test_content_download_utils.py -----------------------------
def test_normalize_pmcid_without_prefix_adds_prefix() -> None:
    assert normalize_pmcid("123456") == "PMC123456"


# ---- test_content_download_utils.py -----------------------------
def test_normalize_pmcid_lowercase() -> None:
    assert normalize_pmcid("pmc123456") == "PMC123456"


# ---- test_content_download_utils.py -----------------------------
def test_strip_prefix_case_insensitive() -> None:
    assert strip_prefix("ARXIV:2301.12345", "arxiv:") == "2301.12345"


# ---- test_content_download_utils.py -----------------------------
def test_dedupe_preserves_order() -> None:
    assert dedupe(["b", "a", "b", "c"]) == ["b", "a", "c"]


# ---- test_content_download_utils.py -----------------------------
def test_dedupe_filters_falsey_values() -> None:
    assert dedupe(["a", "", None, "a"]) == ["a"]


# ---- test_content_download_utils.py -----------------------------
@given(st.lists(st.text()))
def test_dedupe_property(values: List[str]) -> None:
    expected = []
    seen = set()
    for item in values:
        if item and item not in seen:
            expected.append(item)
            seen.add(item)

    assert dedupe(values) == expected


# ---- test_edge_cases.py -----------------------------
pytest.importorskip("pyalex")

# ---- test_edge_cases.py -----------------------------
requests = pytest.importorskip("requests")

# ---- test_edge_cases.py -----------------------------
responses = pytest.importorskip("responses")


# ---- test_edge_cases.py -----------------------------
def _make_artifact(tmp_path: Path, **overrides: Any) -> WorkArtifact:
    params: Dict[str, Any] = dict(
        work_id="WEDGE",
        title="Edge Case",
        publication_year=2024,
        doi="10.1/test",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/resource"],
        open_access_url=None,
        source_display_names=[],
        base_stem="edge-case",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
    )
    params.update(overrides)
    return WorkArtifact(**params)


# ---- test_edge_cases.py -----------------------------
@responses.activate
def test_html_classification_overrides_misleading_content_type(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    url = artifact.pdf_urls[0]
    responses.add(responses.HEAD, url, status=200, headers={"Content-Type": "application/pdf"})
    responses.add(responses.GET, url, status=200, body="<html><body>Fake PDF</body></html>")

    outcome = download_candidate(
        requests.Session(),
        artifact,
        url,
        referer=None,
        timeout=10.0,
        context={"dry_run": False, "extract_html_text": False, "previous": {}},
    )

    assert outcome.classification == "html"
    assert outcome.path and outcome.path.endswith(".html")


# ---- test_edge_cases.py -----------------------------
@responses.activate
def test_wayback_resolver_skips_unavailable_archives(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path, pdf_urls=[])
    artifact.failed_pdf_urls = ["https://example.org/missing.pdf"]
    session = requests.Session()
    config = ResolverConfig()
    responses.add(
        responses.GET,
        "https://archive.org/wayback/available",
        json={
            "archived_snapshots": {"closest": {"available": False, "url": "https://archive.org"}}
        },
        status=200,
    )

    results = list(WaybackResolver().iter_urls(session, config, artifact))
    assert results == []


# ---- test_edge_cases.py -----------------------------
def test_manifest_and_attempts_single_success(tmp_path: Path) -> None:
    work = {
        "id": "https://openalex.org/WEDGE",
        "title": "Edge Case",
        "publication_year": 2024,
        "ids": {"doi": "10.1/test"},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }

    artifact = _make_artifact(tmp_path)
    logger_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(logger_path)
    metrics = ResolverMetrics()

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterable[ResolverResult]:
            yield ResolverResult(url="https://resolver.example/paper.pdf")

    def fake_download(*args: Any, **kwargs: Any) -> DownloadOutcome:
        pdf_path = artifact.pdf_dir / "resolver.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
        return DownloadOutcome(
            classification="pdf",
            path=str(pdf_path),
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=5.0,
            sha256="deadbeef",
            content_length=12,
        )

    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        enable_head_precheck=False,
    )
    pipeline = ResolverPipeline([StubResolver()], config, fake_download, logger, metrics)

    result = process_one_work(
        work,
        requests.Session(),
        artifact.pdf_dir,
        artifact.html_dir,
        pipeline,
        logger,
        metrics,
        dry_run=False,
        extract_html_text=False,
        previous_lookup={},
        resume_completed=set(),
    )

    logger.close()

    assert result["saved"] is True
    records = [
        json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    attempts = [
        entry
        for entry in records
        if entry["record_type"] == "attempt" and entry["status"] in {"pdf", "pdf_unknown"}
    ]
    manifests = [
        entry
        for entry in records
        if entry["record_type"] == "manifest" and entry["classification"] in {"pdf", "pdf_unknown"}
    ]
    assert len(attempts) == 1
    assert len(manifests) == 1
    assert attempts[0]["work_id"] == manifests[0]["work_id"] == "WEDGE"
    assert attempts[0]["sha256"] == "deadbeef"
    assert manifests[0]["path"].endswith("resolver.pdf")
    assert Path(manifests[0]["path"]).exists()


# ---- test_edge_cases.py -----------------------------
def test_openalex_attempts_use_session_headers(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    logger_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(logger_path)
    metrics = ResolverMetrics()
    session = requests.Session()
    session.headers.update({"User-Agent": "EdgeTester/1.0"})
    observed: List[str] = []

    def fake_download(session_obj, art, url, referer, timeout, context=None):
        observed.append(session_obj.headers.get("User-Agent"))
        return DownloadOutcome(
            classification="request_error",
            path=None,
            http_status=None,
            content_type=None,
            elapsed_ms=1.0,
            error="failed",
        )

    config = ResolverConfig(
        resolver_order=["openalex"],
        resolver_toggles={"openalex": True},
        enable_head_precheck=False,
    )
    pipeline = ResolverPipeline(
        [OpenAlexResolver()],
        config,
        fake_download,
        logger,
        metrics,
    )

    pipeline.run(session, artifact)

    assert observed == ["EdgeTester/1.0"]
    logger.close()


# ---- test_edge_cases.py -----------------------------
def test_retry_budget_honours_max_attempts(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    config = ResolverConfig(
        resolver_order=["stub"],
        resolver_toggles={"stub": True},
        max_attempts_per_work=3,
        enable_head_precheck=False,
    )

    class StubResolver:
        name = "stub"

        def is_enabled(self, config: ResolverConfig, artifact: WorkArtifact) -> bool:
            return True

        def iter_urls(
            self,
            session: requests.Session,
            config: ResolverConfig,
            artifact: WorkArtifact,
        ) -> Iterator[ResolverResult]:
            for i in range(10):
                yield ResolverResult(url=f"https://resolver.example/{i}.pdf")

    calls: List[str] = []

    def failing_download(*args: Any, **kwargs: Any) -> DownloadOutcome:
        url = args[2]
        calls.append(url)
        return DownloadOutcome(
            classification="http_error",
            path=None,
            http_status=503,
            content_type="text/plain",
            elapsed_ms=1.0,
            error="server-error",
        )

    class ListLogger:
        def __init__(self) -> None:
            self.records: List[Any] = []

        def log(self, record) -> None:  # pragma: no cover - simple passthrough
            self.records.append(record)

    pipeline = ResolverPipeline(
        [StubResolver()],
        config,
        failing_download,
        ListLogger(),
        ResolverMetrics(),
    )

    result = pipeline.run(requests.Session(), artifact)
    assert result.success is False
    assert len(calls) == config.max_attempts_per_work
