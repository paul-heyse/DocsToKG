# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_network_unit",
#   "purpose": "Pytest coverage for content download network unit scenarios",
#   "sections": [
#     {
#       "id": "_dummy_response",
#       "name": "_DummyResponse",
#       "anchor": "DUMM",
#       "kind": "class"
#     },
#     {
#       "id": "_session_for_response",
#       "name": "_session_for_response",
#       "anchor": "SFR",
#       "kind": "function"
#     },
#     {
#       "id": "test_head_precheck_accepts_pdf_content",
#       "name": "test_head_precheck_accepts_pdf_content",
#       "anchor": "THPAP",
#       "kind": "function"
#     },
#     {
#       "id": "test_head_precheck_rejects_html_payload",
#       "name": "test_head_precheck_rejects_html_payload",
#       "anchor": "THPRH",
#       "kind": "function"
#     },
#     {
#       "id": "test_head_precheck_degrades_to_get",
#       "name": "test_head_precheck_degrades_to_get",
#       "anchor": "THPDT",
#       "kind": "function"
#     },
#     {
#       "id": "test_conditional_request_helper_requires_complete_metadata",
#       "name": "test_conditional_request_helper_requires_complete_metadata",
#       "anchor": "TCRHR",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from __future__ import annotations

from typing import Dict
from unittest.mock import Mock

from DocsToKG.ContentDownload.network import (
    ConditionalRequestHelper,
    head_precheck,
)


class _DummyResponse:
    def __init__(self, status_code: int, headers: Dict[str, str]):
        self.status_code = status_code
        self.headers = headers
        self.closed = False

    def close(self) -> None:  # noqa: D401
        self.closed = True


def _session_for_response(response: _DummyResponse, *, method: str = "HEAD") -> Mock:
    session = Mock()
    session_request = Mock(return_value=response)
    setattr(session, "request", session_request)

    def _request_with_retries(_session, _method, url, **kwargs):
        assert _method == method
        return response

    return session, _request_with_retries


def test_head_precheck_accepts_pdf_content(monkeypatch):
    response = _DummyResponse(200, {"Content-Type": "application/pdf"})
    session, helper = _session_for_response(response)
    monkeypatch.setattr("DocsToKG.ContentDownload.network.request_with_retries", helper)

    assert head_precheck(session, "https://example.org/file.pdf", timeout=10.0)
    assert response.closed


def test_head_precheck_rejects_html_payload(monkeypatch):
    response = _DummyResponse(200, {"Content-Type": "text/html"})
    session, helper = _session_for_response(response)
    monkeypatch.setattr("DocsToKG.ContentDownload.network.request_with_retries", helper)

    assert not head_precheck(session, "https://example.org/page", timeout=2.0)


def test_head_precheck_degrades_to_get(monkeypatch):
    head_response = _DummyResponse(405, {})
    get_response = _DummyResponse(200, {"Content-Type": "application/pdf"})

    def _request_with_retries(_session, method, url, **kwargs):
        if method == "HEAD":
            return head_response
        assert method == "GET"

        class _Stream:
            status_code = get_response.status_code
            headers = get_response.headers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def iter_content(self, chunk_size=1024):  # pragma: no cover - first chunk only
                yield b"%PDF"

            def close(self):
                return None

        return _Stream()

    monkeypatch.setattr(
        "DocsToKG.ContentDownload.network.request_with_retries", _request_with_retries
    )

    session = Mock()
    assert head_precheck(session, "https://example.org/pdf", timeout=3.0)


def test_conditional_request_helper_requires_complete_metadata(caplog):
    helper = ConditionalRequestHelper(
        prior_etag='"etag"',
        prior_last_modified="Wed, 01 May 2024 00:00:00 GMT",
        prior_sha256=None,
        prior_content_length=None,
        prior_path=None,
    )

    with caplog.at_level("WARNING"):
        headers = helper.build_headers()

    assert headers == {}
    assert any("resume-metadata-incomplete" in rec.message for rec in caplog.records)
