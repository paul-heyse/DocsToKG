"""
Conditional Request Handling Tests

This module verifies the downloader's HTTP caching semantics by simulating
conditional requests, ensuring cached assets are reused when remote
servers respond with 304 Not Modified.

Key Scenarios:
- Ensures cached payloads short-circuit redundant downloads
- Confirms manifest entries preserve ETag and Last-Modified metadata

Dependencies:
- pytest: Assertions and fixtures
- DocsToKG.ContentDownload.download_pyalex_pdfs: Conditional request logic

Usage:
    pytest tests/test_conditional_requests.py
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

try:
    import hypothesis
    from hypothesis import strategies as st  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis is required for these tests", allow_module_level=True)

from DocsToKG.ContentDownload.network import (
    CachedResult,
    ConditionalRequestHelper,
    ModifiedResult,
)

HAS_REQUESTS = find_spec("requests") is not None
HAS_PYALEX = find_spec("pyalex") is not None

given = hypothesis.given

if HAS_REQUESTS and HAS_PYALEX:
    from DocsToKG.ContentDownload.download_pyalex_pdfs import (
        ManifestEntry,
        WorkArtifact,
        build_manifest_entry,
        download_candidate,
    )
    from DocsToKG.ContentDownload.resolvers.types import DownloadOutcome

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


@dataclass
class _HelperResponse:
    status_code: int
    headers: Dict[str, str]


def _make_helper_response(
    status_code: int, headers: Optional[Dict[str, str]] = None
) -> _HelperResponse:
    return _HelperResponse(status_code=status_code, headers=headers or {})


def test_build_headers_empty_metadata() -> None:
    helper = ConditionalRequestHelper()

    assert helper.build_headers() == {}


def test_build_headers_etag_only() -> None:
    helper = ConditionalRequestHelper(prior_etag="abc123")

    assert helper.build_headers() == {"If-None-Match": "abc123"}


def test_build_headers_last_modified_only() -> None:
    helper = ConditionalRequestHelper(prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT")

    assert helper.build_headers() == {"If-Modified-Since": "Wed, 21 Oct 2015 07:28:00 GMT"}


def test_build_headers_with_both_headers() -> None:
    helper = ConditionalRequestHelper(
        prior_etag="abc123",
        prior_last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
    )

    assert helper.build_headers() == {
        "If-None-Match": "abc123",
        "If-Modified-Since": "Wed, 21 Oct 2015 07:28:00 GMT",
    }


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


def test_interpret_response_cached_missing_metadata_raises() -> None:
    helper = ConditionalRequestHelper(prior_etag="abc123")
    response = _make_helper_response(304)

    with pytest.raises(ValueError):
        helper.interpret_response(response)  # type: ignore[arg-type]


def test_interpret_response_modified_returns_modified_result() -> None:
    helper = ConditionalRequestHelper()
    response = _make_helper_response(200)

    result = helper.interpret_response(response)  # type: ignore[arg-type]

    assert isinstance(result, ModifiedResult)
    assert result.etag is None
    assert result.last_modified is None


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


def test_interpret_response_missing_metadata_lists_fields() -> None:
    helper = ConditionalRequestHelper(prior_etag="tag-only")
    response = _make_helper_response(304)

    with pytest.raises(ValueError) as excinfo:
        helper.interpret_response(response)  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "path" in message
    assert "sha256" in message
    assert "content_length" in message


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


def test_conditional_helper_rejects_negative_length() -> None:
    with pytest.raises(ValueError):
        ConditionalRequestHelper(prior_content_length=-1)


def test_interpret_response_requires_response_shape() -> None:
    helper = ConditionalRequestHelper()

    with pytest.raises(TypeError):
        helper.interpret_response(object())  # type: ignore[arg-type]
