from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

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
        "etag": "\"etag\"",
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
        etag="\"tag\"",
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
    entry = build_manifest_entry(artifact, "resolver", "https://example.org", outcome, [], dry_run=False)
    assert isinstance(entry, ManifestEntry)
    assert entry.etag == "\"tag\""
    assert entry.last_modified == "Mon, 01 Jan 2024 00:00:00 GMT"
