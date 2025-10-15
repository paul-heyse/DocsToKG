"""
Atomic Download Tests

This module validates the downloader's atomic write guarantees by
simulating partial failures and verifying checksum bookkeeping for
successful PDF retrievals.

Key Scenarios:
- Ensures incomplete downloads leave `.part` files for forensic retries
- Confirms successful downloads compute digests and clean partial files

Dependencies:
- pytest: Assertions and fixtures
- DocsToKG.ContentDownload.download_pyalex_pdfs: Streaming download logic

Usage:
    pytest tests/test_atomic_writes.py
"""

import hashlib
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

import requests

from DocsToKG.ContentDownload.download_pyalex_pdfs import WorkArtifact, download_candidate


class _DummyHeadResponse:
    status_code = 200
    headers = {"Content-Type": "application/pdf"}

    def close(self) -> None:  # pragma: no cover - no side effects
        return


class _BaseDummyResponse:
    def __init__(self, status_code: int = 200, headers: Dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers: Dict[str, str] = {"Content-Type": "application/pdf"}
        if headers:
            self.headers.update(headers)

    def __enter__(self) -> "_BaseDummyResponse":  # noqa: D401 - context manager protocol
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - context manager protocol
        return None

    def close(self) -> None:  # pragma: no cover - no resources to release
        return


class _FailingResponse(_BaseDummyResponse):
    def iter_content(self, chunk_size: int):  # noqa: D401 - streaming interface
        yield b"%PDF-1.4\n"
        raise requests.exceptions.ChunkedEncodingError("simulated failure")


class _SuccessfulResponse(_BaseDummyResponse):
    def __init__(self, payload: bytes, headers: Dict[str, str] | None = None) -> None:
        super().__init__(status_code=200, headers=headers)
        self._payload = payload

    def iter_content(self, chunk_size: int):  # noqa: D401 - streaming interface
        yield self._payload


class _DummySession:
    def __init__(self, response: _BaseDummyResponse) -> None:
        self._response = response

    def head(self, url: str, **kwargs: Any) -> _DummyHeadResponse:  # noqa: D401
        return _DummyHeadResponse()

    def get(self, url: str, **kwargs: Any) -> _BaseDummyResponse:  # noqa: D401
        return self._response


def _make_artifact(tmp_path: Path) -> WorkArtifact:
    pdf_dir = tmp_path / "pdfs"
    html_dir = tmp_path / "html"
    pdf_dir.mkdir()
    html_dir.mkdir()
    return WorkArtifact(
        work_id="W-atomic",
        title="Atomic Test",
        publication_year=2024,
        doi="10.1234/atomic",
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="atomic",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
    )


def _download_with_session(
    session: _DummySession, tmp_path: Path
) -> tuple[WorkArtifact, Path, Dict[str, Dict[str, Any]]]:
    artifact = _make_artifact(tmp_path)
    context: Dict[str, Dict[str, Any]] = {"previous": {}}
    outcome = download_candidate(
        session,
        artifact,
        "https://example.org/test.pdf",
        referer=None,
        timeout=5.0,
        context=context,
    )
    return artifact, artifact.pdf_dir / "atomic.pdf", context, outcome


def test_partial_download_leaves_part_file(tmp_path: Path) -> None:
    session = _DummySession(_FailingResponse())
    artifact, final_path, _, outcome = _download_with_session(session, tmp_path)

    part_path = final_path.with_suffix(".pdf.part")
    assert outcome.classification == "request_error"
    assert not final_path.exists()
    assert part_path.exists()


def test_successful_download_records_digest(tmp_path: Path) -> None:
    body = b"A" * 1500
    payload = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n" + body + b"\n%%EOF\n"
    expected_sha = hashlib.sha256(payload).hexdigest()
    session = _DummySession(_SuccessfulResponse(payload))

    artifact, final_path, _, outcome = _download_with_session(session, tmp_path)

    assert outcome.classification == "pdf"
    assert outcome.sha256 == expected_sha
    assert outcome.content_length == len(payload)
    assert final_path.exists()
    assert not final_path.with_suffix(".pdf.part").exists()
    assert final_path.read_bytes() == payload
