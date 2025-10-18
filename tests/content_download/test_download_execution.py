from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import pytest
import requests

from DocsToKG.ContentDownload import download as downloader
from DocsToKG.ContentDownload import download_execution as execution
from DocsToKG.ContentDownload.core import Classification
from DocsToKG.ContentDownload.networking import CachedResult, ConditionalRequestHelper


def _make_artifact(tmp_path: Path) -> downloader.WorkArtifact:
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()
    work = {
        "id": "https://openalex.org/W123",
        "title": "Sample Work",
        "publication_year": 2024,
        "ids": {},
    }
    return downloader.create_artifact(work, pdf_dir, html_dir, xml_dir)


def test_prepare_candidate_download_skips_head_precheck(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    session = requests.Session()
    calls = {"count": 0}

    def fake_head_precheck(session: requests.Session, url: str, timeout: float, content_policy=None) -> bool:
        calls["count"] += 1
        return True

    monkeypatch.setattr(execution, "head_precheck", fake_head_precheck)

    result = execution.prepare_candidate_download(
        session=session,
        artifact=artifact,
        url="https://example.org/file.pdf",
        referer=None,
        timeout=5.0,
        context_payload={"skip_head_precheck": True},
    )

    assert result.head_precheck_passed is False
    assert calls["count"] == 0


def test_stream_candidate_payload_returns_cached_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    cached_path = artifact.pdf_dir / f"{artifact.base_stem}.pdf"
    cached_path.write_bytes(b"%PDF-1.7\n1 0 obj\nendobj\n%%EOF\n")

    session = requests.Session()
    previous_record = {
        "path": str(cached_path),
        "content_length": cached_path.stat().st_size,
        "etag": "cached-etag",
        "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "mtime_ns": cached_path.stat().st_mtime_ns if hasattr(cached_path.stat(), "st_mtime_ns") else None,
    }

    preflight = execution.prepare_candidate_download(
        session=session,
        artifact=artifact,
        url="https://example.org/file.pdf",
        referer=None,
        timeout=5.0,
        context_payload={"previous": {"https://example.org/file.pdf": previous_record}},
    )

    cached_result = CachedResult(
        path=str(cached_path),
        sha256="deadbeef",
        content_length=cached_path.stat().st_size,
        etag="cached-etag",
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        recorded_mtime_ns=getattr(cached_path.stat(), "st_mtime_ns", None),
    )

    monkeypatch.setattr(
        preflight.cond_helper,
        "interpret_response",
        lambda response: cached_result,
    )

    class StubResponse:
        status_code = 304

        def __init__(self) -> None:
            self.headers = {"Content-Type": "application/pdf"}

        def iter_content(self, chunk_size: int) -> Iterable[bytes]:
            return iter(())

    class StubContextManager:
        def __enter__(self) -> StubResponse:
            return StubResponse()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(
        execution,
        "request_with_retries",
        lambda *args, **kwargs: StubContextManager(),
    )

    result = execution.stream_candidate_payload(
        session=session,
        artifact=artifact,
        url="https://example.org/file.pdf",
        timeout=5.0,
        preflight=preflight,
        cleanup_sidecar=downloader.cleanup_sidecar_files,
        validate_cached_artifact=downloader._validate_cached_artifact,
        strategy_selector=downloader.get_strategy_for_classification,
        strategy_context_factory=downloader.DownloadStrategyContext,
        content_address_factory=lambda path, sha: path,
    )

    assert result.outcome is not None
    assert result.outcome.classification is Classification.CACHED
    assert result.outcome.path == str(cached_path)
    assert result.outcome.metadata.get("cache_validation_mode") == "fast_path"


def test_stream_candidate_payload_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    session = requests.Session()

    pdf_bytes = b\"%PDF-1.7\\n1 0 obj\\n<< /Type /Catalog >>\\nendobj\\n%%EOF\\n\"

    class StreamingResponse:
        status_code = 200

        def __init__(self, chunks: Iterable[bytes]) -> None:
            self.headers = {
                \"Content-Type\": \"application/pdf\",
                \"Content-Length\": str(sum(len(chunk) for chunk in chunks)),
            }
            self._chunks = list(chunks)

        def iter_content(self, chunk_size: int) -> Iterable[bytes]:
            yield from self._chunks

    class ResponseContextManager:
        def __init__(self, response: StreamingResponse) -> None:
            self._response = response

        def __enter__(self) -> StreamingResponse:
            return self._response

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_request_with_retries(*args, **kwargs) -> ResponseContextManager:
        return ResponseContextManager(StreamingResponse([pdf_bytes]))

    monkeypatch.setattr(execution, \"request_with_retries\", fake_request_with_retries)

    preflight = execution.prepare_candidate_download(
        session=session,
        artifact=artifact,
        url=\"https://example.org/file.pdf\",
        referer=None,
        timeout=5.0,
        context_payload={\"skip_head_precheck\": True},
        head_precheck_passed=True,
    )

    result = execution.stream_candidate_payload(
        session=session,
        artifact=artifact,
        url=\"https://example.org/file.pdf\",
        timeout=5.0,
        preflight=preflight,
        cleanup_sidecar=downloader.cleanup_sidecar_files,
        validate_cached_artifact=downloader._validate_cached_artifact,
        strategy_selector=downloader.get_strategy_for_classification,
        strategy_context_factory=downloader.DownloadStrategyContext,
        content_address_factory=lambda path, sha: path,
    )

    assert result.outcome is None
    assert result.classification is Classification.PDF
    final_outcome = execution.finalize_candidate_download(
        artifact=artifact,
        stream_result=result,
    )
    assert final_outcome.classification is Classification.PDF
    assert final_outcome.path is not None
    assert Path(final_outcome.path).exists()
