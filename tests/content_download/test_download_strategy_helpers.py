import contextlib
from pathlib import Path
from typing import Dict

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from DocsToKG.ContentDownload import download as downloader
from DocsToKG.ContentDownload.core import Classification, DownloadContext, ReasonCode, WorkArtifact


class _FakeResponse(requests.Response):
    def __init__(
        self, content: bytes, *, status: int = 200, content_type: str = "application/pdf"
    ) -> None:
        super().__init__()
        self._content = content
        self.status_code = status
        self.headers = CaseInsensitiveDict({"Content-Type": content_type})

    def iter_content(self, chunk_size: int) -> bytes:
        yield self._content


@pytest.fixture
def artifact(tmp_path: Path) -> WorkArtifact:
    art = WorkArtifact(
        work_id="W1",
        title="Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.com/foo.pdf"],
        open_access_url=None,
        source_display_names=[],
        base_stem="example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )
    art.pdf_dir.mkdir(parents=True, exist_ok=True)
    art.html_dir.mkdir(parents=True, exist_ok=True)
    art.xml_dir.mkdir(parents=True, exist_ok=True)
    return art


def _build_response(content_type: str = "application/pdf") -> requests.Response:
    return _FakeResponse(b"%PDF-1.4", content_type=content_type)


def test_validate_classification_html_requires_flag(artifact: WorkArtifact) -> None:
    options = downloader.DownloadOptions(
        dry_run=False,
        list_only=False,
        extract_html_text=False,
        run_id="run",
    )
    result = downloader.validate_classification(Classification.HTML, artifact, options)
    assert not result.is_valid
    assert result.detail == "html-disabled"


def test_validate_classification_html_allowed_with_flag(artifact: WorkArtifact) -> None:
    options = downloader.DownloadOptions(
        dry_run=False,
        list_only=False,
        extract_html_text=True,
        run_id="run",
    )
    result = downloader.validate_classification(Classification.HTML, artifact, options)
    assert result.is_valid
    assert result.expected is Classification.HTML


def test_handle_resume_logic_skips_completed(artifact: WorkArtifact) -> None:
    options = downloader.DownloadOptions(
        dry_run=False,
        list_only=False,
        extract_html_text=False,
        run_id="run",
        resume_completed={artifact.work_id},
    )
    decision = downloader.handle_resume_logic(artifact, {}, options)
    assert decision.should_skip
    assert decision.reason is ReasonCode.RESUME_COMPLETE
    assert decision.resolver == "resume"


def test_handle_resume_logic_normalises_metadata(artifact: WorkArtifact) -> None:
    options = downloader.DownloadOptions(
        dry_run=False,
        list_only=False,
        extract_html_text=False,
        run_id="run",
    )
    previous: Dict[str, Dict[str, str]] = {
        "HTTPS://Example.com/file.pdf": {
            "etag": "abc",
            "last_modified": "yesterday",
            "path": str(artifact.pdf_dir / "old.pdf"),
            "sha256": "123",
            "content_length": "10",
        }
    }
    decision = downloader.handle_resume_logic(artifact, previous, options)
    assert not decision.should_skip
    assert "https://example.com/file.pdf" in decision.previous_map


def test_cleanup_sidecar_files_removes_targets(tmp_path: Path, artifact: WorkArtifact) -> None:
    target = artifact.pdf_dir / "example.pdf"
    part = target.with_suffix(".pdf.part")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"data")
    part.write_bytes(b"temp")
    options = downloader.DownloadOptions(
        dry_run=False,
        list_only=False,
        extract_html_text=False,
        run_id="run",
    )
    downloader.cleanup_sidecar_files(artifact, Classification.PDF, options)
    assert not target.exists()
    assert not part.exists()


def test_build_download_outcome_flags_small_pdf(tmp_path: Path, artifact: WorkArtifact) -> None:
    pdf_path = tmp_path / "small.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    response = _build_response()
    outcome = downloader.build_download_outcome(
        artifact=artifact,
        classification=Classification.PDF,
        dest_path=pdf_path,
        response=response,
        elapsed_ms=10.0,
        flagged_unknown=False,
        sha256="abc",
        content_length=4,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
        tail_bytes=b"",
        head_precheck_passed=False,
        min_pdf_bytes=1024,
        tail_check_bytes=2048,
        retry_after=None,
        options=downloader.DownloadOptions(
            dry_run=False,
            list_only=False,
            extract_html_text=False,
            run_id="run",
        ),
    )
    assert outcome.classification is Classification.MISS
    assert outcome.reason is ReasonCode.PDF_TOO_SMALL


def test_pdf_strategy_finalize_returns_pdf(tmp_path: Path, artifact: WorkArtifact) -> None:
    dest_path = tmp_path / "example.pdf"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(b"%PDF-1.4\n%%EOF")
    context = downloader.DownloadStrategyContext(
        download_context=DownloadContext(dry_run=False),
        dest_path=dest_path,
        content_type="application/pdf",
        elapsed_ms=5.0,
        flagged_unknown=False,
        sha256="abc",
        content_length=10,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        tail_bytes=b"%EOF",
        head_precheck_passed=True,
        min_pdf_bytes=4,
        tail_check_bytes=2048,
        retry_after=None,
        classification_hint=Classification.PDF,
        response=_build_response(),
    )
    strategy = downloader.PdfDownloadStrategy()
    outcome = strategy.finalize_artifact(artifact, Classification.PDF, context)
    assert outcome.classification is Classification.PDF


def test_html_strategy_finalize_handles_html(tmp_path: Path, artifact: WorkArtifact) -> None:
    dest_path = tmp_path / "example.html"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text("<html></html>")
    context = downloader.DownloadStrategyContext(
        download_context=DownloadContext(dry_run=False, extract_html_text=True),
        dest_path=dest_path,
        content_type="text/html",
        elapsed_ms=5.0,
        flagged_unknown=False,
        sha256="abc",
        content_length=12,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        tail_bytes=None,
        head_precheck_passed=True,
        min_pdf_bytes=4,
        tail_check_bytes=2048,
        retry_after=None,
        classification_hint=Classification.HTML,
        response=_build_response(content_type="text/html"),
    )
    strategy = downloader.HtmlDownloadStrategy()
    outcome = strategy.finalize_artifact(artifact, Classification.HTML, context)
    assert outcome.classification is Classification.HTML


def test_xml_strategy_finalize_handles_xml(tmp_path: Path, artifact: WorkArtifact) -> None:
    dest_path = tmp_path / "example.xml"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text("<xml />")
    context = downloader.DownloadStrategyContext(
        download_context=DownloadContext(dry_run=False),
        dest_path=dest_path,
        content_type="application/xml",
        elapsed_ms=5.0,
        flagged_unknown=False,
        sha256="abc",
        content_length=12,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        tail_bytes=None,
        head_precheck_passed=True,
        min_pdf_bytes=4,
        tail_check_bytes=2048,
        retry_after=None,
        classification_hint=Classification.XML,
        response=_build_response(content_type="application/xml"),
    )
    strategy = downloader.XmlDownloadStrategy()
    outcome = strategy.finalize_artifact(artifact, Classification.XML, context)
    assert outcome.classification is Classification.XML


def test_strategy_selection_invoked(monkeypatch, artifact: WorkArtifact) -> None:
    calls: Dict[str, Classification] = {}

    class RecordingStrategy(downloader.PdfDownloadStrategy):
        def finalize_artifact(self, artifact, classification, context):  # type: ignore[override]
            calls["classification"] = classification
            return super().finalize_artifact(artifact, classification, context)

    def fake_get_strategy(classification: Classification) -> downloader.DownloadStrategy:
        return RecordingStrategy()

    monkeypatch.setattr(downloader, "get_strategy_for_classification", fake_get_strategy)

    @contextlib.contextmanager
    def fake_request(*args, **kwargs):
        yield _FakeResponse(b"%PDF-1.4\n%EOF")

    monkeypatch.setattr(downloader, "request_with_retries", fake_request)
    session = requests.Session()
    artifact.pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact.html_dir.mkdir(parents=True, exist_ok=True)
    artifact.xml_dir.mkdir(parents=True, exist_ok=True)
    context = DownloadContext(dry_run=True)
    outcome = downloader.download_candidate(
        session,
        artifact,
        artifact.pdf_urls[0],
        referer=None,
        timeout=5.0,
        context=context,
    )
    assert outcome.classification is Classification.PDF
    assert calls["classification"] is Classification.PDF
