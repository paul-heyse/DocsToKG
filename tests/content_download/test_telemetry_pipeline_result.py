"""Tests for :class:`PipelineResult` and ``RunTelemetry.record_pipeline_result``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from DocsToKG.ContentDownload.core import Classification, ReasonCode, WorkArtifact
from DocsToKG.ContentDownload.telemetry import RunTelemetry
from DocsToKG.ContentDownload.telemetry_records import PipelineResult


class _MemorySink:
    """Minimal sink capturing manifest writes for assertions."""

    def __init__(self) -> None:
        self.manifests: list = []

    def log_attempt(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        return

    def log_io_attempt(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        return

    def log_manifest(self, entry: object) -> None:
        self.manifests.append(entry)

    def log_summary(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        return

    def close(self) -> None:  # pragma: no cover
        return


def _build_artifact(tmp_path: Path) -> WorkArtifact:
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()

    return WorkArtifact(
        work_id="W1",
        title="Example Work",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=[],
        open_access_url=None,
        source_display_names=[],
        base_stem="example-work",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
        xml_dir=xml_dir,
    )


def test_record_pipeline_result_emits_manifest(tmp_path: Path) -> None:
    """Pipeline results should be converted into manifest entries without errors."""

    artifact = _build_artifact(tmp_path)
    pdf_file = artifact.pdf_dir / "example.pdf"
    pdf_file.write_bytes(b"%PDF-1.7\n...")
    html_file = artifact.html_dir / "example.html"
    html_file.write_text("<html></html>", encoding="utf-8")

    outcome = SimpleNamespace(
        classification=Classification.PDF,
        path=str(pdf_file),
        content_type="application/pdf",
        reason=None,
        reason_detail=None,
        sha256="deadbeef",
        content_length=pdf_file.stat().st_size,
        etag="etag-123",
        last_modified="Fri, 01 Jan 2025 00:00:00 GMT",
        extracted_text_path=None,
        canonical_url="https://canonical.example/paper.pdf",
        original_url="https://original.example/paper.pdf",
        http_status=200,
        elapsed_ms=123,
    )

    pipeline_result = PipelineResult(
        resolver_name="test-resolver",
        url="https://example.com/paper.pdf",
        outcome=outcome,
        success=True,
        reason=ReasonCode.RATE_LIMITED,
        reason_detail="rate-limited-manual",
        html_paths=[str(html_file)],
        canonical_url="https://canonical.example/paper.pdf",
        original_url="https://original.example/paper.pdf",
    )

    sink = _MemorySink()
    telemetry = RunTelemetry(sink)

    entry = telemetry.record_pipeline_result(
        artifact,
        pipeline_result,
        dry_run=False,
        run_id="run-123",
    )

    assert entry in sink.manifests
    assert entry.resolver == "test-resolver"
    assert entry.url == "https://example.com/paper.pdf"
    assert entry.html_paths == [str(html_file.resolve())]
    assert entry.reason == ReasonCode.RATE_LIMITED.value
    assert entry.reason_detail == "rate-limited-manual"
    assert entry.canonical_url == "https://canonical.example/paper.pdf"
    assert entry.original_url == "https://original.example/paper.pdf"


def test_pipeline_result_normalizes_html_paths() -> None:
    """Iterables of HTML paths are normalised to an immutable tuple of strings."""

    result = PipelineResult(
        resolver_name="resolver",
        url=None,
        html_paths=[Path("relative.html"), "absolute.html"],
    )

    assert result.html_paths == ("relative.html", "absolute.html")


def test_pipeline_result_rejects_string_html_paths() -> None:
    """A plain string is ambiguous and should be rejected for html_paths."""

    with pytest.raises(TypeError):
        PipelineResult(resolver_name="resolver", url=None, html_paths="oops")
