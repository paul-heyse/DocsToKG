# === NAVMAP v1 ===
# {
#   "module": "tests.content_download.test_classification_unit",
#   "purpose": "Pytest coverage for content download classification unit scenarios",
#   "sections": [
#     {
#       "id": "test-classify-payload-octet-stream-requires-sniff",
#       "name": "test_classify_payload_octet_stream_requires_sniff",
#       "anchor": "function-test-classify-payload-octet-stream-requires-sniff",
#       "kind": "function"
#     },
#     {
#       "id": "test-classify-payload-octet-stream-pdf-signature",
#       "name": "test_classify_payload_octet_stream_pdf_signature",
#       "anchor": "function-test-classify-payload-octet-stream-pdf-signature",
#       "kind": "function"
#     },
#     {
#       "id": "test-build-download-outcome-respects-head-flag",
#       "name": "test_build_download_outcome_respects_head_flag",
#       "anchor": "function-test-build-download-outcome-respects-head-flag",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

from __future__ import annotations

from types import SimpleNamespace

from DocsToKG.ContentDownload import cli as downloader
from DocsToKG.ContentDownload.core import Classification, ReasonCode, WorkArtifact, classify_payload

# --- Test Cases ---


def test_classify_payload_octet_stream_requires_sniff():
    payload = b"binary without signature"
    assert (
        classify_payload(payload, "application/octet-stream", "https://example.org/file.pdf")
        is Classification.UNKNOWN
    )


def test_classify_payload_octet_stream_pdf_signature():
    payload = b"%PDF-1.5"
    assert (
        classify_payload(payload, "application/octet-stream", "https://example.org/file.pdf")
        is Classification.PDF
    )


def test_build_download_outcome_respects_head_flag(tmp_path):
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    xml_dir = tmp_path / "xml"
    pdf_dir.mkdir()
    html_dir.mkdir()
    xml_dir.mkdir()
    artifact = WorkArtifact(
        work_id="W-test",
        title="Test",
        publication_year=2024,
        doi="10.1000/test",
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
        xml_dir=xml_dir,
    )

    pdf_path = pdf_dir / "small.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n0 1 obj\nendobj\nstartxref\n0\n%%EOF")
    response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/pdf"})

    outcome = downloader._build_download_outcome(  # type: ignore[attr-defined]
        artifact=artifact,
        classification=Classification.PDF,
        dest_path=pdf_path,
        response=response,
        elapsed_ms=12.3,
        flagged_unknown=False,
        sha256="abc",
        content_length=pdf_path.stat().st_size,
        etag='"etag"',
        last_modified="Wed, 01 May 2024 00:00:00 GMT",
        extracted_text_path=None,
        tail_bytes=b"%%EOF",
        dry_run=False,
        head_precheck_passed=True,
    )

    assert outcome.classification is Classification.PDF
    assert outcome.path == str(pdf_path)

    pdf_path.write_bytes(b"short")
    outcome_small = downloader._build_download_outcome(  # type: ignore[attr-defined]
        artifact=artifact,
        classification="pdf",
        dest_path=pdf_path,
        response=response,
        elapsed_ms=1.0,
        flagged_unknown=False,
        sha256="def",
        content_length=pdf_path.stat().st_size,
        etag='"etag"',
        last_modified="Wed, 01 May 2024 00:00:00 GMT",
        extracted_text_path=None,
        tail_bytes=b"short",
        dry_run=False,
        head_precheck_passed=False,
    )

    assert outcome_small.classification is Classification.MISS
    assert outcome_small.path is None
    assert outcome_small.reason is ReasonCode.PDF_TOO_SMALL
