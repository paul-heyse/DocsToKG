# === NAVMAP v1 ===
# {
#   "module": "tests.test_html_extraction",
#   "purpose": "Pytest coverage for html extraction scenarios",
#   "sections": [
#     {
#       "id": "faketrafilatura",
#       "name": "_FakeTrafilatura",
#       "anchor": "class-faketrafilatura",
#       "kind": "class"
#     },
#     {
#       "id": "test-html-extraction-creates-text-file",
#       "name": "test_html_extraction_creates_text_file",
#       "anchor": "function-test-html-extraction-creates-text-file",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
HTML Extraction Tests

This module validates the fallback HTML download path when resolver
candidates return web pages instead of PDFs, ensuring the extracted text
artifact aligns with trafilatura processing expectations.

Key Scenarios:
- Confirms HTML responses persist alongside normalized text derivatives
- Verifies temporary extraction directory management for text outputs

Dependencies:
- pytest: Test orchestration
- requests/responses: HTTP simulation for resolver candidates
- DocsToKG.ContentDownload.cli: Download helpers under test

Usage:
    pytest tests/test_html_extraction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.cli import download_candidate
from DocsToKG.ContentDownload.core import Classification, WorkArtifact
from tests.conftest import PatchManager

# --- Globals ---

requests = pytest.importorskip("requests")
responses = pytest.importorskip("responses")


class _FakeTrafilatura:
    @staticmethod
    def extract(text: str) -> str:
        return text.strip().upper()


@responses.activate
# --- Test Cases ---


def test_html_extraction_creates_text_file(tmp_path: Path, patcher: PatchManager) -> None:
    artifact = WorkArtifact(
        work_id="WHTML",
        title="HTML Example",
        publication_year=2023,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/page"],
        open_access_url=None,
        source_display_names=[],
        base_stem="html-example",
        pdf_dir=tmp_path / "pdf",
        html_dir=tmp_path / "html",
        xml_dir=tmp_path / "xml",
    )

    url = artifact.pdf_urls[0]
    responses.add(responses.HEAD, url, status=200, headers={"Content-Type": "text/html"})
    responses.add(responses.GET, url, status=200, body="<html><body>Hello</body></html>")

    patcher.setitem(sys.modules, "trafilatura", _FakeTrafilatura())

    session = requests.Session()
    context = {"dry_run": False, "extract_html_text": True, "previous": {}}
    outcome = download_candidate(session, artifact, url, None, timeout=15.0, context=context)

    assert outcome.classification is Classification.HTML
    assert outcome.extracted_text_path is not None
    html_path = Path(outcome.path)
    text_path = Path(outcome.extracted_text_path)
    assert html_path.exists()
    assert text_path.exists()
    assert text_path.read_text(encoding="utf-8") == "<HTML><BODY>HELLO</BODY></HTML>"
