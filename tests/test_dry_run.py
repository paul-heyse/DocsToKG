"""
Dry Run Behaviour Tests

This module asserts that dry-run execution of the downloader preserves
logging outputs without touching the filesystem, ensuring staging runs
can collect metrics safely.

Key Scenarios:
- Verifies `download_candidate` avoids writing artifacts in dry-run mode
- Confirms manifest entries are still emitted for analytics

Dependencies:
- pytest/responses: HTTP mocking of resolver downloads
- DocsToKG.ContentDownload: Dry-run pathways under test

Usage:
    pytest tests/test_dry_run.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import (
    JsonlLogger,
    WorkArtifact,
    download_candidate,
    process_one_work,
)
from DocsToKG.ContentDownload.resolvers import DownloadOutcome, PipelineResult, ResolverMetrics

requests = pytest.importorskip("requests")
responses = pytest.importorskip("responses")


def _make_artifact(tmp_path: Path) -> WorkArtifact:
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    return WorkArtifact(
        work_id="W1",
        title="Dry Run Example",
        publication_year=2024,
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        landing_urls=[],
        pdf_urls=["https://example.org/paper.pdf"],
        open_access_url=None,
        source_display_names=[],
        base_stem="dry-run-example",
        pdf_dir=pdf_dir,
        html_dir=html_dir,
    )


@responses.activate
def test_download_candidate_dry_run_does_not_create_files(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    session = requests.Session()
    url = artifact.pdf_urls[0]
    responses.add(responses.HEAD, url, status=200, headers={"Content-Type": "application/pdf"})
    responses.add(responses.GET, url, status=200, body=b"%PDF-1.4\n%%EOF\n")

    context = {"dry_run": True, "extract_html_text": False, "previous": {}}
    outcome = download_candidate(session, artifact, url, None, timeout=30.0, context=context)

    assert outcome.classification == "pdf"
    assert outcome.path is None
    assert list((artifact.pdf_dir).glob("*.pdf")) == []


class _StubPipeline:
    def run(self, session, artifact, context=None):  # pragma: no cover - interface shim
        return PipelineResult(
            success=True,
            resolver_name="stub",
            url="https://example.org/paper.pdf",
            outcome=DownloadOutcome(
                classification="pdf",
                path=None,
                http_status=200,
                content_type="application/pdf",
                elapsed_ms=12.0,
            ),
            html_paths=[],
        )


def test_process_one_work_logs_manifest_in_dry_run(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    work = {
        "id": "https://openalex.org/W1",
        "title": artifact.title,
        "publication_year": artifact.publication_year,
        "ids": {},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }

    session = requests.Session()
    logger_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(logger_path)
    metrics = ResolverMetrics()

    result = process_one_work(
        work,
        session,
        artifact.pdf_dir,
        artifact.html_dir,
        pipeline=_StubPipeline(),
        logger=logger,
        metrics=metrics,
        dry_run=True,
        extract_html_text=False,
        previous_lookup={},
        resume_completed=set(),
    )

    logger.close()

    assert result["saved"] is True
    contents = [
        json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    manifest_records = [entry for entry in contents if entry["record_type"] == "manifest"]
    assert manifest_records, "Expected at least one manifest record"
    assert all(record["dry_run"] is True for record in manifest_records)
