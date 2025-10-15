"""
Resume Workflow Tests

This module verifies resume functionality that skips already-processed
works while still emitting manifest records so downstream tooling can
track skipped documents.

Key Scenarios:
- Ensures pipeline is bypassed when work identifier is marked completed
- Confirms manifest entries reflect skipped state without dry-run flags

Dependencies:
- pytest: Assertions and fixtures
- DocsToKG.ContentDownload: Resume-aware processing helpers

Usage:
    pytest tests/test_resume.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("requests")
import requests

from DocsToKG.ContentDownload.download_pyalex_pdfs import JsonlLogger, process_one_work
from DocsToKG.ContentDownload.resolvers import ResolverMetrics


class _NoopPipeline:
    def run(self, session, artifact, context=None):  # pragma: no cover - should not run
        raise AssertionError("Pipeline should not execute when resume skips work")


def _build_work() -> dict:
    return {
        "id": "https://openalex.org/W-RESUME",
        "title": "Resume Example",
        "publication_year": 2020,
        "ids": {},
        "best_oa_location": {},
        "primary_location": {},
        "locations": [],
        "open_access": {"oa_url": None},
    }


def test_resume_skips_completed_work(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdf"
    html_dir = tmp_path / "html"
    session = requests.Session()
    logger_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(logger_path)
    metrics = ResolverMetrics()

    result = process_one_work(
        _build_work(),
        session,
        pdf_dir,
        html_dir,
        pipeline=_NoopPipeline(),
        logger=logger,
        metrics=metrics,
        dry_run=False,
        extract_html_text=False,
        previous_lookup={},
        resume_completed={"W-RESUME"},
    )

    logger.close()

    assert result["skipped"] is True
    entries = [json.loads(line) for line in logger_path.read_text(encoding="utf-8").strip().splitlines()]
    manifest_entries = [entry for entry in entries if entry["record_type"] == "manifest"]
    assert len(manifest_entries) == 1
    manifest = manifest_entries[0]
    assert manifest["work_id"] == "W-RESUME"
    assert manifest["classification"] == "skipped"
    assert manifest["dry_run"] is False
