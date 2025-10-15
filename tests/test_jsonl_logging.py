"""
JSONL Logging Integration Tests

This module ensures the JSON Lines logging utilities capture resolver
attempt metadata and remain compatible with CSV export tooling used for
post-processing download telemetry.

Key Scenarios:
- Validates attempt, manifest, and summary records persist correctly
- Confirms CSV export maintains schema and preserves structured metadata

Dependencies:
- pytest: Provides fixtures and assertions
- DocsToKG.ContentDownload.download_pyalex_pdfs: Logging helpers under test
- scripts.export_attempts_csv: CSV conversion routine

Usage:
    pytest tests/test_jsonl_logging.py
"""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import JsonlLogger, ManifestEntry
from DocsToKG.ContentDownload.resolvers.types import AttemptRecord
from scripts.export_attempts_csv import export_attempts_jsonl_to_csv


def test_jsonl_logger_writes_valid_records(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(log_path)

    attempt = AttemptRecord(
        work_id="W1",
        resolver_name="unpaywall",
        resolver_order=1,
        url="https://example.org/pdf",
        status="pdf",
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=123.4,
        reason=None,
        metadata={"source": "test"},
        sha256="deadbeef",
        content_length=1024,
        dry_run=False,
        resolver_wall_time_ms=321.0,
    )
    logger.log_attempt(attempt)

    manifest_entry = ManifestEntry(
        timestamp="2024-01-01T00:00:00Z",
        work_id="W1",
        title="Example",
        publication_year=2024,
        resolver="unpaywall",
        url="https://example.org/pdf",
        path="/tmp/example.pdf",
        classification="pdf",
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="deadbeef",
        content_length=1024,
        etag='"etag"',
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        extracted_text_path=None,
        dry_run=False,
    )
    logger.log_manifest(manifest_entry)

    logger.log_summary({"total_works": 1})
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    record_types = [entry["record_type"] for entry in parsed]
    assert record_types == ["attempt", "manifest", "summary"]
    attempt_record = parsed[0]
    assert attempt_record["metadata"] == {"source": "test"}
    assert attempt_record["sha256"] == "deadbeef"
    assert attempt_record["resolver_wall_time_ms"] == 321.0


def test_export_attempts_csv(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(log_path)
    attempt = AttemptRecord(
        work_id="W2",
        resolver_name="crossref",
        resolver_order=2,
        url="https://example.org/crossref",
        status="http_error",
        http_status=404,
        content_type="text/html",
        elapsed_ms=50.0,
        reason="not found",
        metadata={"status": 404},
        sha256=None,
        content_length=None,
        dry_run=True,
        resolver_wall_time_ms=111.5,
    )
    logger.log_attempt(attempt)
    logger.close()

    csv_path = tmp_path / "attempts.csv"
    export_attempts_jsonl_to_csv(log_path, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "sha256" in reader.fieldnames
    assert len(rows) == 1
    row = rows[0]
    assert row["work_id"] == "W2"
    assert row["resolver_name"] == "crossref"
    assert row["status"] == "http_error"
    assert row["dry_run"] == "True"
    assert row["metadata"] == json.dumps({"status": 404}, sort_keys=True)
    assert row["resolver_wall_time_ms"] == "111.5"
