"""Regression tests for structured resolver logging outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import CsvAttemptLoggerAdapter, JsonlLogger
from DocsToKG.ContentDownload.resolvers.types import AttemptRecord


@pytest.mark.parametrize("wall_ms", [None, 432.5])
def test_jsonl_attempt_records_include_wall_time(tmp_path: Path, wall_ms: float | None) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(log_path)
    record = AttemptRecord(
        work_id="W-1",
        resolver_name="unpaywall",
        resolver_order=1,
        url="https://example.org/pdf",
        status="pdf",
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=120.0,
        metadata={"source": "test"},
        sha256="abc",
        content_length=1024,
        dry_run=False,
        resolver_wall_time_ms=wall_ms,
    )
    logger.log_attempt(record)
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["resolver_name"] == "unpaywall"
    assert payload["resolver_wall_time_ms"] == wall_ms


def test_csv_adapter_emits_wall_time_column(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "attempts.jsonl"
    csv_path = tmp_path / "attempts.csv"
    base_logger = JsonlLogger(jsonl_path)
    adapter = CsvAttemptLoggerAdapter(base_logger, csv_path)
    record = AttemptRecord(
        work_id="W-2",
        resolver_name="crossref",
        resolver_order=2,
        url="https://example.org/html",
        status="html",
        http_status=200,
        content_type="text/html",
        elapsed_ms=45.0,
        metadata={},
        sha256=None,
        content_length=None,
        dry_run=True,
        resolver_wall_time_ms=987.6,
    )
    adapter.log_attempt(record)
    adapter.close()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "resolver_wall_time_ms" in reader.fieldnames
    assert rows[0]["resolver_wall_time_ms"] == "987.6"
