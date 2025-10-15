"""Structured logging regression tests for download attempt artefacts."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import (  # noqa: E402
    CsvAttemptLoggerAdapter,
    JsonlLogger,
    ManifestEntry,
)
from DocsToKG.ContentDownload.resolvers import AttemptRecord  # noqa: E402
from scripts.export_attempts_csv import export_attempts_jsonl_to_csv  # noqa: E402


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


def _attempt_record(index: int) -> AttemptRecord:
    """Construct an attempt record with deterministic fields for concurrency tests."""

    return AttemptRecord(
        work_id=f"W{index}",
        resolver_name="unpaywall",
        resolver_order=index % 5,
        url=f"https://example.org/{index}",
        status="pdf",
        http_status=200,
        content_type="application/pdf",
        elapsed_ms=42.0,
        metadata={"idx": index},
        sha256=f"{index:032x}"[-8:],
        content_length=2048,
        dry_run=False,
        resolver_wall_time_ms=12.0,
    )


def test_jsonl_logger_thread_safety(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlLogger(log_path)

    def _worker(offset: int) -> None:
        for idx in range(1000):
            logger.log_attempt(_attempt_record(offset * 1000 + idx))

    with ThreadPoolExecutor(max_workers=16) as executor:
        for worker_id in range(16):
            executor.submit(_worker, worker_id)

    logger.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 16_000
    for line in lines[:10]:
        payload = json.loads(line)
        assert payload["record_type"] == "attempt"
        assert "metadata" in payload


def test_csv_adapter_thread_safety(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    csv_path = tmp_path / "attempts.csv"
    adapter = CsvAttemptLoggerAdapter(JsonlLogger(log_path), csv_path)

    def _worker(offset: int) -> None:
        for idx in range(1000):
            adapter.log_attempt(_attempt_record(offset * 1000 + idx))

    with ThreadPoolExecutor(max_workers=8) as executor:
        for worker_id in range(8):
            executor.submit(_worker, worker_id)

    adapter.close()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 8_000
    for row in rows[:5]:
        assert row["status"] == "pdf"
        assert row["content_length"] == "2048"
