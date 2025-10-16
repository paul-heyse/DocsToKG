import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("requests")
pytest.importorskip("pyalex")

from DocsToKG.ContentDownload.download_pyalex_pdfs import (  # noqa: E402
    CsvSink,
    JsonlSink,
    LastAttemptCsvSink,
    ManifestEntry,
    ManifestIndexSink,
    MultiSink,
)
from DocsToKG.ContentDownload.resolvers import AttemptRecord  # noqa: E402
from scripts.export_attempts_csv import export_attempts_jsonl_to_csv  # noqa: E402


def test_jsonl_sink_attempt_records_include_wall_time(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlSink(log_path)
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
        resolver_wall_time_ms=432.5,
    )
    logger.log_attempt(record)
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["resolver_name"] == "unpaywall"
    assert payload["resolver_wall_time_ms"] == 432.5


def test_multi_sink_synchronizes_timestamps(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "attempts.jsonl"
    csv_path = tmp_path / "attempts.csv"
    record = AttemptRecord(
        work_id="W-sync",
        resolver_name="crossref",
        resolver_order=2,
        url="https://example.org/html",
        status="html",
        http_status=200,
        content_type="text/html",
        elapsed_ms=45.0,
        metadata={},
        dry_run=True,
    )
    with JsonlSink(jsonl_path) as jsonl, CsvSink(csv_path) as csv:
        sink = MultiSink([jsonl, csv])
        sink.log_attempt(record)

    json_payload = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert json_payload["timestamp"] == rows[0]["timestamp"]


def test_csv_sink_close_closes_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "attempts.csv"
    sink = CsvSink(csv_path)
    sink.log_attempt(
        AttemptRecord(
            work_id="W-close",
            resolver_name="unpaywall",
            resolver_order=1,
            url="https://example.org/close",
            status="pdf",
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=12.0,
            metadata={},
            sha256="abc123",
            content_length=256,
            dry_run=False,
            resolver_wall_time_ms=10.5,
        )
    )
    sink.close()
    assert sink._file.closed  # type: ignore[attr-defined]
    sink.close()
    assert sink._file.closed  # type: ignore[attr-defined]


def test_csv_adapter_close_closes_file(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "attempts.jsonl"
    csv_path = tmp_path / "attempts.csv"
    adapter = CsvAttemptLoggerAdapter(JsonlLogger(jsonl_path), csv_path)

    adapter.log_attempt(
        AttemptRecord(
            work_id="W-close",
            resolver_name="unpaywall",
            resolver_order=1,
            url="https://example.org/close",
            status="pdf",
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=12.0,
            metadata={},
            sha256="abc123",
            content_length=256,
            dry_run=False,
            resolver_wall_time_ms=10.5,
        )
    )

    adapter.close()
    assert adapter._file.closed  # type: ignore[attr-defined]

    # ``close`` should be idempotent.
    adapter.close()
    assert adapter._file.closed  # type: ignore[attr-defined]


def test_jsonl_sink_writes_valid_records(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlSink(log_path)

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
    logger = JsonlSink(log_path)
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
        dry_run=True,
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


def _attempt_record(index: int) -> AttemptRecord:
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


def test_jsonl_sink_thread_safety(tmp_path: Path) -> None:
    log_path = tmp_path / "attempts.jsonl"
    logger = JsonlSink(log_path)

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


def test_multi_sink_thread_safety(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "attempts.jsonl"
    csv_path = tmp_path / "attempts.csv"
    with JsonlSink(jsonl_path) as jsonl, CsvSink(csv_path) as csv:
        sink = MultiSink([jsonl, csv])

        def _worker(offset: int) -> None:
            for idx in range(1000):
                sink.log_attempt(_attempt_record(offset * 1000 + idx))

        with ThreadPoolExecutor(max_workers=8) as executor:
            for worker_id in range(8):
                executor.submit(_worker, worker_id)

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8_000
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8_000
    for row in rows[:5]:
        assert row["status"] == "pdf"
        assert row["content_length"] == "2048"


def test_manifest_index_sink_writes_sorted_index(tmp_path: Path) -> None:
    index_path = tmp_path / "manifest.index.json"
    sink = ManifestIndexSink(index_path)
    entries = [
        ManifestEntry(
            timestamp="2024-01-01T00:00:00Z",
            work_id="W2",
            title="Example",
            publication_year=2024,
            resolver="unpaywall",
            url="https://example.org/pdf",
            path="/tmp/example.pdf",
            classification="pdf",
            content_type="application/pdf",
            reason=None,
            html_paths=[],
            sha256="abc",
            content_length=1024,
            etag=None,
            last_modified=None,
            extracted_text_path=None,
            dry_run=False,
        ),
        ManifestEntry(
            timestamp="2024-01-01T00:00:01Z",
            work_id="W1",
            title="HTML",
            publication_year=2023,
            resolver="landing",
            url="https://example.org/html",
            path="/tmp/example.html",
            classification="html",
            content_type="text/html",
            reason=None,
            html_paths=["/tmp/example.html"],
            sha256=None,
            content_length=512,
            etag=None,
            last_modified=None,
            extracted_text_path=None,
            dry_run=False,
        ),
    ]
    for entry in entries:
        sink.log_manifest(entry)
    sink.close()

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert list(payload.keys()) == ["W1", "W2"]
    assert payload["W1"]["classification"] == "html"
    assert payload["W1"]["pdf_path"] is None
    assert payload["W2"]["pdf_path"] == "/tmp/example.pdf"
    assert payload["W2"]["sha256"] == "abc"


def test_last_attempt_csv_sink_writes_latest_entries(tmp_path: Path) -> None:
    csv_path = tmp_path / "manifest.last.csv"
    sink = LastAttemptCsvSink(csv_path)
    first = ManifestEntry(
        timestamp="2024-01-01T00:00:00Z",
        work_id="W-last",
        title="Initial",
        publication_year=2024,
        resolver="unpaywall",
        url="https://example.org/first",
        path="/tmp/first.pdf",
        classification="pdf",
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="111",
        content_length=100,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
    )
    second = ManifestEntry(
        timestamp="2024-01-01T00:01:00Z",
        work_id="W-last",
        title="Updated",
        publication_year=2024,
        resolver="crossref",
        url="https://example.org/second",
        path="/tmp/second.pdf",
        classification="pdf",
        content_type="application/pdf",
        reason=None,
        html_paths=[],
        sha256="222",
        content_length=200,
        etag=None,
        last_modified=None,
        extracted_text_path=None,
        dry_run=False,
    )
    sink.log_manifest(first)
    sink.log_manifest(second)
    sink.close()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "work_id": "W-last",
            "title": "Updated",
            "publication_year": "2024",
            "resolver": "crossref",
            "url": "https://example.org/second",
            "classification": "pdf",
            "path": "/tmp/second.pdf",
            "sha256": "222",
            "content_length": "200",
            "etag": "",
            "last_modified": "",
        }
    ]
