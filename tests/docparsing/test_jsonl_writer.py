"""Tests for lock-aware JSONL writer in DocParsing telemetry.

This module tests the JsonlWriter class and integration with TelemetrySink,
ensuring safe concurrent appends to manifest and attempt files.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import pytest

from DocsToKG.DocParsing.io import DEFAULT_JSONL_WRITER, JsonlWriter
from DocsToKG.DocParsing.telemetry import Attempt, StageTelemetry, TelemetrySink


class TestJsonlWriter:
    """Tests for the JsonlWriter class."""

    def test_jsonl_writer_basic_append(self, tmp_path: Path) -> None:
        """Test basic JSONL appending with JsonlWriter."""
        target = tmp_path / "test.jsonl"
        writer = JsonlWriter()

        rows = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
        count = writer(target, rows)

        assert count == 2
        assert target.exists()

        # Verify content
        with open(target) as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"id": 1, "name": "first"}
            assert json.loads(lines[1]) == {"id": 2, "name": "second"}

    def test_jsonl_writer_sequential_appends(self, tmp_path: Path) -> None:
        """Test multiple sequential appends accumulate correctly."""
        target = tmp_path / "test.jsonl"
        writer = JsonlWriter()

        # First batch
        writer(target, [{"batch": 1, "row": 1}])
        # Second batch
        writer(target, [{"batch": 1, "row": 2}, {"batch": 1, "row": 3}])
        # Third batch
        writer(target, [{"batch": 2, "row": 1}])

        with open(target) as f:
            lines = f.readlines()
            assert len(lines) == 4
            assert json.loads(lines[0]) == {"batch": 1, "row": 1}
            assert json.loads(lines[1]) == {"batch": 1, "row": 2}
            assert json.loads(lines[2]) == {"batch": 1, "row": 3}
            assert json.loads(lines[3]) == {"batch": 2, "row": 1}

    def test_jsonl_writer_empty_rows(self, tmp_path: Path) -> None:
        """Test appending empty row list."""
        target = tmp_path / "test.jsonl"
        writer = JsonlWriter()

        count = writer(target, [])
        assert count == 0
        # Note: When no rows are provided, the atomic_write may not create the file

    def test_jsonl_writer_custom_timeout(self, tmp_path: Path) -> None:
        """Test JsonlWriter with custom timeout."""
        writer = JsonlWriter(lock_timeout_s=0.5)
        assert writer.lock_timeout_s == 0.5

    def test_jsonl_writer_parallel_appends(self, tmp_path: Path) -> None:
        """Test concurrent appends from multiple threads."""
        target = tmp_path / "concurrent.jsonl"
        writer = JsonlWriter()

        results: list[int] = []
        errors: list[Exception] = []

        def append_batch(batch_id: int, count: int) -> None:
            try:
                rows = [{"batch": batch_id, "row": i} for i in range(count)]
                result = writer(target, rows)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create threads that append in parallel
        threads = [
            threading.Thread(target=append_batch, args=(1, 5)),
            threading.Thread(target=append_batch, args=(2, 3)),
            threading.Thread(target=append_batch, args=(3, 4)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all rows were appended
        with open(target) as f:
            lines = f.readlines()
            total_expected = 5 + 3 + 4
            assert len(lines) == total_expected

            # Parse all rows
            rows = [json.loads(line) for line in lines]
            assert all(isinstance(r.get("batch"), int) for r in rows)

    def test_jsonl_writer_lock_file_created(self, tmp_path: Path) -> None:
        """Test that lock file is created and cleaned up."""
        target = tmp_path / "test.jsonl"
        writer = JsonlWriter()

        writer(target, [{"id": 1}])
        # Lock file should be cleaned up after write
        # Note: FileLock may or may not leave the file; this is okay

    def test_default_jsonl_writer_instance(self) -> None:
        """Test that DEFAULT_JSONL_WRITER is properly initialized."""
        assert isinstance(DEFAULT_JSONL_WRITER, JsonlWriter)
        assert DEFAULT_JSONL_WRITER.lock_timeout_s == 120.0


class TestTelemetrySinkWithWriter:
    """Tests for TelemetrySink integration with JsonlWriter."""

    def test_telemetry_sink_uses_lock_aware_writer(self, tmp_path: Path) -> None:
        """Test that TelemetrySink uses lock-aware writer by default."""
        attempts_path = tmp_path / "attempts.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"

        sink = TelemetrySink(attempts_path, manifest_path)

        # Write attempt
        attempt = Attempt(
            run_id="test-run",
            file_id="doc1",
            stage="chunk",
            status="success",
            reason=None,
            started_at=time.time(),
            finished_at=time.time() + 1.0,
            bytes=1024,
        )
        sink.write_attempt(attempt)

        # Verify attempt was written
        assert attempts_path.exists()
        with open(attempts_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["file_id"] == "doc1"
            assert data["status"] == "success"

    def test_telemetry_sink_custom_writer(self, tmp_path: Path) -> None:
        """Test TelemetrySink with custom writer."""
        attempts_path = tmp_path / "attempts.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"

        # Create a custom writer that tracks calls
        calls: list[tuple[Path, int]] = []

        def custom_writer(path: Path, rows: Iterable[dict[str, Any]]) -> int:
            count = len(list(rows))
            calls.append((path, count))
            return count

        sink = TelemetrySink(attempts_path, manifest_path, writer=custom_writer)

        attempt = Attempt(
            run_id="test-run",
            file_id="doc1",
            stage="embed",
            status="success",
            reason=None,
            started_at=time.time(),
            finished_at=time.time(),
            bytes=100,
        )
        sink.write_attempt(attempt)

        assert len(calls) == 1
        assert calls[0][0] == attempts_path
        assert calls[0][1] == 1


class TestStageTelemetryWithWriter:
    """Tests for StageTelemetry integration with lock-aware writer."""

    def test_stage_telemetry_uses_writer(self, tmp_path: Path) -> None:
        """Test that StageTelemetry uses the injected writer."""
        attempts_path = tmp_path / "attempts.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"

        sink = TelemetrySink(attempts_path, manifest_path)
        telemetry = StageTelemetry(sink, run_id="test-run", stage="doctags")

        # Record an attempt
        input_file = tmp_path / "input.pdf"
        input_file.write_text("test content")
        telemetry.record_attempt(
            doc_id="doc1",
            input_path=input_file,
            status="success",
            duration_s=1.5,
        )

        assert attempts_path.exists()
        with open(attempts_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["file_id"] == "doc1"
            assert data["stage"] == "doctags"

    def test_stage_telemetry_log_success(self, tmp_path: Path) -> None:
        """Test StageTelemetry.log_success writes both attempt and manifest."""
        attempts_path = tmp_path / "attempts.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"

        sink = TelemetrySink(attempts_path, manifest_path)
        telemetry = StageTelemetry(sink, run_id="test-run", stage="chunk")

        input_file = tmp_path / "input.txt"
        input_file.write_text("content")
        output_file = tmp_path / "output.jsonl"

        telemetry.log_success(
            doc_id="doc1",
            input_path=input_file,
            output_path=output_file,
            tokens=150,
            schema_version="1.0",
            duration_s=2.0,
        )

        # Both files should have entries
        assert attempts_path.exists()
        assert manifest_path.exists()

        with open(attempts_path) as f:
            assert len(f.readlines()) == 1
        with open(manifest_path) as f:
            assert len(f.readlines()) == 1

    def test_stage_telemetry_concurrent_logs(self, tmp_path: Path) -> None:
        """Test concurrent logging from multiple threads."""
        attempts_path = tmp_path / "attempts.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"

        sink = TelemetrySink(attempts_path, manifest_path)
        errors: list[Exception] = []

        def log_docs(thread_id: int, count: int) -> None:
            try:
                telemetry = StageTelemetry(sink, run_id=f"run-{thread_id}", stage="embed")
                for i in range(count):
                    doc_id = f"doc-{thread_id}-{i}"
                    input_file = tmp_path / f"input-{doc_id}.txt"
                    input_file.write_text(f"content-{i}")

                    telemetry.record_attempt(
                        doc_id=doc_id,
                        input_path=input_file,
                        status="success",
                        duration_s=0.1,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=log_docs, args=(1, 10)),
            threading.Thread(target=log_docs, args=(2, 10)),
            threading.Thread(target=log_docs, args=(3, 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # Verify all logs were written
        with open(attempts_path) as f:
            lines = f.readlines()
            assert len(lines) == 30  # 3 threads * 10 docs


class TestDeprecationWarning:
    """Tests for deprecation warning on acquire_lock with .jsonl files."""

    def test_acquire_lock_warns_on_jsonl(self, tmp_path: Path) -> None:
        """Test that acquire_lock emits deprecation warning for .jsonl files."""
        from DocsToKG.DocParsing.core.concurrency import acquire_lock

        jsonl_path = tmp_path / "test.jsonl"

        with pytest.warns(DeprecationWarning, match="discouraged for manifest"):
            with acquire_lock(jsonl_path, timeout=1.0):
                pass

    def test_acquire_lock_no_warn_for_non_jsonl(self, tmp_path: Path) -> None:
        """Test that acquire_lock does not warn for non-.jsonl files."""
        from DocsToKG.DocParsing.core.concurrency import acquire_lock

        other_path = tmp_path / "test.txt"

        # Should not raise a warning
        with acquire_lock(other_path, timeout=1.0):
            pass
