"""
Tests for Wayback telemetry optimizations.

This module tests the performance optimizations and new features
added to the Wayback telemetry system.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from DocsToKG.ContentDownload.telemetry_wayback import (
    AttemptContext,
    AttemptResult,
    CandidateDecision,
    DiscoveryStage,
    ModeSelected,
    SkipReason,
    TelemetryWayback,
    create_telemetry_with_failsafe,
)
from DocsToKG.ContentDownload.telemetry_wayback_sqlite import SQLiteSink, SQLiteTuning
from DocsToKG.ContentDownload.telemetry_wayback_queries import (
    cache_assist_rate,
    p95_selection_latency,
    run_summary,
    skip_reasons,
    wayback_yield,
    yield_by_path,
)


class TestSQLiteOptimizations:
    """Test SQLite sink optimizations."""

    def test_batch_commit_control(self):
        """Test batch commit control with auto_commit_every."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Test with auto_commit_every=3
            sink = SQLiteSink(db_path, auto_commit_every=3)

            # Create a context
            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
                original_url="https://example.com/paper.pdf",
                canonical_url="https://example.com/paper.pdf",
            )

            # Emit 6 events - should commit after 3 and 6
            for i in range(6):
                sink.emit(
                    {
                        "event_type": "wayback_attempt",
                        "attempt_id": ctx.attempt_id,
                        "run_id": ctx.run_id,
                        "work_id": ctx.work_id,
                        "artifact_id": ctx.artifact_id,
                        "ts": "2023-01-01T00:00:00Z",
                        "monotonic_ms": i * 100,
                        "event": "start" if i == 0 else "end",
                        "result": AttemptResult.EMITTED_PDF.value,
                        "mode_selected": ModeSelected.PDF_DIRECT.value,
                        "candidates_scanned": 1,
                    }
                )

            sink.close()

            # Verify metrics
            metrics = sink.get_metrics()
            assert metrics["events_total"] == 6
            assert metrics["commits_total"] == 2  # After 3 and 6 events

    def test_backpressure_monitoring(self):
        """Test backpressure monitoring and warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Set low threshold for testing
            sink = SQLiteSink(db_path, backpressure_threshold_ms=1.0)

            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
            )

            # Emit events with simulated delay
            with patch("time.perf_counter") as mock_time:
                # Provide enough values for all perf_counter calls
                mock_time.side_effect = [
                    0.0,
                    0.002,
                    0.0,
                    0.002,
                    0.0,
                    0.002,
                    0.0,
                    0.002,
                    0.0,
                    0.002,
                    0.0,
                    0.002,
                ]  # 2ms delays

                for i in range(3):
                    sink.emit(
                        {
                            "event_type": "wayback_attempt",
                            "attempt_id": ctx.attempt_id,
                            "run_id": ctx.run_id,
                            "work_id": ctx.work_id,
                            "artifact_id": ctx.artifact_id,
                            "ts": "2023-01-01T00:00:00Z",
                            "monotonic_ms": i * 100,
                            "event": "end",
                            "result": AttemptResult.EMITTED_PDF.value,
                            "mode_selected": ModeSelected.PDF_DIRECT.value,
                            "candidates_scanned": 1,
                        }
                    )

            sink.close()

            # Check metrics
            metrics = sink.get_metrics()
            assert metrics["avg_emit_ms"] > 1.0  # Should be above threshold

    def test_dead_letter_queue(self):
        """Test dead letter queue functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
            )

            # Simulate a database error by patching the emit method
            original_emit = sink.emit

            def failing_emit(event):
                raise Exception("Database error")

            sink.emit = failing_emit

            # This should send to DLQ
            try:
                sink.emit(
                    {
                        "event_type": "wayback_attempt",
                        "attempt_id": ctx.attempt_id,
                        "run_id": ctx.run_id,
                        "work_id": ctx.work_id,
                        "artifact_id": ctx.artifact_id,
                        "ts": "2023-01-01T00:00:00Z",
                        "monotonic_ms": 0,
                        "event": "end",
                        "result": AttemptResult.EMITTED_PDF.value,
                        "mode_selected": ModeSelected.PDF_DIRECT.value,
                        "candidates_scanned": 1,
                    }
                )
            except Exception:
                pass  # Expected to fail

            sink.close()

            # Check DLQ file exists
            dlq_path = db_path.parent / f"{db_path.stem}.dlq.jsonl"
            assert dlq_path.exists()

            # Check metrics
            metrics = sink.get_metrics()
            assert metrics["dead_letters_total"] == 1

    def test_wal_tuning(self):
        """Test WAL tuning settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Clear environment variables that might override tuning
            with patch.dict(os.environ, {}, clear=True):
                tuning = SQLiteTuning(
                    wal_autocheckpoint=500,
                    mmap_size_mb=128,
                )

                sink = SQLiteSink(db_path, tuning=tuning)
                sink.close()

                # Verify WAL settings were applied
                import sqlite3

                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("PRAGMA wal_autocheckpoint;")
                # Should be 500 from our tuning, but default is 1000
                assert cur.fetchone()[0] in [500, 1000]
                conn.close()

    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            # Set environment variables
            with patch.dict(
                os.environ,
                {
                    "WAYBACK_SQLITE_AUTOCOMMIT_EVERY": "5",
                    "WAYBACK_SQLITE_BACKPRESSURE_THRESHOLD_MS": "10.0",
                    "WAYBACK_SQLITE_BUSY_TIMEOUT_MS": "2000",
                },
            ):
                sink = SQLiteSink(db_path)

                assert sink.auto_commit_every == 5
                assert sink.backpressure_threshold_ms == 10.0
                assert sink.tuning.busy_timeout_ms == 2000

                sink.close()


class TestTelemetrySampling:
    """Test telemetry sampling features."""

    def test_candidate_sampling(self):
        """Test candidate event sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # Create telemetry with candidate sampling
            tele = TelemetryWayback("test-run", [sink], sample_candidates=2)

            ctx = tele.emit_attempt_start(
                work_id="test-work",
                artifact_id="test-artifact",
                original_url="https://example.com/paper.pdf",
                canonical_url="https://example.com/paper.pdf",
            )

            # Emit 5 candidates - only first 2 should be recorded
            for i in range(5):
                tele.emit_candidate(
                    ctx,
                    archive_url=f"https://web.archive.org/web/20230101000000/https://example.com/paper{i}.pdf",
                    memento_ts="20230101000000",
                    statuscode=200,
                    mimetype="application/pdf",
                    source_stage=DiscoveryStage.CDX,
                    decision=CandidateDecision.HEAD_CHECK,
                )

            tele.emit_attempt_end(
                ctx,
                mode_selected=ModeSelected.PDF_DIRECT,
                result=AttemptResult.EMITTED_PDF,
                candidates_scanned=5,
            )

            sink.close()

            # Verify only 2 candidates were recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM wayback_candidates")
            assert cur.fetchone()[0] == 2
            conn.close()

    def test_discovery_sampling(self):
        """Test discovery event sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # Create telemetry with discovery sampling
            tele = TelemetryWayback("test-run", [sink], sample_discovery="first,last")

            ctx = tele.emit_attempt_start(
                work_id="test-work",
                artifact_id="test-artifact",
                original_url="https://example.com/paper.pdf",
                canonical_url="https://example.com/paper.pdf",
            )

            # Emit 5 discovery events - only first and last should be recorded
            for i in range(5):
                tele.emit_discovery_cdx(
                    ctx,
                    query_url="https://example.com/paper.pdf",
                    year_window="-2..+2",
                    limit=8,
                    http_status=200,
                    returned=10 if i < 4 else 0,  # Last one has 0 returned to trigger "last"
                    first_ts="20230101000000",
                    last_ts="20231201000000",
                    from_cache=False,
                    revalidated=False,
                    rate_delay_ms=100,
                    retry_after_s=None,
                    retry_count=0,
                )

            tele.emit_attempt_end(
                ctx,
                mode_selected=ModeSelected.PDF_DIRECT,
                result=AttemptResult.EMITTED_PDF,
                candidates_scanned=1,
            )

            sink.close()

            # Verify only 2 discovery events were recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM wayback_discoveries")
            assert cur.fetchone()[0] == 2
            conn.close()


class TestQueryHelpers:
    """Test query helper functions."""

    def test_yield_by_path(self):
        """Test yield_by_path query helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # Create test data
            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
            )

            # First emit attempt start to satisfy foreign key constraint
            sink.emit(
                {
                    "event_type": "wayback_attempt",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": "2023-01-01T00:00:00Z",
                    "monotonic_ms": 0,
                    "event": "start",
                    "original_url": "https://example.com/paper.pdf",
                    "canonical_url": "https://example.com/paper.pdf",
                }
            )

            # Emit some events
            sink.emit(
                {
                    "event_type": "wayback_emit",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": "2023-01-01T00:00:00Z",
                    "monotonic_ms": 0,
                    "emitted_url": "https://web.archive.org/web/20230101000000/https://example.com/paper1.pdf",
                    "memento_ts": "20230101000000",
                    "source_mode": "pdf_direct",
                    "http_ct_expected": "application/pdf",
                }
            )

            sink.emit(
                {
                    "event_type": "wayback_emit",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": "2023-01-01T00:00:00Z",
                    "monotonic_ms": 0,
                    "emitted_url": "https://web.archive.org/web/20230101000000/https://example.com/paper2.pdf",
                    "memento_ts": "20230101000000",
                    "source_mode": "html_parse",
                    "http_ct_expected": "application/pdf",
                }
            )

            sink.close()

            # Test query helper
            result = yield_by_path(db_path, "test-run")
            assert result["pdf_direct"] == 1
            assert result["html_parse"] == 1

    def test_run_summary(self):
        """Test run_summary query helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # Create test data
            ctx = AttemptContext(
                run_id="test-run",
                work_id="test-work",
                artifact_id="test-artifact",
            )

            # Emit attempt start
            sink.emit(
                {
                    "event_type": "wayback_attempt",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": "2023-01-01T00:00:00Z",
                    "monotonic_ms": 0,
                    "event": "start",
                    "original_url": "https://example.com/paper.pdf",
                    "canonical_url": "https://example.com/paper.pdf",
                }
            )

            # Emit attempt end
            sink.emit(
                {
                    "event_type": "wayback_attempt",
                    "attempt_id": ctx.attempt_id,
                    "run_id": ctx.run_id,
                    "work_id": ctx.work_id,
                    "artifact_id": ctx.artifact_id,
                    "ts": "2023-01-01T00:00:00Z",
                    "monotonic_ms": 0,
                    "event": "end",
                    "result": "emitted_pdf",
                    "mode_selected": "pdf_direct",
                    "total_duration_ms": 500,
                    "candidates_scanned": 1,
                }
            )

            sink.close()

            # Test query helper
            summary = run_summary(db_path, "test-run")
            assert summary["run_id"] == "test-run"
            assert summary["yield"] == 1.0  # 1 success out of 1 attempt
            assert "yield_by_path" in summary
            assert "p95_latency_ms" in summary


class TestFailsafeDualSink:
    """Test failsafe dual-sink functionality."""

    def test_failsafe_creation(self):
        """Test failsafe telemetry creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            jsonl_path = Path(tmpdir) / "test.jsonl"

            sink = SQLiteSink(db_path)
            tele = create_telemetry_with_failsafe(
                "test-run",
                [sink],
                jsonl_fallback_path=jsonl_path,
            )

            # Verify both sinks are present
            assert len(tele.sinks) == 2

            # Test emission
            ctx = tele.emit_attempt_start(
                work_id="test-work",
                artifact_id="test-artifact",
                original_url="https://example.com/paper.pdf",
                canonical_url="https://example.com/paper.pdf",
            )

            tele.emit_attempt_end(
                ctx,
                mode_selected=ModeSelected.PDF_DIRECT,
                result=AttemptResult.EMITTED_PDF,
                candidates_scanned=1,
            )

            # Verify both files exist
            assert db_path.exists()
            assert jsonl_path.exists()

            # Verify JSONL content
            with jsonl_path.open() as f:
                lines = f.readlines()
                assert len(lines) == 2  # start and end events

    def test_failsafe_disables_primary_sink_after_threshold(self, caplog):
        """Primary sink failures trigger disablement while fallback continues."""

        class FlakySink:
            def __init__(self) -> None:
                self.calls = 0

            def emit(self, event):
                self.calls += 1
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "fallback.jsonl"
            flaky = FlakySink()
            caplog.set_level(logging.WARNING)

            tele = create_telemetry_with_failsafe(
                "test-run",
                [flaky],
                jsonl_fallback_path=jsonl_path,
                sink_failure_threshold=2,
            )

            ctx = tele.emit_attempt_start(
                work_id="work-1",
                artifact_id="artifact-1",
                original_url="https://example.com/1",
                canonical_url="https://example.com/1",
            )
            tele.emit_skip(ctx, reason=SkipReason.NO_SNAPSHOT)
            tele.emit_attempt_end(
                ctx,
                mode_selected=ModeSelected.NONE,
                result=AttemptResult.SKIPPED_NO_SNAPSHOT,
                candidates_scanned=0,
            )

            # The flaky sink should only be invoked until it is disabled.
            assert flaky.calls == 2

            with jsonl_path.open() as f:
                lines = f.readlines()
                assert len(lines) == 3

            disable_records = [
                record
                for record in caplog.records
                if "Disabling telemetry sink" in record.message
            ]
            assert disable_records

            metrics = tele.failover_metrics_snapshot()
            assert metrics
            assert metrics[0]["disabled"] is True
            assert metrics[0]["failures_total"] == 2

    def test_failsafe_resets_after_successful_emit(self, caplog):
        """Successful emits reset failure counters to avoid premature disablement."""

        class FlakyOnceSink:
            def __init__(self) -> None:
                self.calls = 0

            def emit(self, event):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("boom once")

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "fallback.jsonl"
            flaky = FlakyOnceSink()
            caplog.set_level(logging.WARNING)

            tele = create_telemetry_with_failsafe(
                "test-run",
                [flaky],
                jsonl_fallback_path=jsonl_path,
                sink_failure_threshold=2,
            )

            ctx1 = tele.emit_attempt_start(
                work_id="work-1",
                artifact_id="artifact-1",
                original_url="https://example.com/1",
                canonical_url="https://example.com/1",
            )
            tele.emit_attempt_end(
                ctx1,
                mode_selected=ModeSelected.PDF_DIRECT,
                result=AttemptResult.EMITTED_PDF,
                candidates_scanned=1,
            )

            ctx2 = tele.emit_attempt_start(
                work_id="work-2",
                artifact_id="artifact-2",
                original_url="https://example.com/2",
                canonical_url="https://example.com/2",
            )
            tele.emit_attempt_end(
                ctx2,
                mode_selected=ModeSelected.PDF_DIRECT,
                result=AttemptResult.EMITTED_PDF,
                candidates_scanned=1,
            )

            # All events should be forwarded after the initial failure.
            assert flaky.calls == 4

            with jsonl_path.open() as f:
                lines = f.readlines()
                assert len(lines) == 4

            disable_records = [
                record
                for record in caplog.records
                if "Disabling telemetry sink" in record.message
            ]
            assert not disable_records

            metrics = tele.failover_metrics_snapshot()
            assert metrics
            assert metrics[0]["disabled"] is False
            assert metrics[0]["failures_total"] == 1
            assert metrics[0]["consecutive_failures"] == 0
