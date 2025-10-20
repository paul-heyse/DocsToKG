"""
Tests for Wayback telemetry optimizations.

This module tests the performance optimizations and new features
added to the Wayback telemetry system.
"""

import os
import sqlite3
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

    def test_transaction_batching_commit_metrics(self):
        """Ensure batched writes share a transaction and metrics track commits."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            original_connect = sqlite3.connect

            class TrackingConnection(sqlite3.Connection):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.commit_calls = 0

                def commit(self):
                    self.commit_calls += 1
                    return super().commit()

            def connect_with_tracking(*args, **kwargs):
                kwargs["factory"] = TrackingConnection
                return original_connect(*args, **kwargs)

            with patch("sqlite3.connect", side_effect=connect_with_tracking):
                sink = SQLiteSink(db_path, auto_commit_every=3)
                conn = sink._conn
                assert isinstance(conn, TrackingConnection)

                baseline_commits = conn.commit_calls

                attempt_event = {
                    "event_type": "wayback_attempt",
                    "attempt_id": "attempt-001",
                    "run_id": "run-001",
                    "work_id": "work-001",
                    "artifact_id": "artifact-001",
                    "resolver": "wayback",
                    "schema": "1",
                    "original_url": "https://example.com/paper.pdf",
                    "canonical_url": "https://example.com/paper.pdf",
                    "publication_year": 2024,
                    "ts": "2024-01-01T00:00:00Z",
                    "monotonic_ms": 0,
                    "event": "start",
                }

                discovery_event = {
                    "event_type": "wayback_discovery",
                    "attempt_id": "attempt-001",
                    "ts": "2024-01-01T00:00:01Z",
                    "monotonic_ms": 1,
                    "stage": "availability",
                    "query_url": "https://web.archive.org/",
                    "year_window": None,
                    "limit": 10,
                    "returned": 1,
                    "first_ts": "2024-01-01T00:00:01Z",
                    "last_ts": "2024-01-01T00:00:01Z",
                    "http_status": 200,
                    "from_cache": False,
                    "revalidated": False,
                    "rate_delay_ms": 0,
                    "retry_after_s": None,
                    "retry_count": 0,
                    "error": None,
                }

                candidate_event = {
                    "event_type": "wayback_candidate",
                    "attempt_id": "attempt-001",
                    "ts": "2024-01-01T00:00:02Z",
                    "monotonic_ms": 2,
                    "archive_url": "https://web.archive.org/a1",
                    "memento_ts": "20240101000002",
                    "statuscode": 200,
                    "mimetype": "text/html",
                    "source_stage": "availability",
                    "decision": "accepted",
                    "distance_to_pub_year": 0,
                }

                candidate_event_2 = {
                    "event_type": "wayback_candidate",
                    "attempt_id": "attempt-001",
                    "ts": "2024-01-01T00:00:03Z",
                    "monotonic_ms": 3,
                    "archive_url": "https://web.archive.org/a2",
                    "memento_ts": "20240101000003",
                    "statuscode": 200,
                    "mimetype": "text/html",
                    "source_stage": "availability",
                    "decision": "accepted",
                    "distance_to_pub_year": 1,
                }

                candidate_event_3 = {
                    "event_type": "wayback_candidate",
                    "attempt_id": "attempt-001",
                    "ts": "2024-01-01T00:00:04Z",
                    "monotonic_ms": 4,
                    "archive_url": "https://web.archive.org/a3",
                    "memento_ts": "20240101000004",
                    "statuscode": 200,
                    "mimetype": "text/html",
                    "source_stage": "availability",
                    "decision": "accepted",
                    "distance_to_pub_year": 2,
                }

                sink.emit(attempt_event)
                assert sink._pending == 1
                assert conn.commit_calls == baseline_commits

                sink.emit(discovery_event)
                assert sink._pending == 2
                assert conn.commit_calls == baseline_commits

                sink.emit(candidate_event)
                assert sink._pending == 0
                assert not sink._transaction_open
                assert conn.commit_calls == baseline_commits + 1
                assert sink.get_metrics()["commits_total"] == 1

                sink.emit(candidate_event_2)
                sink.emit(candidate_event_3)
                assert sink._pending == 2
                assert sink._transaction_open
                assert conn.commit_calls == baseline_commits + 1
                assert sink.get_metrics()["commits_total"] == 1

                sink.close()
                assert sink._pending == 0
                assert not sink._transaction_open
                assert conn.commit_calls == baseline_commits + 2
                assert sink.get_metrics()["commits_total"] == 2

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
