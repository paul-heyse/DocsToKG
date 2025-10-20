"""Tests for the Wayback telemetry implementation."""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from DocsToKG.ContentDownload.telemetry_wayback import (
    AttemptContext,
    AttemptResult,
    CandidateDecision,
    DiscoveryStage,
    JsonlSink,
    ModeSelected,
    PdfDiscoveryMethod,
    SkipReason,
    TelemetrySink,
    TelemetryWayback,
)
from DocsToKG.ContentDownload.telemetry_wayback_sqlite import SQLiteSink, SQLiteTuning


class MonotonicStub:
    """Deterministic monotonic clock for tests."""

    def __init__(self, start: float = 100.0, step: float = 1.0) -> None:
        self.start = start
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        value = self.start + self.step * (self.calls // 2)
        self.calls += 1
        return value


class TestTelemetryWayback:
    """Test cases for TelemetryWayback."""

    @pytest.fixture
    def run_id(self):
        """Create a test run ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def sinks(self):
        """Create test sinks."""
        return []

    @pytest.fixture
    def monotonic_stub(self):
        """Provide a deterministic monotonic clock stub."""
        return MonotonicStub(start=200.0, step=1.5)

    @pytest.fixture
    def telemetry(self, run_id, sinks, monotonic_stub):
        """Create a TelemetryWayback instance."""
        return TelemetryWayback(run_id, sinks, monotonic_fn=monotonic_stub)

    def test_emit_attempt_start(self, telemetry):
        """Test starting an attempt."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
            publication_year=2023,
        )

        assert isinstance(ctx, AttemptContext)
        assert ctx.run_id == telemetry.run_id
        assert ctx.work_id == "work-123"
        assert ctx.artifact_id == "artifact-456"
        assert ctx.original_url == "https://example.com/paper.pdf"
        assert ctx.canonical_url == "https://example.com/paper.pdf"
        assert ctx.publication_year == 2023
        assert ctx.attempt_id is not None

    def test_emit_attempt_end(self, telemetry):
        """Test ending an attempt."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_attempt_end(
            ctx,
            mode_selected=ModeSelected.PDF_DIRECT,
            result=AttemptResult.EMITTED_PDF,
            candidates_scanned=5,
        )

        # Verify context was updated
        assert ctx.monotonic_ms_since_start() >= 0

    def test_attempt_duration_uses_monotonic_stub(self, run_id):
        """Durations should reflect the injected monotonic function."""

        events = []

        class CollectSink:
            def emit(self, event):
                events.append(event)

        monotonic_stub = MonotonicStub(start=42.0, step=2.0)
        telemetry = TelemetryWayback(
            run_id,
            [CollectSink()],
            monotonic_fn=monotonic_stub,
        )

        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_attempt_end(
            ctx,
            mode_selected=ModeSelected.PDF_DIRECT,
            result=AttemptResult.EMITTED_PDF,
            candidates_scanned=3,
        )

        assert len(events) == 2
        start_event, end_event = events
        assert start_event["monotonic_ms"] == 0
        expected_ms = int(monotonic_stub.step * 1000)
        assert end_event["total_duration_ms"] == expected_ms
        assert end_event["monotonic_ms"] == expected_ms

    def test_emit_discovery_availability(self, telemetry):
        """Test emitting availability discovery."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_discovery_availability(
            ctx,
            query_url="https://example.com/paper.pdf",
            year_window="-2..+2",
            http_status=200,
            from_cache=True,
            revalidated=False,
            rate_delay_ms=100,
            retry_after_s=None,
            retry_count=0,
        )

    def test_emit_discovery_cdx(self, telemetry):
        """Test emitting CDX discovery."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_discovery_cdx(
            ctx,
            query_url="https://example.com/paper.pdf",
            year_window="-2..+2",
            limit=8,
            http_status=200,
            returned=3,
            first_ts="20230101000000",
            last_ts="20231201000000",
            from_cache=False,
            revalidated=False,
            rate_delay_ms=200,
            retry_after_s=None,
            retry_count=0,
        )

    def test_emit_candidate(self, telemetry):
        """Test emitting candidate evaluation."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_candidate(
            ctx,
            archive_url="https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
            memento_ts="20230101000000",
            statuscode=200,
            mimetype="application/pdf",
            source_stage=DiscoveryStage.CDX,
            decision=CandidateDecision.HEAD_CHECK,
            distance_to_pub_year=0,
        )

    def test_emit_html_parse(self, telemetry):
        """Test emitting HTML parse event."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_html_parse(
            ctx,
            archive_html_url="https://web.archive.org/web/20230101000000/https://example.com/",
            html_http_status=200,
            from_cache=True,
            revalidated=False,
            html_bytes=5000,
            pdf_link_found=True,
            pdf_discovery_method=PdfDiscoveryMethod.META,
            discovered_pdf_url="https://example.com/paper.pdf",
        )

    def test_emit_pdf_check(self, telemetry):
        """Test emitting PDF check event."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_pdf_check(
            ctx,
            archive_pdf_url="https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
            head_status=200,
            content_type="application/pdf",
            content_length=50000,
            is_pdf_signature=True,
            min_bytes_pass=True,
            decision=CandidateDecision.HEAD_CHECK,
        )

    def test_emit_emit(self, telemetry):
        """Test emitting success event."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_emit(
            ctx,
            emitted_url="https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
            memento_ts="20230101000000",
            source_mode=ModeSelected.PDF_DIRECT,
        )

    def test_emit_skip(self, telemetry):
        """Test emitting skip event."""
        ctx = telemetry.emit_attempt_start(
            work_id="work-123",
            artifact_id="artifact-456",
            original_url="https://example.com/paper.pdf",
            canonical_url="https://example.com/paper.pdf",
        )

        telemetry.emit_skip(
            ctx, reason=SkipReason.NO_SNAPSHOT, details="No snapshots found for this URL"
        )


class TestJsonlSink:
    """Test cases for JsonlSink."""

    def test_emit(self):
        """Test emitting events to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sink_path = Path(tmpdir) / "test.jsonl"
            sink = JsonlSink(sink_path)

            event = {
                "event_type": "wayback_attempt",
                "run_id": "test-run",
                "work_id": "work-123",
                "artifact_id": "artifact-456",
                "attempt_id": "attempt-789",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 100,
                "event": "start",
            }

            sink.emit(event)

            # Verify file was created and contains the event
            assert sink_path.exists()
            with sink_path.open("r") as f:
                line = f.readline().strip()
                data = json.loads(line)
                assert data["event_type"] == "wayback_attempt"
                assert data["run_id"] == "test-run"

    def test_emit_multiple_events(self):
        """Test emitting multiple events to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sink_path = Path(tmpdir) / "test.jsonl"
            sink = JsonlSink(sink_path)

            events = [
                {"event_type": "wayback_attempt", "run_id": "test-run", "event": "start"},
                {"event_type": "wayback_discovery", "run_id": "test-run", "stage": "availability"},
                {
                    "event_type": "wayback_emit",
                    "run_id": "test-run",
                    "emitted_url": "https://example.com",
                },
            ]

            for event in events:
                sink.emit(event)

            # Verify all events were written
            assert sink_path.exists()
            with sink_path.open("r") as f:
                lines = f.readlines()
                assert len(lines) == 3

                for i, line in enumerate(lines):
                    data = json.loads(line.strip())
                    assert data["event_type"] == events[i]["event_type"]


class TestSQLiteSink:
    """Test cases for SQLiteSink."""

    def test_emit_attempt_start(self):
        """Test emitting attempt start event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_attempt",
                "attempt_id": "attempt-123",
                "run_id": "run-456",
                "work_id": "work-789",
                "artifact_id": "artifact-abc",
                "resolver": "wayback",
                "schema": "1",
                "original_url": "https://example.com/paper.pdf",
                "canonical_url": "https://example.com/paper.pdf",
                "publication_year": 2023,
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 0,
                "event": "start",
            }

            sink.emit(event)
            sink.close()

            # Verify database was created and contains the attempt
            assert db_path.exists()

            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_attempts")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute("SELECT attempt_id, run_id, work_id FROM wayback_attempts")
            row = cursor.fetchone()
            assert row[0] == "attempt-123"
            assert row[1] == "run-456"
            assert row[2] == "work-789"

            conn.close()

    def test_emit_attempt_end(self):
        """Test emitting attempt end event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            # First emit start
            start_event = {
                "event_type": "wayback_attempt",
                "attempt_id": "attempt-123",
                "run_id": "run-456",
                "work_id": "work-789",
                "artifact_id": "artifact-abc",
                "resolver": "wayback",
                "schema": "1",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 0,
                "event": "start",
            }
            sink.emit(start_event)

            # Then emit end
            end_event = {
                "event_type": "wayback_attempt",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 1000,
                "event": "end",
                "result": "emitted_pdf",
                "mode_selected": "pdf_direct",
                "total_duration_ms": 1000,
                "candidates_scanned": 5,
            }
            sink.emit(end_event)
            sink.close()

            # Verify both start and end data are present
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT start_ts, end_ts, result, mode_selected FROM wayback_attempts")
            row = cursor.fetchone()
            assert row[0] is not None  # start_ts
            assert row[1] is not None  # end_ts
            assert row[2] == "emitted_pdf"  # result
            assert row[3] == "pdf_direct"  # mode_selected

            conn.close()

    def test_emit_discovery(self):
        """Test emitting discovery event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_discovery",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 100,
                "stage": "availability",
                "query_url": "https://example.com/paper.pdf",
                "year_window": "-2..+2",
                "limit": 8,
                "returned": 3,
                "first_ts": "20230101000000",
                "last_ts": "20231201000000",
                "http_status": 200,
                "from_cache": True,
                "revalidated": False,
                "rate_delay_ms": 100,
                "retry_after_s": None,
                "retry_count": 0,
                "error": None,
            }

            sink.emit(event)
            sink.close()

            # Verify discovery was recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_discoveries")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute("SELECT stage, query_url, http_status FROM wayback_discoveries")
            row = cursor.fetchone()
            assert row[0] == "availability"
            assert row[1] == "https://example.com/paper.pdf"
            assert row[2] == 200

            conn.close()

    def test_emit_candidate(self):
        """Test emitting candidate event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_candidate",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 200,
                "archive_url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
                "memento_ts": "20230101000000",
                "statuscode": 200,
                "mimetype": "application/pdf",
                "source_stage": "cdx",
                "decision": "head_check",
                "distance_to_pub_year": 0,
            }

            sink.emit(event)
            sink.close()

            # Verify candidate was recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_candidates")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute("SELECT archive_url, memento_ts, statuscode FROM wayback_candidates")
            row = cursor.fetchone()
            assert (
                row[0] == "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"
            )
            assert row[1] == "20230101000000"
            assert row[2] == 200

            conn.close()

    def test_emit_html_parse(self):
        """Test emitting HTML parse event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_html_parse",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 300,
                "archive_html_url": "https://web.archive.org/web/20230101000000/https://example.com/",
                "html_http_status": 200,
                "from_cache": True,
                "revalidated": False,
                "html_bytes": 5000,
                "pdf_link_found": True,
                "pdf_discovery_method": "meta",
                "discovered_pdf_url": "https://example.com/paper.pdf",
            }

            sink.emit(event)
            sink.close()

            # Verify HTML parse was recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_html_parses")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute(
                "SELECT archive_html_url, pdf_link_found, pdf_discovery_method FROM wayback_html_parses"
            )
            row = cursor.fetchone()
            assert row[0] == "https://web.archive.org/web/20230101000000/https://example.com/"
            assert row[1] == 1  # True as integer
            assert row[2] == "meta"

            conn.close()

    def test_emit_pdf_check(self):
        """Test emitting PDF check event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_pdf_check",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 400,
                "archive_pdf_url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
                "head_status": 200,
                "content_type": "application/pdf",
                "content_length": 50000,
                "is_pdf_signature": True,
                "min_bytes_pass": True,
                "decision": "head_check",
            }

            sink.emit(event)
            sink.close()

            # Verify PDF check was recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_pdf_checks")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute(
                "SELECT archive_pdf_url, content_type, content_length FROM wayback_pdf_checks"
            )
            row = cursor.fetchone()
            assert (
                row[0] == "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"
            )
            assert row[1] == "application/pdf"
            assert row[2] == 50000

            conn.close()

    def test_emit_emit(self):
        """Test emitting success event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_emit",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 500,
                "emitted_url": "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf",
                "memento_ts": "20230101000000",
                "source_mode": "pdf_direct",
                "http_ct_expected": "application/pdf",
            }

            sink.emit(event)
            sink.close()

            # Verify emit was recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_emits")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute("SELECT emitted_url, memento_ts, source_mode FROM wayback_emits")
            row = cursor.fetchone()
            assert (
                row[0] == "https://web.archive.org/web/20230101000000/https://example.com/paper.pdf"
            )
            assert row[1] == "20230101000000"
            assert row[2] == "pdf_direct"

            conn.close()

    def test_emit_skip(self):
        """Test emitting skip event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "wayback_skip",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 600,
                "reason": "no_snapshot",
                "details": "No snapshots found for this URL",
            }

            sink.emit(event)
            sink.close()

            # Verify skip was recorded
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM wayback_skips")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor.execute("SELECT reason, details FROM wayback_skips")
            row = cursor.fetchone()
            assert row[0] == "no_snapshot"
            assert row[1] == "No snapshots found for this URL"

            conn.close()

    def test_unknown_event_type(self):
        """Test handling of unknown event types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            sink = SQLiteSink(db_path)

            event = {
                "event_type": "unknown_event",
                "attempt_id": "attempt-123",
                "ts": datetime.now(timezone.utc).isoformat(),
                "monotonic_ms": 0,
            }

            # Should not raise an error
            sink.emit(event)
            sink.close()

            # Verify no tables were created for unknown event
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Should only have meta table and wayback tables, no unknown tables
            assert "unknown_event" not in tables

            conn.close()


class TestAttemptContext:
    """Test cases for AttemptContext."""

    def test_monotonic_ms_since_start(self):
        """Test monotonic time calculation."""
        mono = MonotonicStub(start=10.0, step=0.5)
        ctx = AttemptContext(
            run_id="test-run",
            work_id="work-123",
            artifact_id="artifact-456",
            monotonic=mono,
        )

        # First call should be zero because the clock hasn't advanced.
        assert ctx.monotonic_ms_since_start() == 0

        # Next call should advance by the stubbed step (0.5s -> 500ms).
        assert ctx.monotonic_ms_since_start() == int(mono.step * 1000)
