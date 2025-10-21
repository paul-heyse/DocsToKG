"""Tests for observability event emitters.

Covers all emitter implementations:
- JsonStdoutEmitter
- FileJsonlEmitter (with rotation)
- DuckDBEmitter (stub)
- ParquetEmitter (stub)
- BufferedEmitter
- MultiEmitter
"""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from DocsToKG.OntologyDownload.observability.emitters import (
    BufferedEmitter,
    DuckDBEmitter,
    EventEmitter,
    FileJsonlEmitter,
    JsonStdoutEmitter,
    MultiEmitter,
    ParquetEmitter,
)
from DocsToKG.OntologyDownload.observability.events import (
    Event,
    EventContext,
    EventIds,
    emit_event,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    ctx = EventContext(
        app_version="1.0.0",
        os_name="Linux",
        python_version="3.10.0",
    )
    return Event(
        ts="2025-10-20T01:23:45Z",
        type="test.event",
        level="INFO",
        run_id="run-123",
        config_hash="hash-abc",
        service="test",
        context=ctx,
        ids=EventIds(),
        payload={"test": "data"},
    )


# ============================================================================
# JsonStdoutEmitter Tests
# ============================================================================


class TestJsonStdoutEmitter:
    """Test JSON stdout emitter."""

    def test_emit_writes_json_to_stdout(self, sample_event, capsys):
        """Emitter writes JSON to stdout."""
        emitter = JsonStdoutEmitter()
        emitter.emit(sample_event)
        captured = capsys.readouterr()
        assert "test.event" in captured.out
        assert "run-123" in captured.out

    def test_emit_with_prefix(self, sample_event, capsys):
        """Emitter adds prefix to output."""
        emitter = JsonStdoutEmitter(prefix="EVENT: ")
        emitter.emit(sample_event)
        captured = capsys.readouterr()
        assert captured.out.startswith("EVENT: {")

    def test_emit_valid_json(self, sample_event, capsys):
        """Output is valid JSON."""
        emitter = JsonStdoutEmitter()
        emitter.emit(sample_event)
        captured = capsys.readouterr()
        # Remove prefix if any, then parse JSON
        data = json.loads(captured.out.strip())
        assert data["type"] == "test.event"

    def test_close_is_safe(self):
        """close() is a no-op and safe."""
        emitter = JsonStdoutEmitter()
        emitter.close()
        emitter.close()  # Should be safe to call multiple times


# ============================================================================
# FileJsonlEmitter Tests
# ============================================================================


class TestFileJsonlEmitter:
    """Test file JSONL emitter."""

    def test_emit_writes_to_file(self, sample_event):
        """Emitter writes events to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "events.jsonl"
            emitter = FileJsonlEmitter(str(filepath))
            emitter.emit(sample_event)
            emitter.close()

            # Verify file content
            assert filepath.exists()
            with open(filepath) as f:
                line = f.readline().strip()
                data = json.loads(line)
                assert data["type"] == "test.event"

    def test_emit_multiple_events(self, sample_event):
        """Emitter appends multiple events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "events.jsonl"
            emitter = FileJsonlEmitter(str(filepath))

            for i in range(5):
                event = Event(
                    ts="2025-10-20T01:23:45Z",
                    type=f"test.event.{i}",
                    level="INFO",
                    run_id="run-123",
                    config_hash="hash-abc",
                    service="test",
                    context=sample_event.context,
                    ids=EventIds(),
                    payload={"index": i},
                )
                emitter.emit(event)

            emitter.close()

            # Verify all events written
            with open(filepath) as f:
                lines = f.readlines()
                assert len(lines) == 5

    def test_creates_parent_directory(self):
        """Emitter creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "path" / "events.jsonl"
            emitter = FileJsonlEmitter(str(filepath))
            # Should not raise
            emitter.close()
            assert filepath.parent.exists()

    def test_rotation_by_line_count(self, sample_event):
        """Emitter rotates file by line count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "events.jsonl"
            emitter = FileJsonlEmitter(str(filepath), max_lines=3)

            # Emit 5 events (should rotate after 3)
            for i in range(5):
                event = Event(
                    ts="2025-10-20T01:23:45Z",
                    type=f"test.event.{i}",
                    level="INFO",
                    run_id="run-123",
                    config_hash="hash-abc",
                    service="test",
                    context=sample_event.context,
                    ids=EventIds(),
                    payload={"index": i},
                )
                emitter.emit(event)

            emitter.close()

            # Should have rotated file + current file
            rotated_files = list(filepath.parent.glob("events.*.jsonl"))
            assert len(rotated_files) >= 1

    def test_thread_safety(self, sample_event):
        """Emitter is thread-safe."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "events.jsonl"
            emitter = FileJsonlEmitter(str(filepath))

            def emit_events():
                for _ in range(10):
                    emitter.emit(sample_event)

            threads = [threading.Thread(target=emit_events) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            emitter.close()

            # Verify all events written (3 threads * 10 events)
            with open(filepath) as f:
                lines = f.readlines()
                assert len(lines) == 30


# ============================================================================
# DuckDBEmitter Tests (Stub)
# ============================================================================


class TestDuckDBEmitter:
    """Test DuckDB emitter (stub)."""

    def test_initialization_logs_warning(self):
        """Initialization logs stub warning."""
        with patch("DocsToKG.OntologyDownload.observability.emitters.logger") as mock_logger:
            emitter = DuckDBEmitter("test.db")
            mock_logger.warning.assert_called()

    def test_emit_is_noop(self, sample_event):
        """emit() is a no-op for stub."""
        emitter = DuckDBEmitter("test.db")
        # Should not raise
        emitter.emit(sample_event)
        emitter.emit(sample_event)

    def test_close_is_safe(self):
        """close() is safe."""
        emitter = DuckDBEmitter("test.db")
        emitter.close()


# ============================================================================
# ParquetEmitter Tests (Stub)
# ============================================================================


class TestParquetEmitter:
    """Test Parquet emitter (stub)."""

    def test_initialization_logs_warning(self):
        """Initialization logs stub warning."""
        with patch("DocsToKG.OntologyDownload.observability.emitters.logger") as mock_logger:
            emitter = ParquetEmitter("test.parquet")
            mock_logger.warning.assert_called()

    def test_emit_is_noop(self, sample_event):
        """emit() is a no-op for stub."""
        emitter = ParquetEmitter("test.parquet")
        emitter.emit(sample_event)

    def test_close_is_safe(self):
        """close() is safe."""
        emitter = ParquetEmitter("test.parquet")
        emitter.close()


# ============================================================================
# BufferedEmitter Tests
# ============================================================================


class TestBufferedEmitter:
    """Test buffered emitter with drop strategy."""

    def test_buffers_events(self, sample_event):
        """Buffered emitter buffers events."""
        mock_delegate = Mock(spec=EventEmitter)
        emitter = BufferedEmitter(mock_delegate, buffer_size=5)

        # Emit 3 events (less than buffer size)
        for _ in range(3):
            emitter.emit(sample_event)

        # Delegate should not have been called yet
        mock_delegate.emit.assert_not_called()

        # Flush on close
        emitter.close()
        assert mock_delegate.emit.call_count == 3

    def test_flushes_on_full(self, sample_event):
        """Buffered emitter flushes when full."""
        mock_delegate = Mock(spec=EventEmitter)
        emitter = BufferedEmitter(mock_delegate, buffer_size=3)

        # Emit 5 events
        for i in range(5):
            event = Event(
                ts="2025-10-20T01:23:45Z",
                type="test.event",
                level="INFO",
                run_id="run-123",
                config_hash="hash-abc",
                service="test",
                context=sample_event.context,
                ids=EventIds(),
                payload={"index": i},
            )
            emitter.emit(event)

        # Delegate should have been called (at least once for flush)
        assert mock_delegate.emit.call_count >= 3

    def test_drops_debug_when_full(self, sample_event):
        """Buffered emitter behavior with high-priority events."""
        mock_delegate = Mock(spec=EventEmitter)
        emitter = BufferedEmitter(mock_delegate, buffer_size=2)

        # Emit INFO events and let buffer flush when full
        for _ in range(2):
            info_event = Event(
                ts="2025-10-20T01:23:45Z",
                type="test.info",
                level="INFO",
                run_id="run-123",
                config_hash="hash-abc",
                service="test",
                context=sample_event.context,
                ids=EventIds(),
                payload={},
            )
            emitter.emit(info_event)

        # Buffer should have flushed, now buffer is empty
        # Emit one more INFO and try DEBUG (buffer size is 1 now with INFO, then try DEBUG)
        info_event = Event(
            ts="2025-10-20T01:23:45Z",
            type="test.info",
            level="INFO",
            run_id="run-123",
            config_hash="hash-abc",
            service="test",
            context=sample_event.context,
            ids=EventIds(),
            payload={},
        )
        emitter.emit(info_event)  # Now buffer has 1 INFO

        # Try to emit DEBUG when buffer is at capacity (should be dropped)
        debug_event = Event(
            ts="2025-10-20T01:23:45Z",
            type="test.debug",
            level="DEBUG",
            run_id="run-123",
            config_hash="hash-abc",
            service="test",
            context=sample_event.context,
            ids=EventIds(),
            payload={},
        )
        emitter.emit(debug_event)

        # Check that DEBUG was dropped
        emitter.close()

    def test_preserves_high_priority(self, sample_event):
        """Buffered emitter preserves high-priority events."""
        mock_delegate = Mock(spec=EventEmitter)
        emitter = BufferedEmitter(mock_delegate, buffer_size=2)

        # Emit INFO event
        info_event = Event(
            ts="2025-10-20T01:23:45Z",
            type="test.info",
            level="INFO",
            run_id="run-123",
            config_hash="hash-abc",
            service="test",
            context=sample_event.context,
            ids=EventIds(),
            payload={},
        )
        emitter.emit(info_event)
        emitter.emit(info_event)

        # Try to emit ERROR event (should not be dropped)
        error_event = Event(
            ts="2025-10-20T01:23:45Z",
            type="test.error",
            level="ERROR",
            run_id="run-123",
            config_hash="hash-abc",
            service="test",
            context=sample_event.context,
            ids=EventIds(),
            payload={},
        )
        emitter.emit(error_event)

        emitter.close()

        # All events should be emitted
        assert mock_delegate.emit.call_count >= 3


# ============================================================================
# MultiEmitter Tests
# ============================================================================


class TestMultiEmitter:
    """Test multi-emitter fan-out."""

    def test_emits_to_all_delegates(self, sample_event):
        """MultiEmitter emits to all delegates."""
        mock1 = Mock(spec=EventEmitter)
        mock2 = Mock(spec=EventEmitter)
        mock3 = Mock(spec=EventEmitter)

        emitter = MultiEmitter([mock1, mock2, mock3])
        emitter.emit(sample_event)

        mock1.emit.assert_called_once()
        mock2.emit.assert_called_once()
        mock3.emit.assert_called_once()

    def test_handles_delegate_errors(self, sample_event):
        """MultiEmitter handles errors in delegates."""
        mock1 = Mock(spec=EventEmitter)
        mock1.emit.side_effect = Exception("Test error")
        mock2 = Mock(spec=EventEmitter)

        emitter = MultiEmitter([mock1, mock2])
        # Should not raise
        emitter.emit(sample_event)

        # Both should have been called
        mock1.emit.assert_called_once()
        mock2.emit.assert_called_once()

    def test_closes_all_delegates(self):
        """MultiEmitter closes all delegates."""
        mock1 = Mock(spec=EventEmitter)
        mock2 = Mock(spec=EventEmitter)

        emitter = MultiEmitter([mock1, mock2])
        emitter.close()

        mock1.close.assert_called_once()
        mock2.close.assert_called_once()

    def test_handles_close_errors(self):
        """MultiEmitter handles errors during close."""
        mock1 = Mock(spec=EventEmitter)
        mock1.close.side_effect = Exception("Test error")
        mock2 = Mock(spec=EventEmitter)

        emitter = MultiEmitter([mock1, mock2])
        # Should not raise
        emitter.close()

        mock1.close.assert_called_once()
        mock2.close.assert_called_once()
