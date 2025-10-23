# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.observability.emitters",
#   "purpose": "Pluggable event sinks for observability.",
#   "sections": [
#     {
#       "id": "eventemitter",
#       "name": "EventEmitter",
#       "anchor": "class-eventemitter",
#       "kind": "class"
#     },
#     {
#       "id": "jsonstdoutemitter",
#       "name": "JsonStdoutEmitter",
#       "anchor": "class-jsonstdoutemitter",
#       "kind": "class"
#     },
#     {
#       "id": "filejsonlemitter",
#       "name": "FileJsonlEmitter",
#       "anchor": "class-filejsonlemitter",
#       "kind": "class"
#     },
#     {
#       "id": "duckdbemitter",
#       "name": "DuckDBEmitter",
#       "anchor": "class-duckdbemitter",
#       "kind": "class"
#     },
#     {
#       "id": "parquetemitter",
#       "name": "ParquetEmitter",
#       "anchor": "class-parquetemitter",
#       "kind": "class"
#     },
#     {
#       "id": "bufferedemitter",
#       "name": "BufferedEmitter",
#       "anchor": "class-bufferedemitter",
#       "kind": "class"
#     },
#     {
#       "id": "multiemitter",
#       "name": "MultiEmitter",
#       "anchor": "class-multiemitter",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Pluggable event sinks for observability.

Implementations for writing events to different backends:
- JsonStdoutEmitter: Write JSON to stdout (for container logging)
- FileJsonlEmitter: Append-only file with rotation support
- DuckDBEmitter: Insert into DuckDB events table (for SQL queries)
- ParquetEmitter: Write to Parquet logs (optional)
- BufferedEmitter: Buffering with drop strategy

All emitters implement the simple `emit(event: Event)` interface.
"""

import logging
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from DocsToKG.OntologyDownload.observability.events import Event

logger = logging.getLogger(__name__)


# ============================================================================
# Base Emitter Class
# ============================================================================


class EventEmitter(ABC):
    """Abstract base for event sinks."""

    @abstractmethod
    def emit(self, event) -> None:
        """Emit an event to this sink.

        Args:
            event: Event object with to_json() method

        Should not raise; emitter errors are logged but don't propagate.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the emitter and release resources.

        Safe to call multiple times.
        """
        pass


# ============================================================================
# JSON Stdout Emitter
# ============================================================================


class JsonStdoutEmitter(EventEmitter):
    """Write events to stdout as JSON (one per line).

    Useful for container logging where stdout is captured and forwarded.
    """

    def __init__(self, prefix: str = ""):
        """Initialize stdout emitter.

        Args:
            prefix: Optional prefix for each line (e.g., "EVENT: ")
        """
        self.prefix = prefix

    def emit(self, event) -> None:
        """Write event as JSON line to stdout."""
        try:
            line = event.to_json()
            if self.prefix:
                line = f"{self.prefix}{line}"
            print(line, flush=True)
        except Exception as e:
            logger.error(f"Error emitting to stdout: {e}")

    def close(self) -> None:
        """No-op for stdout."""
        pass


# ============================================================================
# File JSONL Emitter
# ============================================================================


class FileJsonlEmitter(EventEmitter):
    """Append-only JSONL file with optional rotation.

    Each event is written as a JSON line. Supports rotation by size or count.
    """

    def __init__(
        self,
        filepath: str,
        max_size_bytes: int | None = None,
        max_lines: int | None = None,
    ):
        """Initialize file emitter.

        Args:
            filepath: Path to output file
            max_size_bytes: Rotate when file exceeds this size (None = no rotation)
            max_lines: Rotate when file exceeds this line count (None = no rotation)
        """
        self.filepath = Path(filepath)
        self.max_size_bytes = max_size_bytes or 100 * 1024 * 1024  # 100 MB default
        self.max_lines = max_lines or 100000  # 100K lines default
        self.line_count = 0
        self.lock = threading.Lock()

        # Create parent directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Count existing lines if file exists
        if self.filepath.exists():
            try:
                with open(self.filepath) as f:
                    self.line_count = sum(1 for _ in f)
            except Exception as e:
                logger.warning(f"Could not count existing lines: {e}")

    def emit(self, event) -> None:
        """Append event as JSON line to file."""
        try:
            with self.lock:
                # Check if rotation needed
                if self._should_rotate():
                    self._rotate()

                # Write event
                with open(self.filepath, "a") as f:
                    f.write(event.to_json() + "\n")
                    f.flush()

                self.line_count += 1

        except Exception as e:
            logger.error(f"Error emitting to file: {e}")

    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        if self.line_count >= self.max_lines:
            return True
        if self.filepath.exists() and self.filepath.stat().st_size >= self.max_size_bytes:
            return True
        return False

    def _rotate(self) -> None:
        """Rotate the log file."""
        if not self.filepath.exists():
            return

        # Rename current file with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup_path = self.filepath.parent / f"{self.filepath.stem}.{timestamp}.jsonl"

        try:
            self.filepath.rename(backup_path)
            logger.info(f"Rotated log file to {backup_path}")
            self.line_count = 0
        except Exception as e:
            logger.error(f"Error rotating log file: {e}")

    def close(self) -> None:
        """Close file handle (implicit on flush)."""
        pass


# ============================================================================
# DuckDB Emitter (Full Implementation)
# ============================================================================


class DuckDBEmitter(EventEmitter):
    """Write events to DuckDB events table with batch collection."""

    def __init__(self, db_path: str, batch_size: int = 100):
        """Initialize DuckDB emitter.

        Args:
            db_path: Path to DuckDB database file
            batch_size: Number of events to batch before flushing
        """
        try:
            import duckdb
        except ImportError:
            logger.warning("duckdb not installed; DuckDBEmitter will not persist events")
            self.conn = None
            return

        self.db_path = db_path
        self.batch_size = batch_size
        self._batch_buffer: list[Event] = []
        self._lock = threading.Lock()

        try:
            self.conn = duckdb.connect(str(db_path))
            self._create_table()
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB connection to {db_path}: {e}")
            self.conn = None

    def _create_table(self) -> None:
        """Create events table if not exists."""
        if not self.conn:
            return

        try:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    ts TIMESTAMP,
                    type VARCHAR,
                    level VARCHAR,
                    run_id VARCHAR,
                    config_hash VARCHAR,
                    service VARCHAR,
                    app_version VARCHAR,
                    os_name VARCHAR,
                    python_version VARCHAR,
                    libarchive_version VARCHAR,
                    hostname VARCHAR,
                    pid INTEGER,
                    version_id VARCHAR,
                    artifact_id VARCHAR,
                    file_id VARCHAR,
                    request_id VARCHAR,
                    payload JSON
                )
            """
            )
            logger.debug("Created events table in DuckDB")
        except Exception as e:
            logger.error(f"Failed to create events table: {e}")

    def emit(self, event: Event) -> None:
        """Buffer event and flush when batch size reached."""
        if not self.conn:
            return

        try:
            with self._lock:
                self._batch_buffer.append(event)

                if len(self._batch_buffer) >= self.batch_size:
                    self._flush()

        except Exception as e:
            logger.error(f"Error buffering event: {e}")

    def _flush(self) -> None:
        """Flush buffered events to DuckDB."""
        if not self.conn or not self._batch_buffer:
            return

        try:
            for event in self._batch_buffer:
                self.conn.execute(
                    """
                    INSERT INTO events (
                        ts, type, level, run_id, config_hash, service,
                        app_version, os_name, python_version, libarchive_version,
                        hostname, pid, version_id, artifact_id, file_id, request_id, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        event.ts,
                        event.type,
                        event.level,
                        event.run_id,
                        event.config_hash,
                        event.service,
                        event.context.app_version,
                        event.context.os_name,
                        event.context.python_version,
                        event.context.libarchive_version,
                        event.context.hostname,
                        event.context.pid,
                        event.ids.version_id,
                        event.ids.artifact_id,
                        event.ids.file_id,
                        event.ids.request_id,
                        event.payload,
                    ],
                )

            self.conn.commit()
            self._batch_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing events to DuckDB: {e}")
            self._batch_buffer.clear()

    def close(self) -> None:
        """Flush pending events and close connection."""
        if not self.conn:
            return

        try:
            with self._lock:
                self._flush()
                # Note: DuckDB connections don't require explicit closing
        except Exception as e:
            logger.error(f"Error closing DuckDB emitter: {e}")


# ============================================================================
# Parquet Emitter (Full Implementation)
# ============================================================================


class ParquetEmitter(EventEmitter):
    """Write events to Parquet file with batch collection."""

    def __init__(self, filepath: str, batch_size: int = 1000):
        """Initialize Parquet emitter.

        Args:
            filepath: Path to output Parquet file
            batch_size: Number of events to batch before writing
        """
        try:
            import pyarrow as pa
        except ImportError:
            logger.warning("pyarrow not installed; ParquetEmitter will not persist events")
            self.pa = None
            return

        self.pa = pa
        self.filepath = Path(filepath)
        self.batch_size = batch_size
        self._batch_buffer: list[dict] = []
        self._lock = threading.Lock()
        self._row_count = 0

        # Create parent directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Event) -> None:
        """Buffer event and flush when batch size reached."""
        if not self.pa:
            return

        try:
            event_dict = event.to_dict()
            # Flatten context and ids for Parquet
            event_dict.update({f"context_{k}": v for k, v in event_dict.pop("context", {}).items()})
            event_dict.update({f"ids_{k}": v for k, v in event_dict.pop("ids", {}).items()})

            with self._lock:
                self._batch_buffer.append(event_dict)

                if len(self._batch_buffer) >= self.batch_size:
                    self._flush()

        except Exception as e:
            logger.error(f"Error buffering event for Parquet: {e}")

    def _flush(self) -> None:
        """Flush buffered events to Parquet."""
        if not self.pa or not self._batch_buffer:
            return

        try:
            # Convert batch to PyArrow Table
            table = self.pa.Table.from_pylist(self._batch_buffer)

            # Write or append to Parquet
            if self.filepath.exists():
                # Append mode: read existing and append
                existing_table = self.pa.parquet.read_table(str(self.filepath))
                combined_table = self.pa.concat_tables([existing_table, table])
                self.pa.parquet.write_table(combined_table, str(self.filepath))
            else:
                # Write new file
                self.pa.parquet.write_table(table, str(self.filepath))

            self._row_count += len(self._batch_buffer)
            self._batch_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing events to Parquet: {e}")
            self._batch_buffer.clear()

    def close(self) -> None:
        """Flush remaining events and close."""
        if not self.pa:
            return

        try:
            with self._lock:
                self._flush()
        except Exception as e:
            logger.error(f"Error closing Parquet emitter: {e}")


# ============================================================================
# Buffered Emitter (with Drop Strategy)
# ============================================================================


class BufferedEmitter(EventEmitter):
    """Buffered emitter with drop strategy for high-volume scenarios.

    Useful for performance-critical paths where we want to drop DEBUG
    events if the buffer gets too full, while preserving INFO/WARN/ERROR.
    """

    def __init__(self, delegate: EventEmitter, buffer_size: int = 1000):
        """Initialize buffered emitter.

        Args:
            delegate: Underlying emitter to delegate to
            buffer_size: Max events in buffer before drops occur
        """
        self.delegate = delegate
        self.buffer_size = buffer_size
        self.buffer: list[Event] = []
        self.lock = threading.Lock()
        self.dropped = 0

    def emit(self, event) -> None:
        """Buffer event, dropping DEBUG if full."""
        try:
            with self.lock:
                # If buffer is full and event is DEBUG, drop it
                if len(self.buffer) >= self.buffer_size and event.level == "DEBUG":
                    self.dropped += 1
                    return

                # Add to buffer
                self.buffer.append(event)

                # Flush if buffer is full (keeping high-priority events)
                if len(self.buffer) >= self.buffer_size:
                    self._flush()

        except Exception as e:
            logger.error(f"Error buffering event: {e}")

    def _flush(self) -> None:
        """Flush buffered events to delegate."""
        for event in self.buffer:
            try:
                self.delegate.emit(event)
            except Exception as e:
                logger.error(f"Error flushing event to delegate: {e}")

        self.buffer.clear()

    def close(self) -> None:
        """Flush remaining events and close delegate."""
        with self.lock:
            self._flush()
            if self.dropped > 0:
                logger.info(f"BufferedEmitter dropped {self.dropped} DEBUG events")

        self.delegate.close()


# ============================================================================
# Multi-Sink Emitter
# ============================================================================


class MultiEmitter(EventEmitter):
    """Fan-out to multiple emitters."""

    def __init__(self, emitters: list):
        """Initialize with list of emitters."""
        self.emitters = emitters

    def emit(self, event) -> None:
        """Emit to all registered emitters."""
        for emitter in self.emitters:
            try:
                emitter.emit(event)
            except Exception as e:
                logger.error(f"Error in {emitter.__class__.__name__}: {e}")

    def close(self) -> None:
        """Close all emitters."""
        for emitter in self.emitters:
            try:
                emitter.close()
            except Exception as e:
                logger.error(f"Error closing {emitter.__class__.__name__}: {e}")


__all__ = [
    "EventEmitter",
    "JsonStdoutEmitter",
    "FileJsonlEmitter",
    "DuckDBEmitter",
    "ParquetEmitter",
    "BufferedEmitter",
    "MultiEmitter",
]
