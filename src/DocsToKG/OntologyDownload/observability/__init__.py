"""Observability: Event system, telemetry, and queryable storage.

This package provides a unified event schema, emission system, and sinks that
make the entire platform observable and queryable. Every subsystem emits
structured events that can be queried to answer operational questions without
grep or ad-hoc parsing.

Modules:
  - events: Canonical Event model and emission API
  - schema: JSON Schema for events (for validation and docs)
  - emitters: Pluggable sinks (stdout, file, DuckDB, Parquet)
  - queries: Pre-built SQL queries for common questions

Example:
    >>> from DocsToKG.OntologyDownload.observability import emit_event, initialize_events
    >>> initialize_events()  # Wire up sinks
    >>> emit_event(
    ...     type="extract.done",
    ...     level="INFO",
    ...     payload={"entries_total": 100, "bytes_written": 50000},
    ... )
"""

import logging
from pathlib import Path
from typing import Optional

from DocsToKG.OntologyDownload.observability.events import (
    Event,
    clear_context,
    emit_event,
    flush_events,
    get_context,
    register_sink,
    set_context,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Event",
    "emit_event",
    "get_context",
    "set_context",
    "clear_context",
    "flush_events",
    "register_sink",
    "initialize_events",
]


def initialize_events(
    enable_stdout: bool = True,
    enable_duckdb: bool = False,
    db_path: Optional[Path] = None,
    enable_file: bool = False,
    file_path: Optional[Path] = None,
) -> None:
    """Initialize event sinks for production deployment.

    Configures which backends receive emitted events:
    - stdout: JSON lines to standard output (for container logging)
    - duckdb: Persistent storage in DuckDB database (for audit trails)
    - file: Append-only JSONL file (for local archiving)

    Args:
        enable_stdout: Send events to stdout (default True for containers)
        enable_duckdb: Persist events to DuckDB database (default False - opt-in)
        db_path: Path to DuckDB database (if enable_duckdb=True)
        enable_file: Write events to file (default False)
        file_path: Path to output file (if enable_file=True)

    Example:
        >>> # Production with database persistence
        >>> initialize_events(
        ...     enable_stdout=True,
        ...     enable_duckdb=True,
        ...     db_path=Path(".catalog/events.duckdb"),
        ... )

        >>> # Lightweight with stdout only
        >>> initialize_events(enable_stdout=True)
    """
    try:
        if enable_stdout:
            from DocsToKG.OntologyDownload.observability.emitters import JsonStdoutEmitter

            stdout_emitter = JsonStdoutEmitter()
            register_sink(stdout_emitter)
            logger.debug("Initialized JSON stdout emitter")

        if enable_duckdb and db_path:
            from DocsToKG.OntologyDownload.observability.emitters import DuckDBEmitter

            db_emitter = DuckDBEmitter(str(db_path), batch_size=100)
            if db_emitter.conn is not None:
                register_sink(db_emitter)
                logger.debug(f"Initialized DuckDB event emitter at {db_path}")
            else:
                logger.warning(
                    "DuckDB emitter failed to initialize; events will not persist to database"
                )

        if enable_file and file_path:
            from DocsToKG.OntologyDownload.observability.emitters import FileJsonlEmitter

            file_emitter = FileJsonlEmitter(str(file_path))
            register_sink(file_emitter)
            logger.debug(f"Initialized file JSONL emitter at {file_path}")

    except Exception as e:
        logger.error(f"Error initializing event sinks: {e}")
        raise
