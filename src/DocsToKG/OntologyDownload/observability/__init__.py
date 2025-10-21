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
    >>> from DocsToKG.OntologyDownload.observability import emit_event
    >>> emit_event(
    ...     type="extract.done",
    ...     level="INFO",
    ...     payload={"entries_total": 100, "bytes_written": 50000},
    ... )
"""

from DocsToKG.OntologyDownload.observability.events import (
    Event,
    clear_context,
    emit_event,
    get_context,
    set_context,
)

__all__ = [
    "Event",
    "emit_event",
    "get_context",
    "set_context",
    "clear_context",
]
