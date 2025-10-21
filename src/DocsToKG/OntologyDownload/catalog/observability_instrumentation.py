# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.observability_instrumentation",
#   "purpose": "Observability wiring for catalog operations (Task 1.3)",
#   "sections": [
#     {"id": "boundary_events", "name": "Boundary Event Helpers", "anchor": "BOUND", "kind": "utils"},
#     {"id": "doctor_events", "name": "Doctor Operation Events", "anchor": "DOC", "kind": "utils"},
#     {"id": "prune_events", "name": "Prune Operation Events", "anchor": "PRUNE", "kind": "utils"},
#     {"id": "cli_events", "name": "CLI Command Events", "anchor": "CLI", "kind": "utils"},
#     {"id": "perf_events", "name": "Performance Events", "anchor": "PERF", "kind": "utils"}
#   ]
# }
# === /NAVMAP ===

"""Observability instrumentation for DuckDB catalog operations.

Provides helper functions and decorators for emitting structured events
throughout catalog operations for comprehensive observability, debugging,
and performance monitoring.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import asdict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Context variables for event correlation
_operation_start_time: ContextVar[float] = ContextVar("operation_start_time", default=0.0)

try:
    from ..observability.events import emit_event, EventIds, Event
except ImportError:
    def emit_event(*args, **kwargs) -> None:  # type: ignore
        """Fallback no-op emitter if observability not available."""
        pass


# ============================================================================
# BOUNDARY EVENT HELPERS (BOUND)
# ============================================================================


def emit_boundary_begin(
    boundary: str,
    artifact_id: str,
    version_id: str,
    service: str,
    extra_payload: Optional[dict] = None,
) -> None:
    """Emit boundary operation begin event.

    Args:
        boundary: Boundary name (download, extract, validate, latest)
        artifact_id: Artifact SHA256 hash
        version_id: Version identifier
        service: Service name
        extra_payload: Additional event fields
    """
    payload = {
        "boundary": boundary,
        "artifact_id": artifact_id,
        "version_id": version_id,
        "service": service,
        "phase": "begin",
    }
    if extra_payload:
        payload.update(extra_payload)

    emit_event(
        event_type=f"boundary.{boundary}.begin",
        level="INFO",
        ids=EventIds(artifact_id=artifact_id, version_id=version_id),
        payload=payload,
    )


def emit_boundary_success(
    boundary: str,
    artifact_id: str,
    version_id: str,
    duration_ms: float,
    extra_payload: Optional[dict] = None,
) -> None:
    """Emit boundary operation success event.

    Args:
        boundary: Boundary name
        artifact_id: Artifact SHA256
        version_id: Version ID
        duration_ms: Operation duration in milliseconds
        extra_payload: Additional event fields
    """
    payload = {
        "boundary": boundary,
        "artifact_id": artifact_id,
        "version_id": version_id,
        "phase": "success",
        "duration_ms": round(duration_ms, 1),
    }
    if extra_payload:
        payload.update(extra_payload)

    emit_event(
        event_type=f"boundary.{boundary}.success",
        level="INFO",
        ids=EventIds(artifact_id=artifact_id, version_id=version_id),
        payload=payload,
    )


def emit_boundary_error(
    boundary: str,
    artifact_id: str,
    version_id: str,
    error: Exception,
    duration_ms: float,
    extra_payload: Optional[dict] = None,
) -> None:
    """Emit boundary operation error event.

    Args:
        boundary: Boundary name
        artifact_id: Artifact SHA256
        version_id: Version ID
        error: Exception that occurred
        duration_ms: Operation duration before error
        extra_payload: Additional event fields
    """
    payload = {
        "boundary": boundary,
        "artifact_id": artifact_id,
        "version_id": version_id,
        "phase": "error",
        "duration_ms": round(duration_ms, 1),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    if extra_payload:
        payload.update(extra_payload)

    emit_event(
        event_type=f"boundary.{boundary}.error",
        level="ERROR",
        ids=EventIds(artifact_id=artifact_id, version_id=version_id),
        payload=payload,
    )


# ============================================================================
# DOCTOR OPERATION EVENTS (DOC)
# ============================================================================


def emit_doctor_begin() -> None:
    """Emit doctor operation begin event."""
    emit_event(
        event_type="catalog.doctor.begin",
        level="INFO",
        payload={"operation": "doctor", "phase": "begin"},
    )


def emit_doctor_issue_found(
    issue_type: str,
    severity: str,
    affected_records: int,
    details: Optional[dict] = None,
) -> None:
    """Emit doctor issue found event.

    Args:
        issue_type: Type of issue (missing_file, orphan_record, mismatch)
        severity: Severity level (warning, error)
        affected_records: Number of affected records
        details: Additional issue details
    """
    payload = {
        "operation": "doctor",
        "phase": "issue_found",
        "issue_type": issue_type,
        "severity": severity,
        "affected_records": affected_records,
    }
    if details:
        payload["details"] = details

    emit_event(
        event_type="catalog.doctor.issue_found",
        level="WARN" if severity == "warning" else "ERROR",
        payload=payload,
    )


def emit_doctor_fixed(issue_type: str, count: int) -> None:
    """Emit doctor issue fixed event.

    Args:
        issue_type: Type of issue that was fixed
        count: Number of issues fixed
    """
    payload = {
        "operation": "doctor",
        "phase": "fixed",
        "issue_type": issue_type,
        "fixed_count": count,
    }
    emit_event(
        event_type="catalog.doctor.fixed",
        level="INFO",
        payload=payload,
    )


def emit_doctor_complete(total_issues: int, fixed: int, duration_ms: float) -> None:
    """Emit doctor operation complete event.

    Args:
        total_issues: Total issues found
        fixed: Issues fixed
        duration_ms: Total operation duration
    """
    payload = {
        "operation": "doctor",
        "phase": "complete",
        "total_issues": total_issues,
        "fixed": fixed,
        "duration_ms": round(duration_ms, 1),
    }
    emit_event(
        event_type="catalog.doctor.complete",
        level="INFO",
        payload=payload,
    )


# ============================================================================
# PRUNE OPERATION EVENTS (PRUNE)
# ============================================================================


def emit_prune_begin(dry_run: bool = False) -> None:
    """Emit prune operation begin event.

    Args:
        dry_run: Whether this is a dry-run
    """
    emit_event(
        event_type="catalog.prune.begin",
        level="INFO",
        payload={
            "operation": "prune",
            "phase": "begin",
            "dry_run": dry_run,
        },
    )


def emit_prune_orphan_found(
    path: str,
    size_bytes: int,
    age_days: Optional[int] = None,
) -> None:
    """Emit prune orphan found event.

    Args:
        path: Relative path of orphan file
        size_bytes: Size of orphan file in bytes
        age_days: Age of file in days
    """
    payload = {
        "operation": "prune",
        "phase": "orphan_found",
        "path": path,
        "size_bytes": size_bytes,
    }
    if age_days is not None:
        payload["age_days"] = age_days

    emit_event(
        event_type="catalog.prune.orphan_found",
        level="INFO",
        payload=payload,
    )


def emit_prune_deleted(count: int, total_bytes: int, duration_ms: float) -> None:
    """Emit prune completion event.

    Args:
        count: Number of files deleted
        total_bytes: Total bytes freed
        duration_ms: Operation duration
    """
    payload = {
        "operation": "prune",
        "phase": "complete",
        "deleted_count": count,
        "freed_bytes": total_bytes,
        "duration_ms": round(duration_ms, 1),
    }
    emit_event(
        event_type="catalog.prune.complete",
        level="INFO",
        payload=payload,
    )


# ============================================================================
# CLI COMMAND EVENTS (CLI)
# ============================================================================


def emit_cli_command_begin(command: str, args: Optional[dict] = None) -> float:
    """Emit CLI command begin event.

    Args:
        command: Command name
        args: Command arguments

    Returns:
        Start time for later duration calculation
    """
    start_time = time.time()

    payload = {
        "command": command,
        "phase": "begin",
    }
    if args:
        payload["args"] = args

    emit_event(
        event_type=f"cli.{command}.begin",
        level="INFO",
        payload=payload,
    )

    return start_time


def emit_cli_command_success(
    command: str,
    duration_ms: float,
    result_summary: Optional[dict] = None,
) -> None:
    """Emit CLI command success event.

    Args:
        command: Command name
        duration_ms: Command duration in milliseconds
        result_summary: Summary of command results
    """
    payload = {
        "command": command,
        "phase": "success",
        "duration_ms": round(duration_ms, 1),
    }
    if result_summary:
        payload["result_summary"] = result_summary

    emit_event(
        event_type=f"cli.{command}.success",
        level="INFO",
        payload=payload,
    )


def emit_cli_command_error(
    command: str,
    duration_ms: float,
    error: Exception,
) -> None:
    """Emit CLI command error event.

    Args:
        command: Command name
        duration_ms: Command duration before error
        error: Exception that occurred
    """
    payload = {
        "command": command,
        "phase": "error",
        "duration_ms": round(duration_ms, 1),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    emit_event(
        event_type=f"cli.{command}.error",
        level="ERROR",
        payload=payload,
    )


# ============================================================================
# PERFORMANCE EVENTS (PERF)
# ============================================================================


def emit_slow_operation(
    operation: str,
    duration_ms: float,
    threshold_ms: float = 1000,
    details: Optional[dict] = None,
) -> None:
    """Emit slow operation warning event.

    Args:
        operation: Operation name
        duration_ms: Actual duration
        threshold_ms: Threshold for "slow"
        details: Additional operation details
    """
    if duration_ms < threshold_ms:
        return  # Not slow enough to warn

    payload = {
        "operation": operation,
        "duration_ms": round(duration_ms, 1),
        "threshold_ms": threshold_ms,
        "slowdown_factor": round(duration_ms / threshold_ms, 1),
    }
    if details:
        payload["details"] = details

    emit_event(
        event_type="perf.slow_operation",
        level="WARN",
        payload=payload,
    )


def emit_slow_query(
    query_type: str,
    duration_ms: float,
    rows_examined: int,
    threshold_ms: float = 500,
) -> None:
    """Emit slow query warning event.

    Args:
        query_type: Type of query
        duration_ms: Query duration
        rows_examined: Number of rows examined
        threshold_ms: Threshold for "slow"
    """
    if duration_ms < threshold_ms:
        return  # Not slow enough to warn

    payload = {
        "query_type": query_type,
        "duration_ms": round(duration_ms, 1),
        "threshold_ms": threshold_ms,
        "rows_examined": rows_examined,
        "rows_per_ms": round(rows_examined / (duration_ms or 1), 1),
    }
    emit_event(
        event_type="perf.slow_query",
        level="WARN",
        payload=payload,
    )


# ============================================================================
# TIMING CONTEXT MANAGERS
# ============================================================================


class TimedOperation:
    """Context manager for timing operations and emitting events."""

    def __init__(self, operation_name: str):
        """Initialize timer.

        Args:
            operation_name: Name of operation being timed
        """
        self.operation_name = operation_name
        self.start_time = 0.0

    def __enter__(self) -> "TimedOperation":
        """Enter context and start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record duration."""
        if exc_type is not None:
            return  # Caller will handle error emission

        duration_ms = (time.time() - self.start_time) * 1000
        emit_slow_operation(self.operation_name, duration_ms)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000


if __name__ == "__main__":
    # Simple test of emission functions
    print("✅ Observability instrumentation module loaded")

