# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.instrumentation",
#   "purpose": "Database catalog instrumentation and telemetry.",
#   "sections": [
#     {
#       "id": "emit-tx-commit-event",
#       "name": "emit_tx_commit_event",
#       "anchor": "function-emit-tx-commit-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-tx-rollback-event",
#       "name": "emit_tx_rollback_event",
#       "anchor": "function-emit-tx-rollback-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-migration-applied-event",
#       "name": "emit_migration_applied_event",
#       "anchor": "function-emit-migration-applied-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-boundary-check-event",
#       "name": "emit_boundary_check_event",
#       "anchor": "function-emit-boundary-check-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-latest-mismatch-event",
#       "name": "emit_latest_mismatch_event",
#       "anchor": "function-emit-latest-mismatch-event",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Database catalog instrumentation and telemetry.

Emits db.tx.*, db.migrate.*, and db.boundary.* events for observability
into database operations and transactional boundaries.
"""


from DocsToKG.OntologyDownload.observability.events import emit_event


def emit_tx_commit_event(
    tables_affected: list[str],
    rows_changed: int,
    duration_ms: float,
) -> None:
    """Emit event when database transaction commits.

    Args:
        tables_affected: List of table names modified
        rows_changed: Total rows inserted/updated/deleted
        duration_ms: Transaction duration in milliseconds
    """
    try:
        emit_event(
            type="db.tx.commit",
            level="INFO",
            payload={
                "tables_affected": tables_affected,
                "rows_changed": rows_changed,
                "duration_ms": duration_ms,
                "table_count": len(tables_affected),
            },
        )
    except Exception:
        pass


def emit_tx_rollback_event(
    reason: str,
    tables_involved: list[str] | None = None,
    duration_ms: float = 0,
) -> None:
    """Emit event when database transaction rolls back.

    Args:
        reason: Reason for rollback
        tables_involved: List of table names involved
        duration_ms: Transaction duration before rollback
    """
    try:
        payload = {
            "reason": reason,
            "duration_ms": duration_ms,
        }
        if tables_involved:
            payload["tables_involved"] = tables_involved
            payload["table_count"] = len(tables_involved)

        emit_event(
            type="db.tx.rollback",
            level="WARN",
            payload=payload,
        )
    except Exception:
        pass


def emit_migration_applied_event(
    migration_id: str,
    description: str,
    duration_ms: float,
) -> None:
    """Emit event when database migration is applied.

    Args:
        migration_id: Migration identifier (e.g., version number)
        description: Migration description
        duration_ms: Migration duration in milliseconds
    """
    try:
        emit_event(
            type="db.migrate.applied",
            level="INFO",
            payload={
                "migration_id": migration_id,
                "description": description,
                "duration_ms": duration_ms,
            },
        )
    except Exception:
        pass


def emit_boundary_check_event(
    check_type: str,
    passed: bool,
    details: dict | None = None,
) -> None:
    """Emit event for database boundary validation.

    Args:
        check_type: Type of check (e.g., 'latest_pointer_consistency')
        passed: Whether check passed
        details: Optional details about the check
    """
    try:
        payload = {
            "check_type": check_type,
            "passed": passed,
        }
        if details:
            payload.update(details)

        emit_event(
            type="db.boundary.check",
            level="WARN" if not passed else "INFO",
            payload=payload,
        )
    except Exception:
        pass


def emit_latest_mismatch_event(
    expected: str,
    actual: str,
) -> None:
    """Emit event when latest pointer/marker mismatches.

    Args:
        expected: Expected latest pointer value
        actual: Actual latest pointer value
    """
    try:
        emit_event(
            type="db.latest.mismatch",
            level="ERROR",
            payload={
                "expected": expected[:50],
                "actual": actual[:50],
            },
        )
    except Exception:
        pass


__all__ = [
    "emit_tx_commit_event",
    "emit_tx_rollback_event",
    "emit_migration_applied_event",
    "emit_boundary_check_event",
    "emit_latest_mismatch_event",
]
