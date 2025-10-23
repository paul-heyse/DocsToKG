# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.gc",
#   "purpose": "Garbage collection and prune operations for DuckDB catalog",
#   "sections": [
#     {
#       "id": "orphaneditem",
#       "name": "OrphanedItem",
#       "anchor": "class-orphaneditem",
#       "kind": "class"
#     },
#     {
#       "id": "pruneresult",
#       "name": "PruneResult",
#       "anchor": "class-pruneresult",
#       "kind": "class"
#     },
#     {
#       "id": "vacuumresult",
#       "name": "VacuumResult",
#       "anchor": "class-vacuumresult",
#       "kind": "class"
#     },
#     {
#       "id": "identify-orphaned-artifacts",
#       "name": "identify_orphaned_artifacts",
#       "anchor": "function-identify-orphaned-artifacts",
#       "kind": "function"
#     },
#     {
#       "id": "identify-orphaned-files",
#       "name": "identify_orphaned_files",
#       "anchor": "function-identify-orphaned-files",
#       "kind": "function"
#     },
#     {
#       "id": "prune-by-retention-days",
#       "name": "prune_by_retention_days",
#       "anchor": "function-prune-by-retention-days",
#       "kind": "function"
#     },
#     {
#       "id": "prune-keep-latest-n",
#       "name": "prune_keep_latest_n",
#       "anchor": "function-prune-keep-latest-n",
#       "kind": "function"
#     },
#     {
#       "id": "vacuum-database",
#       "name": "vacuum_database",
#       "anchor": "function-vacuum-database",
#       "kind": "function"
#     },
#     {
#       "id": "garbage-collect",
#       "name": "garbage_collect",
#       "anchor": "function-garbage-collect",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Garbage collection and prune operations for DuckDB catalog.

Responsibilities:
- Identify orphaned files and invalid records
- Safe deletion with dry-run support
- Vacuum and defragmentation
- Retention policy enforcement
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:  # pragma: no cover
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError("duckdb is required for catalog GC. Ensure .venv is initialized.") from exc

from .observability_instrumentation import (
    emit_prune_begin,
    emit_prune_deleted,
    emit_prune_orphan_found,
)

logger = logging.getLogger(__name__)


# ============================================================================
# GC RESULT TYPES (TYP)
# ============================================================================


@dataclass(frozen=True)
class OrphanedItem:
    """An artifact or file with no active reference."""

    item_type: str  # 'artifact' | 'file'
    item_id: str
    fs_path: Optional[Path]
    size_bytes: int
    orphaned_at: datetime


@dataclass(frozen=True)
class PruneResult:
    """Result of a prune operation."""

    items_identified: int
    items_deleted: int
    bytes_freed: int
    dry_run: bool
    duration_ms: float


@dataclass(frozen=True)
class VacuumResult:
    """Result of a VACUUM operation."""

    bytes_before: int
    bytes_after: int
    bytes_freed: int
    tables_vacuumed: int
    duration_ms: float


# ============================================================================
# ORPHAN DETECTION (ORPHAN)
# ============================================================================


def identify_orphaned_artifacts(
    conn: duckdb.DuckDBPyConnection,
    older_than_days: int = 30,
) -> list[OrphanedItem]:
    """Identify artifacts that are old and unused.

    Args:
        conn: DuckDB reader connection
        older_than_days: Only consider artifacts older than this

    Returns:
        List of OrphanedItem records
    """
    cutoff_date = datetime.now() - timedelta(days=older_than_days)

    # Find artifacts with no extracted files
    result = conn.execute(
        """
        SELECT a.artifact_id, a.fs_relpath, a.size, a.downloaded_at
        FROM artifacts a
        LEFT JOIN extracted_files f ON a.artifact_id = f.artifact_id
        WHERE a.downloaded_at < ?
        AND f.file_id IS NULL
        ORDER BY a.downloaded_at DESC
        """,
        [cutoff_date],
    ).fetchall()

    orphans = []
    for artifact_id, fs_relpath, size, downloaded_at in result:
        fs_path = Path(fs_relpath) if fs_relpath else None
        orphans.append(
            OrphanedItem(
                item_type="artifact",
                item_id=artifact_id,
                fs_path=fs_path,
                size_bytes=size or 0,
                orphaned_at=datetime.now(),
            )
        )

    logger.info(f"Identified {len(orphans)} orphaned artifacts")
    return orphans


def identify_orphaned_files(
    conn: duckdb.DuckDBPyConnection,
    older_than_days: int = 30,
) -> list[OrphanedItem]:
    """Identify files with failed validations.

    Args:
        conn: DuckDB reader connection
        older_than_days: Only consider files extracted longer ago

    Returns:
        List of OrphanedItem records
    """
    cutoff_date = datetime.now() - timedelta(days=older_than_days)

    # Find files with only failed validations
    result = conn.execute(
        """
        SELECT f.file_id, f.relpath, f.size, f.extracted_at
        FROM extracted_files f
        LEFT JOIN validations v ON f.file_id = v.file_id
        WHERE f.extracted_at < ?
        AND (v.file_id IS NULL OR (
            SELECT COUNT(*) FROM validations
            WHERE file_id = f.file_id AND status != 'fail'
        ) = 0)
        ORDER BY f.extracted_at DESC
        """,
        [cutoff_date],
    ).fetchall()

    orphans = []
    for file_id, relpath, size, extracted_at in result:
        fs_path = Path(relpath) if relpath else None
        orphans.append(
            OrphanedItem(
                item_type="file",
                item_id=file_id,
                fs_path=fs_path,
                size_bytes=size or 0,
                orphaned_at=datetime.now(),
            )
        )

    logger.info(f"Identified {len(orphans)} orphaned files")
    return orphans


# ============================================================================
# PRUNE OPERATIONS (PRUNE)
# ============================================================================


def prune_by_retention_days(
    conn: duckdb.DuckDBPyConnection,
    keep_days: int = 90,
    dry_run: bool = True,
) -> PruneResult:
    """Prune versions older than retention policy.

    Args:
        conn: DuckDB writer connection
        keep_days: Keep versions newer than this
        dry_run: If True, don't actually delete

    Returns:
        PruneResult with counts and bytes freed
    """
    import time

    start_ms = time.time() * 1000

    cutoff_date = datetime.now() - timedelta(days=keep_days)

    # Identify old versions
    old_versions = conn.execute(
        "SELECT version_id FROM versions WHERE ts < ?",
        [cutoff_date],
    ).fetchall()

    old_version_ids = [v[0] for v in old_versions]

    if not old_version_ids:
        duration_ms = (time.time() * 1000) - start_ms
        return PruneResult(
            items_identified=0,
            items_deleted=0,
            bytes_freed=0,
            dry_run=dry_run,
            duration_ms=duration_ms,
        )

    # Calculate bytes to free
    bytes_to_free = 0
    for version_id in old_version_ids:
        result = conn.execute(
            "SELECT SUM(size) FROM artifacts WHERE version_id = ?",
            [version_id],
        ).fetchone()
        if result and result[0]:
            bytes_to_free += result[0]

    if dry_run:
        duration_ms = (time.time() * 1000) - start_ms
        logger.info(
            f"DRY RUN: Would prune {len(old_version_ids)} versions, freeing {bytes_to_free} bytes"
        )
        return PruneResult(
            items_identified=len(old_version_ids),
            items_deleted=0,
            bytes_freed=0,
            dry_run=True,
            duration_ms=duration_ms,
        )

    # Actually delete (cascade: latest_pointer → validations → extracted_files → artifacts → versions)
    try:
        conn.begin()

        for version_id in old_version_ids:
            # Delete latest pointer
            conn.execute(
                "DELETE FROM latest_pointer WHERE version_id = ?",
                [version_id],
            )

            # Delete validations
            conn.execute(
                """DELETE FROM validations WHERE file_id IN (
                   SELECT file_id FROM extracted_files WHERE artifact_id IN (
                     SELECT artifact_id FROM artifacts WHERE version_id = ?
                   )
                )""",
                [version_id],
            )

            # Delete extracted files
            conn.execute(
                """DELETE FROM extracted_files WHERE artifact_id IN (
                   SELECT artifact_id FROM artifacts WHERE version_id = ?
                )""",
                [version_id],
            )

            # Delete artifacts
            conn.execute(
                "DELETE FROM artifacts WHERE version_id = ?",
                [version_id],
            )

        # Delete version records
        conn.execute(
            "DELETE FROM versions WHERE ts < ?",
            [cutoff_date],
        )

        conn.commit()

        duration_ms = (time.time() * 1000) - start_ms
        logger.info(f"Pruned {len(old_version_ids)} versions, freed {bytes_to_free} bytes")
        return PruneResult(
            items_identified=len(old_version_ids),
            items_deleted=len(old_version_ids),
            bytes_freed=bytes_to_free,
            dry_run=False,
            duration_ms=duration_ms,
        )

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Prune failed: {exc}")
        raise


def prune_keep_latest_n(
    conn: duckdb.DuckDBPyConnection,
    keep_count: int = 5,
    service: Optional[str] = None,
    dry_run: bool = True,
) -> PruneResult:
    """Prune old versions, keeping only N latest.

    Args:
        conn: DuckDB writer connection
        keep_count: Number of latest versions to keep
        service: Filter by service, or None for all
        dry_run: If True, don't actually delete

    Returns:
        PruneResult
    """
    # Emit observability begin event
    emit_prune_begin(dry_run=dry_run)
    start_ms = time.time() * 1000

    # Find all versions (optionally filtered by service)
    if service:
        all_versions = conn.execute(
            "SELECT version_id FROM versions WHERE service = ? ORDER BY ts DESC",
            [service],
        ).fetchall()
    else:
        all_versions = conn.execute("SELECT version_id FROM versions ORDER BY ts DESC").fetchall()

    all_version_ids = [v[0] for v in all_versions]

    # Keep only the latest N
    delete_ids = all_version_ids[keep_count:]

    items_identified = len(delete_ids)

    # Calculate bytes to free
    bytes_to_free = 0
    for version_id in delete_ids:
        result = conn.execute(
            "SELECT SUM(size) FROM artifacts WHERE version_id = ?",
            [version_id],
        ).fetchone()
        if result and result[0]:
            bytes_to_free += result[0]

    if dry_run:
        duration_ms = (time.time() * 1000) - start_ms
        logger.info(
            f"DRY RUN: Would delete {items_identified} versions, freeing {bytes_to_free} bytes"
        )
        return PruneResult(
            items_identified=items_identified,
            items_deleted=0,
            bytes_freed=0,
            dry_run=True,
            duration_ms=duration_ms,
        )

    # Actually delete (FK cascade)
    try:
        conn.begin()

        for version_id in delete_ids:
            # Emit observability orphan_found event for each version being deleted
            emit_prune_orphan_found(item_type="version", item_id=version_id, size_bytes=0)

            # Delete latest pointer
            conn.execute(
                "DELETE FROM latest_pointer WHERE version_id = ?",
                [version_id],
            )

            # Delete validations
            conn.execute(
                """DELETE FROM validations WHERE file_id IN (
                   SELECT file_id FROM extracted_files WHERE artifact_id IN (
                     SELECT artifact_id FROM artifacts WHERE version_id = ?
                   )
                )""",
                [version_id],
            )

            # Delete extracted files
            conn.execute(
                """DELETE FROM extracted_files WHERE artifact_id IN (
                   SELECT artifact_id FROM artifacts WHERE version_id = ?
                )""",
                [version_id],
            )

            # Delete artifacts
            conn.execute(
                "DELETE FROM artifacts WHERE version_id = ?",
                [version_id],
            )

            # Delete version
            conn.execute(
                "DELETE FROM versions WHERE version_id = ?",
                [version_id],
            )

        conn.commit()

        duration_ms = (time.time() * 1000) - start_ms
        logger.info(f"Pruned {items_identified} versions, freed {bytes_to_free} bytes")

        # Emit observability deleted event
        emit_prune_deleted(
            deleted_count=items_identified,
            freed_bytes=bytes_to_free,
            duration_ms=duration_ms,
            dry_run=False,
        )

        return PruneResult(
            items_identified=items_identified,
            items_deleted=items_identified,
            bytes_freed=bytes_to_free,
            dry_run=False,
            duration_ms=duration_ms,
        )

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Prune failed: {exc}")
        raise


# ============================================================================
# VACUUM & DEFRAGMENTATION (VACUUM)
# ============================================================================


def vacuum_database(
    conn: duckdb.DuckDBPyConnection,
) -> VacuumResult:
    """Vacuum the database to reclaim space and defragment.

    Args:
        conn: DuckDB writer connection

    Returns:
        VacuumResult with space reclaimed
    """
    import time

    start_ms = time.time() * 1000

    # Get tables to vacuum
    tables_to_vacuum = [
        "schema_version",
        "versions",
        "artifacts",
        "extracted_files",
        "validations",
    ]

    try:
        conn.begin()

        for table in tables_to_vacuum:
            conn.execute(f"VACUUM {table}")

        conn.commit()

        duration_ms = (time.time() * 1000) - start_ms
        logger.info(f"Vacuumed {len(tables_to_vacuum)} tables in {duration_ms}ms")

        return VacuumResult(
            bytes_before=0,
            bytes_after=0,
            bytes_freed=0,
            tables_vacuumed=len(tables_to_vacuum),
            duration_ms=duration_ms,
        )

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Vacuum failed: {exc}")
        raise


def garbage_collect(
    conn: duckdb.DuckDBPyConnection,
    keep_latest_n: int = 5,
    older_than_days: int = 30,
    dry_run: bool = True,
) -> tuple[PruneResult, VacuumResult]:
    """Full garbage collection: prune + vacuum.

    Args:
        conn: DuckDB writer connection
        keep_latest_n: Retention policy (keep N latest)
        older_than_days: Orphan age threshold
        dry_run: If True, don't modify DB

    Returns:
        Tuple of (PruneResult, VacuumResult)
    """
    # Emit observability begin event
    emit_prune_begin(dry_run=dry_run)
    gc_start_time = time.time()

    # Prune old versions
    prune_result = prune_keep_latest_n(conn, keep_count=keep_latest_n, dry_run=dry_run)

    # Vacuum (only if not dry-run)
    vacuum_result = VacuumResult(
        bytes_before=0,
        bytes_after=0,
        bytes_freed=0,
        tables_vacuumed=0,
        duration_ms=0.0,
    )

    if not dry_run:
        vacuum_result = vacuum_database(conn)

    logger.info(
        f"GC complete: {prune_result.items_deleted} versions pruned, "
        f"{prune_result.bytes_freed} bytes freed"
    )

    # Emit observability deleted event with combined results
    gc_duration_ms = (time.time() - gc_start_time) * 1000
    emit_prune_deleted(
        deleted_count=prune_result.items_deleted,
        freed_bytes=prune_result.bytes_freed,
        duration_ms=gc_duration_ms,
        dry_run=dry_run,
    )

    return prune_result, vacuum_result
