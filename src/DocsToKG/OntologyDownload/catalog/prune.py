# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.prune",
#   "purpose": "Prune orphaned files via staging\u2192v_fs_orphans view; keep DB\u2194FS synchronized",
#   "sections": [
#     {
#       "id": "prunestats",
#       "name": "PruneStats",
#       "anchor": "class-prunestats",
#       "kind": "class"
#     },
#     {
#       "id": "load-staging-from-fs",
#       "name": "load_staging_from_fs",
#       "anchor": "function-load-staging-from-fs",
#       "kind": "function"
#     },
#     {
#       "id": "list-orphans",
#       "name": "list_orphans",
#       "anchor": "function-list-orphans",
#       "kind": "function"
#     },
#     {
#       "id": "count-orphans",
#       "name": "count_orphans",
#       "anchor": "function-count-orphans",
#       "kind": "function"
#     },
#     {
#       "id": "delete-orphans",
#       "name": "delete_orphans",
#       "anchor": "function-delete-orphans",
#       "kind": "function"
#     },
#     {
#       "id": "prune-with-staging",
#       "name": "prune_with_staging",
#       "anchor": "function-prune-with-staging",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Prune module for orphaned ontology files.

Responsibilities:
- Load filesystem snapshot into staging table
- Query v_fs_orphans view to find unreferenced files
- Delete orphans in safe batches
- Emit events for observability
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:  # pragma: no cover
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError("duckdb is required for catalog prune. Ensure .venv is initialized.") from exc

from .observability_instrumentation import (
    emit_prune_begin,
    emit_prune_deleted,
    emit_prune_orphan_found,
)

logger = logging.getLogger(__name__)

PRUNE_SCOPE = "prune"


# ============================================================================
# STAGING & RESULT TYPES (TYP)
# ============================================================================


@dataclass
class PruneStats:
    """Statistics from a prune operation."""

    staged_count: int = 0
    """Number of files loaded into staging table."""

    orphan_count: int = 0
    """Number of orphans detected in v_fs_orphans."""

    deleted_count: int = 0
    """Number of files successfully deleted."""

    total_bytes_freed: int = 0
    """Total bytes freed by deletion."""

    errors: list[str] = field(default_factory=list)
    """List of deletion errors (if any)."""


# ============================================================================
# FILESYSTEM STAGING LOADER (STG)
# ============================================================================


def load_staging_from_fs(
    cfg: duckdb.DuckDBPyConnection,
    root: Path,
    *,
    scope: str = PRUNE_SCOPE,
) -> int:
    """
    Walk filesystem under <root> and populate staging_fs_listing table.

    Pre-requisites:
    - Table staging_fs_listing must exist (created by migration 0005_staging_prune.sql)
      Schema: (scope TEXT, relpath TEXT, size_bytes BIGINT, mtime TIMESTAMP)

    Args:
        cfg: DuckDB connection (writer mode)
        root: Filesystem root to scan

    Returns:
        Count of files inserted into staging_fs_listing
    """
    count = 0
    root = root.resolve()

    # Remove previous staging rows for this scope
    cfg.execute("DELETE FROM staging_fs_listing WHERE scope = ?", [scope])

    for base, _dirs, files in os.walk(root):
        for fn in files:
            p = Path(base) / fn
            try:
                rel = p.resolve().relative_to(root).as_posix()
                st = p.stat()
                cfg.execute(
                    "INSERT INTO staging_fs_listing(scope, relpath, size_bytes, mtime) "
                    "VALUES (?, ?, ?, ?)",
                    [scope, rel, int(st.st_size), datetime.fromtimestamp(st.st_mtime)],
                )
                count += 1
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to stat {p}: {e}")
                continue

    return count


# ============================================================================
# ORPHAN DETECTION (ORP)
# ============================================================================


def list_orphans(
    cfg: duckdb.DuckDBPyConnection,
    *,
    scope: str = PRUNE_SCOPE,
) -> list[tuple[str, int]]:
    """
    Query v_fs_orphans view to find files on disk not referenced by DB.

    Pre-requisites:
    - staging_fs_listing table populated (call load_staging_from_fs first)
    - v_fs_orphans view must exist (created by migration 0005_staging_prune.sql)

    Returns:
        List of (relpath, size_bytes) tuples for orphaned files
    """
    result = cfg.execute(
        """
        SELECT relpath, size_bytes
        FROM staging_fs_listing
        WHERE scope = ?
          AND relpath NOT IN (
            SELECT fs_relpath FROM artifacts
            UNION ALL
            SELECT v.service || '/' || f.version_id || '/' || f.relpath_in_version
            FROM extracted_files AS f
            JOIN versions AS v ON v.version_id = f.version_id
        )
        ORDER BY size_bytes DESC
        """,
        [scope],
    ).fetchall()
    return result


def count_orphans(cfg: duckdb.DuckDBPyConnection, *, scope: str = PRUNE_SCOPE) -> int:
    """Count total orphaned files (for quick checks)."""
    row = cfg.execute(
        "SELECT COUNT(*) FROM staging_fs_listing WHERE scope = ? AND relpath NOT IN ("
        "SELECT fs_relpath FROM artifacts UNION ALL SELECT v.service || '/' || f.version_id || '/' || f.relpath_in_version"
        " FROM extracted_files AS f JOIN versions AS v ON v.version_id = f.version_id)",
        [scope],
    ).fetchone()
    return row[0] if row else 0


# ============================================================================
# DELETION & CLEANUP (DEL)
# ============================================================================


def delete_orphans(
    cfg: duckdb.DuckDBPyConnection,
    root: Path,
    entries: list[tuple[str, int]],
    batch_size: int = 100,
) -> PruneStats:
    """
    Delete orphaned files from filesystem in safe batches.

    Design:
    - DB is the source of truth; FS rows are never mutated in this function
    - Deletion happens outside of transactions to avoid hold-locks
    - Errors are logged but don't halt the prune operation
    - Returns summary stats for observability

    Args:
        cfg: DuckDB connection (used for boundary emissions only)
        root: Filesystem root (for constructing absolute paths)
        relpaths: List of relative paths to delete
        batch_size: Max files to delete before yielding (default 100)

    Returns:
        PruneStats with counts and error list
    """
    import time

    root = root.resolve()
    stats = PruneStats()
    stats.orphan_count = len(entries)
    start_time = time.time()

    deleted = 0
    total_bytes = 0
    for i, (rel, size_hint) in enumerate(entries):
        try:
            fpath = root / rel
            if fpath.exists():
                stat_size = fpath.stat().st_size
                fpath.unlink(missing_ok=True)
                deleted += 1
                total_bytes += stat_size
            else:
                logger.debug("Path already missing during prune: %s", fpath)
        except Exception as e:
            msg = f"Failed to delete {rel}: {e}"
            logger.warning(msg)
            stats.errors.append(msg)
            continue

        # Emit intermediate progress every batch_size
        if (i + 1) % batch_size == 0:
            logger.info(f"Prune progress: {deleted}/{len(entries)} deleted")

    duration_ms = (time.time() - start_time) * 1000
    stats.deleted_count = deleted
    stats.total_bytes_freed = total_bytes
    emit_prune_deleted(
        deleted_count=deleted,
        freed_bytes=total_bytes,
        duration_ms=duration_ms,
        dry_run=False,
    )

    return stats


def prune_with_staging(
    cfg: duckdb.DuckDBPyConnection,
    root: Path,
    max_items: int | None = None,
    dry_run: bool = True,
) -> PruneStats:
    """
    High-level prune flow: stage → detect orphans → optionally delete.

    This is the main entry point for prune operations (used by CLI).

    Args:
        cfg: DuckDB connection
        root: Filesystem root
        max_items: Limit orphan deletion to N items (None = all)
        dry_run: If True, only report; if False, actually delete

    Returns:
        PruneStats with full operation summary
    """
    emit_prune_begin(dry_run=dry_run)
    operation_start = time.time()

    stats = PruneStats()

    # Load FS into staging
    stats.staged_count = load_staging_from_fs(cfg, root, scope=PRUNE_SCOPE)
    logger.info(f"Staged {stats.staged_count} files from {root}")

    # Detect orphans
    orphans = list_orphans(cfg, scope=PRUNE_SCOPE)
    stats.orphan_count = len(orphans)

    if max_items and stats.orphan_count > max_items:
        orphans = orphans[:max_items]
        logger.info(f"Limiting deletion to first {max_items} orphans")

    # Emit orphan events early for observability
    for relpath, size in orphans:
        emit_prune_orphan_found(path=relpath, size_bytes=size)

    potential_bytes = sum(size for _, size in orphans)

    if dry_run:
        logger.info(f"DRY-RUN: would delete {len(orphans)} orphaned files")
        stats.total_bytes_freed = potential_bytes
        emit_prune_deleted(
            deleted_count=0,
            freed_bytes=potential_bytes,
            duration_ms=(time.time() - operation_start) * 1000,
            dry_run=True,
        )
        return stats

    # Actually delete
    delete_stats = delete_orphans(cfg, root, orphans)
    stats.deleted_count = delete_stats.deleted_count
    stats.total_bytes_freed = delete_stats.total_bytes_freed
    stats.errors = delete_stats.errors

    return stats
