# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.prune",
#   "purpose": "Prune orphaned files via staging→v_fs_orphans view; keep DB↔FS synchronized",
#   "sections": [
#     {"id": "types", "name": "Staging & Result Types", "anchor": "TYP", "kind": "models"},
#     {"id": "staging", "name": "FS Staging Loader", "anchor": "STG", "kind": "api"},
#     {"id": "orphans", "name": "Orphan Detection", "anchor": "ORP", "kind": "api"},
#     {"id": "delete", "name": "Deletion & Cleanup", "anchor": "DEL", "kind": "api"}
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
from dataclasses import dataclass, field
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


def load_staging_from_fs(cfg: duckdb.DuckDBPyConnection, root: Path) -> int:
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

    # Truncate staging before reload
    cfg.execute("TRUNCATE staging_fs_listing")

    for base, _dirs, files in os.walk(root):
        for fn in files:
            p = Path(base) / fn
            try:
                rel = p.resolve().relative_to(root).as_posix()
                st = p.stat()
                cfg.execute(
                    "INSERT INTO staging_fs_listing(scope, relpath, size_bytes, mtime) "
                    "VALUES ('version', ?, ?, NULL)",
                    [rel, int(st.st_size)],
                )
                count += 1
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to stat {p}: {e}")
                continue

    return count


# ============================================================================
# ORPHAN DETECTION (ORP)
# ============================================================================


def list_orphans(cfg: duckdb.DuckDBPyConnection) -> list[tuple[str, int]]:
    """
    Query v_fs_orphans view to find files on disk not referenced by DB.

    Pre-requisites:
    - staging_fs_listing table populated (call load_staging_from_fs first)
    - v_fs_orphans view must exist (created by migration 0005_staging_prune.sql)

    Returns:
        List of (relpath, size_bytes) tuples for orphaned files
    """
    result = cfg.execute(
        "SELECT relpath, size_bytes FROM v_fs_orphans ORDER BY size_bytes DESC"
    ).fetchall()
    return result


def count_orphans(cfg: duckdb.DuckDBPyConnection) -> int:
    """Count total orphaned files (for quick checks)."""
    row = cfg.execute("SELECT COUNT(*) FROM v_fs_orphans").fetchone()
    return row[0] if row else 0


# ============================================================================
# DELETION & CLEANUP (DEL)
# ============================================================================


def delete_orphans(
    cfg: duckdb.DuckDBPyConnection,
    root: Path,
    relpaths: list[str],
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
    stats.orphan_count = len(relpaths)
    start_time = time.time()

    emit_prune_begin(dry_run=False)

    deleted = 0
    total_bytes = 0
    for i, rel in enumerate(relpaths):
        try:
            fpath = root / rel
            if fpath.exists():
                size = fpath.stat().st_size
                fpath.unlink(missing_ok=True)
                deleted += 1
                total_bytes += size
                emit_prune_orphan_found(path=rel, size_bytes=size)
        except Exception as e:
            msg = f"Failed to delete {rel}: {e}"
            logger.warning(msg)
            stats.errors.append(msg)
            continue

        # Emit intermediate progress every batch_size
        if (i + 1) % batch_size == 0:
            logger.info(f"Prune progress: {deleted}/{len(relpaths)} deleted")

    duration_ms = (time.time() - start_time) * 1000
    stats.deleted_count = deleted
    stats.total_bytes_freed = total_bytes
    emit_prune_deleted(count=deleted, total_bytes=total_bytes, duration_ms=duration_ms)

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
    stats = PruneStats()

    # Load FS into staging
    stats.staged_count = load_staging_from_fs(cfg, root)
    logger.info(f"Staged {stats.staged_count} files from {root}")

    # Detect orphans
    orphans = list_orphans(cfg)
    stats.orphan_count = len(orphans)

    if max_items and stats.orphan_count > max_items:
        orphans = orphans[:max_items]
        logger.info(f"Limiting deletion to first {max_items} orphans")

    if dry_run:
        logger.info(f"DRY-RUN: would delete {len(orphans)} orphaned files")
        return stats

    # Actually delete
    relpaths = [rel for rel, _size in orphans]
    delete_stats = delete_orphans(cfg, root, relpaths)
    stats.deleted_count = delete_stats.deleted_count
    stats.errors = delete_stats.errors

    return stats
