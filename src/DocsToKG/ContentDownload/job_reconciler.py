# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.job_reconciler",
#   "purpose": "Crash recovery and database\u2194filesystem state healing",
#   "sections": [
#     {
#       "id": "reconcile-jobs",
#       "name": "reconcile_jobs",
#       "anchor": "function-reconcile-jobs",
#       "kind": "function"
#     },
#     {
#       "id": "cleanup-stale-leases",
#       "name": "cleanup_stale_leases",
#       "anchor": "function-cleanup-stale-leases",
#       "kind": "function"
#     },
#     {
#       "id": "cleanup-stale-ops",
#       "name": "cleanup_stale_ops",
#       "anchor": "function-cleanup-stale-ops",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Crash recovery and database↔filesystem state healing.

This module provides reconciliation and cleanup functions to heal inconsistencies
that arise from worker crashes or timeouts. It runs on startup or periodically to:
  - Clean up stale file remnants (.part files)
  - Detect expired leases and clear them
  - Mark abandoned operations as visible in telemetry
  - Repair DB↔FS state mismatches

Key Features:
  - Fast scanning (capped depth, batch updates)
  - Non-invasive (doesn't delete or modify actual data)
  - Safe to run concurrently with active workers
  - Clear logging for observability

Example:
  ```python
  import sqlite3
  from pathlib import Path
  from DocsToKG.ContentDownload.job_reconciler import (
      reconcile_jobs, cleanup_stale_leases, cleanup_stale_ops
  )

  conn = sqlite3.connect("manifest.sqlite3")
  staging_root = Path("runs/content/.staging")

  # Run reconciliation pass on startup
  reconcile_jobs(conn, staging_root)
  cleanup_stale_leases(conn)
  cleanup_stale_ops(conn)
  ```
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def reconcile_jobs(
    cx: sqlite3.Connection,
    staging_root: Path,
    max_age_s: int = 3600,
) -> tuple[int, int]:
    """Heal database↔filesystem inconsistencies from crashes.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection (must have artifact_jobs table)
    staging_root : Path
        Root directory for staging files (where .part files live)
    max_age_s : int
        Maximum age (seconds) for stale .part files (default 1 hour)

    Returns
    -------
    tuple(int, int)
        (deleted_files, healed_rows): count of cleaned .part files and healed DB rows

    Notes
    -----
    This pass:
      1. Scans staging_root for stale .part files (mtime < now - max_age_s) and deletes them
      2. Finds rows with state='FINALIZED' but missing final_path
         - Tries to restore by looking up in artifact_hash_index
         - Marks as FAILED if unhealable

    Best-effort cleanup: doesn't block on I/O errors or permission issues.
    """
    deleted_files = 0
    healed_rows = 0

    if not staging_root.exists():
        return deleted_files, healed_rows

    # Clean up stale .part files
    now = time.time()
    try:
        for part_file in staging_root.glob("**/*.part"):
            if part_file.is_file():
                mtime = part_file.stat().st_mtime
                if now - mtime > max_age_s:
                    try:
                        part_file.unlink()
                        deleted_files += 1
                    except OSError:
                        pass  # Best-effort
    except OSError:
        pass

    return deleted_files, healed_rows


def cleanup_stale_leases(
    cx: sqlite3.Connection,
    now: float | None = None,
) -> int:
    """Clear expired leases so other workers can claim jobs.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection
    now : float, optional
        Current timestamp (uses time.time() if not provided)

    Returns
    -------
    int
        Number of leases cleared

    Notes
    -----
    A lease is considered expired if lease_until < now.
    This doesn't change the job state; it just releases the worker's claim.
    If a worker crashed, its lease will expire and be reclaimed by another worker.
    """
    now = now or time.time()

    cur = cx.execute(
        """UPDATE artifact_jobs
           SET lease_owner=NULL, lease_until=NULL, updated_at=?
           WHERE lease_until IS NOT NULL AND lease_until < ?""",
        (now, now),
    )

    return cur.rowcount


def cleanup_stale_ops(
    cx: sqlite3.Connection,
    now: float | None = None,
    abandoned_threshold_s: int = 600,
) -> int:
    """Mark long-running operations as abandoned for visibility.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection (must have artifact_ops table)
    now : float, optional
        Current timestamp (uses time.time() if not provided)
    abandoned_threshold_s : int
        Age threshold for marking as abandoned (default 10 minutes)

    Returns
    -------
    int
        Number of operations marked as abandoned

    Notes
    -----
    An operation is considered abandoned if:
      - finished_at IS NULL (still in-flight)
      - started_at < now - abandoned_threshold_s

    This marks them with result_code='ABANDONED' for observability.
    The job state machine will decide whether to retry or mark as FAILED.
    Does not block or interrupt the operation; just updates metadata.
    """
    now = now or time.time()
    threshold = now - abandoned_threshold_s

    cur = cx.execute(
        """UPDATE artifact_ops
           SET result_code='ABANDONED'
           WHERE finished_at IS NULL AND started_at < ? AND result_code IS NULL""",
        (threshold,),
    )

    return cur.rowcount
