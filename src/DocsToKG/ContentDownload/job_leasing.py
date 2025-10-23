# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.job_leasing",
#   "purpose": "Concurrency-safe job leasing for multi-worker coordination",
#   "sections": [
#     {
#       "id": "lease-next-job",
#       "name": "lease_next_job",
#       "anchor": "function-lease-next-job",
#       "kind": "function"
#     },
#     {
#       "id": "renew-lease",
#       "name": "renew_lease",
#       "anchor": "function-renew-lease",
#       "kind": "function"
#     },
#     {
#       "id": "release-lease",
#       "name": "release_lease",
#       "anchor": "function-release-lease",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Concurrency-safe job leasing for multi-worker coordination.

This module provides a SQLite-friendly leasing mechanism to ensure only one worker
processes a given artifact job at a time. Leases have a time-to-live (TTL) and can
be renewed or released by the owning worker.

Key Features:
  - Atomic lease acquisition for one available job
  - Configurable lease TTL with renewal support
  - Automatic cleanup of expired leases
  - Safe for multi-process environments

Example:
  ```python
  import sqlite3
  from DocsToKG.ContentDownload.job_leasing import lease_next_job, renew_lease, release_lease

  conn = sqlite3.connect("manifest.sqlite3")
  owner_id = "worker-123"

  # Claim a job
  job = lease_next_job(conn, owner=owner_id, ttl_s=120)
  if job:
      try:
          # Do work...
          renew_lease(conn, job_id=job["job_id"], owner=owner_id, ttl_s=120)
          # More work...
      finally:
          release_lease(conn, job_id=job["job_id"], owner=owner_id)
  ```
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from typing import Any


def _row_to_dict(row: Any, description: Sequence[tuple[Any, ...]] | None) -> dict[str, Any]:
    """Convert sqlite3 rows (Row or tuple) into dictionaries."""

    if isinstance(row, sqlite3.Row):
        return dict(row)
    if description:
        columns = [col[0] for col in description]
        return {column: value for column, value in zip(columns, row, strict=False)}
    return {str(index): value for index, value in enumerate(row)}


def lease_next_job(
    cx: sqlite3.Connection,
    *,
    owner: str,
    ttl_s: int = 120,
    from_states: tuple[str, ...] = ("PLANNED", "FAILED"),
) -> dict[str, Any] | None:
    """Atomically claim the next available job for this worker.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection (must have artifact_jobs table)
    owner : str
        Worker/process identifier (e.g., hostname, PID, UUID)
    ttl_s : int
        Lease time-to-live in seconds (default 120)
    from_states : tuple[str, ...]
        States considered available for leasing (default: PLANNED, FAILED)

    Returns
    -------
    dict or None
        Leased job row as dict, or None if no jobs available

    Notes
    -----
    This operation is atomic: exactly one worker wins the lease for a given job.
    Jobs with expired leases (lease_until < now) are considered available for re-claiming.
    The state is advanced to 'LEASED' atomically with the lease claim.
    """
    now = time.time()
    state_placeholders = ",".join("?" * len(from_states))

    # First, select the job to lease
    select_sql = f"""
    SELECT job_id FROM artifact_jobs
    WHERE state IN ({state_placeholders})
      AND (lease_until IS NULL OR lease_until < ?)
    ORDER BY created_at
    LIMIT 1
    """
    row = cx.execute(select_sql, (*from_states, now)).fetchone()
    if not row:
        return None

    target_job_id = row[0]

    # Now try to update it
    update_sql = """
    UPDATE artifact_jobs
    SET lease_owner=?, lease_until=?, state='LEASED', updated_at=?
    WHERE job_id=? AND (lease_until IS NULL OR lease_until < ?)
    """
    cur = cx.execute(update_sql, (owner, now + ttl_s, now, target_job_id, now))

    if cur.rowcount == 1:
        # Successfully leased; fetch and return
        result_cursor = cx.execute(
            "SELECT * FROM artifact_jobs WHERE job_id=?",
            (target_job_id,),
        )
        result_row = result_cursor.fetchone()
        return _row_to_dict(result_row, result_cursor.description) if result_row else None

    return None


def renew_lease(
    cx: sqlite3.Connection,
    *,
    job_id: str,
    owner: str,
    ttl_s: int = 120,
) -> bool:
    """Renew the lease for a job owned by this worker.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection
    job_id : str
        Job ID to renew
    owner : str
        Current lease owner (must match for renewal to succeed)
    ttl_s : int
        New lease time-to-live in seconds (default 120)

    Returns
    -------
    bool
        True if renewal succeeded, False if job not owned by this worker

    Notes
    -----
    Only the current lease owner can renew the lease. This prevents
    workers from interfering with each other's work.
    """
    now = time.time()
    cur = cx.execute(
        """UPDATE artifact_jobs SET lease_until=?, updated_at=?
           WHERE job_id=? AND lease_owner=?""",
        (now + ttl_s, now, job_id, owner),
    )
    return cur.rowcount == 1


def release_lease(
    cx: sqlite3.Connection,
    *,
    job_id: str,
    owner: str,
) -> bool:
    """Release the lease for a job, allowing other workers to claim it.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection
    job_id : str
        Job ID to release
    owner : str
        Current lease owner (must match for release to succeed)

    Returns
    -------
    bool
        True if release succeeded, False if job not owned by this worker

    Notes
    -----
    This is best-effort cleanup. If a worker crashes before releasing,
    the lease will expire automatically after TTL seconds. For permanent
    completion, update the job state to a terminal state (FINALIZED, etc.)
    instead of just releasing.
    """
    cur = cx.execute(
        """UPDATE artifact_jobs SET lease_until=NULL, lease_owner=NULL, updated_at=?
           WHERE job_id=? AND lease_owner=?""",
        (time.time(), job_id, owner),
    )
    return cur.rowcount == 1
