# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.job_planning",
#   "purpose": "Idempotent artifact job planning and creation",
#   "sections": [
#     {
#       "id": "plan-job-if-absent",
#       "name": "plan_job_if_absent",
#       "anchor": "function-plan-job-if-absent",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Idempotent artifact job planning and creation.

This module provides functions to plan new artifact download jobs in an idempotent manner.
When a job with the same (work_id, artifact_id, canonical_url) is planned, the function
returns the existing job_id instead of creating a duplicate.

Key Features:
  - Atomic job creation with idempotency constraints
  - Deterministic job IDs based on artifact identity
  - Support for replanning across multiple runs
  - Integration with state machine for tracking

Example:
  ```python
  import sqlite3
  from DocsToKG.ContentDownload.job_planning import plan_job_if_absent

  conn = sqlite3.connect("manifest.sqlite3")
  job_id = plan_job_if_absent(
      conn,
      work_id="work-123",
      artifact_id="arxiv-2024-001",
      canonical_url="https://api.arxiv.org/pdf/2024.00001v1"
  )
  # On retry, same parameters return same job_id (no duplicate created)
  ```
"""

from __future__ import annotations

import sqlite3
import time
import uuid

from DocsToKG.ContentDownload.idempotency import job_key


def plan_job_if_absent(
    cx: sqlite3.Connection,
    *,
    work_id: str,
    artifact_id: str,
    canonical_url: str,
) -> str:
    """Plan a new artifact job, or return existing if already planned.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection (must have artifact_jobs table)
    work_id : str
        OpenAlex work identifier
    artifact_id : str
        Internal artifact identifier
    canonical_url : str
        Canonicalized URL for the artifact

    Returns
    -------
    str
        Job ID (UUID) for the planned job

    Raises
    ------
    sqlite3.Error
        If database operation fails

    Notes
    -----
    Uses the idempotency key derived from (work_id, artifact_id, canonical_url).
    If a job with the same key already exists, returns its job_id without
    creating a duplicate. All fields are frozen at creation time.

    The returned job_id can be used immediately to advance state or claim work.
    """
    ik = job_key(work_id, artifact_id, canonical_url)
    now = time.time()
    new_job_id = str(uuid.uuid4())

    try:
        cx.execute(
            """INSERT INTO artifact_jobs(job_id, work_id, artifact_id, canonical_url,
                                       state, created_at, updated_at, idempotency_key)
               VALUES (?, ?, ?, ?, 'PLANNED', ?, ?, ?)""",
            (new_job_id, work_id, artifact_id, canonical_url, now, now, ik),
        )
        return new_job_id
    except sqlite3.IntegrityError:
        # Job already exists; return its ID
        row = cx.execute(
            "SELECT job_id FROM artifact_jobs WHERE idempotency_key=?",
            (ik,),
        ).fetchone()
        if row:
            return row[0]
        # Should not reach here if constraint is working correctly
        raise
