# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.job_state",
#   "purpose": "Monotonic state machine for artifact jobs",
#   "sections": [
#     {
#       "id": "advance-state",
#       "name": "advance_state",
#       "anchor": "function-advance-state",
#       "kind": "function"
#     },
#     {
#       "id": "get-current-state",
#       "name": "get_current_state",
#       "anchor": "function-get-current-state",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Monotonic state machine for artifact jobs.

This module enforces the artifact job state machine (PLANNED → LEASED → ... → FINALIZED).
State transitions are strictly monotonic: only forward progress is allowed. Any attempt
to transition from a disallowed previous state raises an error, helping catch logic bugs.

State Machine Diagram:
  PLANNED
    └─(lease)→ LEASED
        ├─(head done)→ HEAD_DONE
        │   └─(resume ok)→ RESUME_OK
        │       └─(stream)→ STREAMING
        │           └─(finalize)→ FINALIZED
        │               ├─(index)→ INDEXED
        │               └─(dedupe)→ DEDUPED
        └─(error)→ FAILED

  (found duplicate) → SKIPPED_DUPLICATE

Key Features:
  - Enforced monotonic progression
  - Clear error messages on invalid transitions
  - Support for alternate paths (e.g., direct to STREAMING)
  - Multi-role state tracking

Example:
  ```python
  import sqlite3
  from DocsToKG.ContentDownload.job_state import advance_state

  conn = sqlite3.connect("manifest.sqlite3")

  # Advance from LEASED to HEAD_DONE
  advance_state(conn, job_id="abc123", to_state="HEAD_DONE", allowed_from=("LEASED",))

  # Try invalid transition → raises RuntimeError
  advance_state(conn, job_id="abc123", to_state="STREAMING", allowed_from=("PLANNED",))
  ```
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional


def advance_state(
    cx: sqlite3.Connection,
    *,
    job_id: str,
    to_state: str,
    allowed_from: tuple[str, ...],
) -> None:
    """Advance job state with strict monotonic enforcement.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection (must have artifact_jobs table)
    job_id : str
        Job ID to advance
    to_state : str
        Target state (must be in CHECK constraint list)
    allowed_from : tuple[str, ...]
        Previous states from which this transition is valid

    Raises
    ------
    RuntimeError
        If the job is not in one of the allowed_from states.
        Error message includes current state and allowed previous states.

    Notes
    -----
    This function enforces forward-only progression. To retry a job,
    move it to FAILED state and re-queue for PLANNED, rather than
    going backwards in state.

    Examples
    --------
    After leasing and completing HEAD:
    >>> advance_state(cx, job_id=job_id, to_state="HEAD_DONE", allowed_from=("LEASED",))

    Multiple allowed previous states:
    >>> advance_state(
    ...     cx, job_id=job_id, to_state="STREAMING",
    ...     allowed_from=("HEAD_DONE", "RESUME_OK")
    ... )
    """
    now = time.time()
    state_placeholders = ",".join("?" * len(allowed_from))

    cur = cx.execute(
        f"""UPDATE artifact_jobs SET state=?, updated_at=?
           WHERE job_id=? AND state IN ({state_placeholders})""",
        (to_state, now, job_id, *allowed_from),
    )

    if cur.rowcount != 1:
        # Get current state for error message
        row = cx.execute(
            "SELECT state FROM artifact_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        current_state = row[0] if row else "MISSING"
        raise RuntimeError(
            f"state_transition_denied job={job_id} wants={to_state} "
            f"from={allowed_from} have={current_state}"
        )


def get_current_state(
    cx: sqlite3.Connection,
    *,
    job_id: str,
) -> Optional[str]:
    """Get the current state of a job.

    Parameters
    ----------
    cx : sqlite3.Connection
        Database connection
    job_id : str
        Job ID to query

    Returns
    -------
    str or None
        Current state, or None if job not found
    """
    row = cx.execute(
        "SELECT state FROM artifact_jobs WHERE job_id=?",
        (job_id,),
    ).fetchone()
    return row[0] if row else None
