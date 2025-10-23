# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.idempotency",
#   "purpose": "Idempotency keys and exactly-once effect wrappers.",
#   "sections": [
#     {
#       "id": "ikey",
#       "name": "ikey",
#       "anchor": "function-ikey",
#       "kind": "function"
#     },
#     {
#       "id": "job-key",
#       "name": "job_key",
#       "anchor": "function-job-key",
#       "kind": "function"
#     },
#     {
#       "id": "op-key",
#       "name": "op_key",
#       "anchor": "function-op-key",
#       "kind": "function"
#     },
#     {
#       "id": "acquire-lease",
#       "name": "acquire_lease",
#       "anchor": "function-acquire-lease",
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
#     },
#     {
#       "id": "advance-state",
#       "name": "advance_state",
#       "anchor": "function-advance-state",
#       "kind": "function"
#     },
#     {
#       "id": "run-effect",
#       "name": "run_effect",
#       "anchor": "function-run-effect",
#       "kind": "function"
#     },
#     {
#       "id": "reconcile-stale-leases",
#       "name": "reconcile_stale_leases",
#       "anchor": "function-reconcile-stale-leases",
#       "kind": "function"
#     },
#     {
#       "id": "reconcile-abandoned-ops",
#       "name": "reconcile_abandoned_ops",
#       "anchor": "function-reconcile-abandoned-ops",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Idempotency keys and exactly-once effect wrappers.

This module provides deterministic key generation for both job-level and
operation-level idempotency, enabling crash recovery and multi-worker safety.

Idempotency Model:
  - Job keys: Prevent duplicate artifact jobs (UNIQUE constraint)
  - Operation keys: Prevent duplicate effects (INSERT OR IGNORE pattern)
  - State machine: Enforce monotonic state transitions
  - Leases: Ensure single worker processes job at a time

Key composition:
  - Job key: SHA-256({work_id, artifact_id, canonical_url, role, headers})
  - Op key: SHA-256({kind, job_id, url, range_start, ...context})

Database schema (see streaming_integration.py):
  - artifact_jobs: State machine + leasing
  - artifact_ops: Operation ledger with results
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Key Generators (Deterministic SHA-256)
# ============================================================================


def ikey(obj: Dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 key from a dictionary.

    Serializes with sorted keys and no whitespace to ensure determinism.
    Multiple Python processes computing the same object will produce the same key.

    Args:
        obj: Dictionary to hash (may contain nested dicts/lists)

    Returns:
        Lowercase hex-encoded SHA-256 hash
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def job_key(work_id: str, artifact_id: str, canonical_url: str) -> str:
    """Generate an idempotency key for an artifact job.

    This key ensures that the same (work, artifact, URL) combination always
    maps to the same database row, preventing duplicate job creation.

    Args:
        work_id: Unique work/task identifier
        artifact_id: Unique artifact identifier
        canonical_url: Canonicalized URL (from canonical_for_index)

    Returns:
        Idempotency key suitable for INSERT OR IGNORE
    """
    obj = {
        "v": 1,
        "kind": "JOB",
        "work_id": work_id,
        "artifact_id": artifact_id,
        "url": canonical_url,
        "role": "artifact",  # Always "artifact" for this scope
    }
    return ikey(obj)


def op_key(kind: str, job_id: str, **context) -> str:
    """Generate an idempotency key for an operation (side effect).

    Each kind of operation (HEAD, STREAM, FINALIZE, INDEX, DEDUPE) gets its own key.
    Additional context (url, range_start, sha256, etc.) is included to distinguish
    between different invocations of the same operation kind.

    Args:
        kind: Operation type ("HEAD", "STREAM", "FINALIZE", "INDEX", "DEDUPE")
        job_id: Job ID this operation belongs to
        **context: Additional context (url, range_start, sha256, part_path, target_path, etc.)

    Returns:
        Idempotency key suitable for INSERT OR IGNORE
    """
    obj = {
        "v": 1,
        "kind": kind,
        "job_id": job_id,
    }
    obj.update(context)
    return ikey(obj)


# ============================================================================
# Lease Helpers
# ============================================================================


def acquire_lease(
    conn: Any,
    owner: str,
    ttl_seconds: float,
    now_fn: Callable[[], float] = None,
) -> Optional[str]:
    """Acquire a lease on the next available job.

    Atomically updates the first PLANNED or FAILED job with lease info.
    Only succeeds if no other worker holds the lease (lease_until < now).

    Args:
        conn: SQLite database connection
        owner: Worker identifier (hostname:pid or UUID)
        ttl_seconds: Lease duration in seconds
        now_fn: Function returning current time (default: time.time)

    Returns:
        job_id if lease acquired, None if no available jobs

    Raises:
        sqlite3.Error: On database errors
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()
    lease_until = now + ttl_seconds

    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE artifact_jobs
        SET lease_owner = ?, lease_until = ?, state = 'LEASED', updated_at = ?
        WHERE job_id = (
            SELECT job_id FROM artifact_jobs
            WHERE state IN ('PLANNED', 'FAILED')
              AND (lease_until IS NULL OR lease_until < ?)
            ORDER BY created_at ASC
            LIMIT 1
        )
        AND (lease_until IS NULL OR lease_until < ?)
        RETURNING job_id
        """,
        (owner, lease_until, now, now, now),
    )

    row = cursor.fetchone()
    return row[0] if row else None


def renew_lease(
    conn: Any,
    job_id: str,
    owner: str,
    ttl_seconds: float,
    now_fn: Callable[[], float] = None,
) -> bool:
    """Renew an active lease (prevents expiry during long operations).

    Args:
        conn: SQLite database connection
        job_id: Job to renew
        owner: Current lease owner
        ttl_seconds: New lease duration in seconds
        now_fn: Function returning current time

    Returns:
        True if lease was renewed, False if not owned or not leased

    Raises:
        sqlite3.Error: On database errors
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()
    lease_until = now + ttl_seconds

    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE artifact_jobs
        SET lease_until = ?, updated_at = ?
        WHERE job_id = ? AND lease_owner = ? AND state = 'LEASED'
        """,
        (lease_until, now, job_id, owner),
    )

    return cursor.rowcount > 0


def release_lease(
    conn: Any,
    job_id: str,
    owner: str,
    now_fn: Callable[[], float] = None,
) -> bool:
    """Release an active lease (best-effort cleanup on success).

    Args:
        conn: SQLite database connection
        job_id: Job to release
        owner: Current lease owner
        now_fn: Function returning current time

    Returns:
        True if lease was released, False if not owned

    Raises:
        sqlite3.Error: On database errors
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()

    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE artifact_jobs
        SET lease_until = NULL, lease_owner = NULL, updated_at = ?
        WHERE job_id = ? AND lease_owner = ?
        """,
        (now, job_id, owner),
    )

    return cursor.rowcount > 0


# ============================================================================
# State Machine Enforcement
# ============================================================================


def advance_state(
    conn: Any,
    job_id: str,
    to_state: str,
    allowed_from: set,
    now_fn: Callable[[], float] = None,
) -> bool:
    """Atomically advance job state with validation.

    Only allows forward transitions from the allowed set of previous states.
    Raises on invalid transitions.

    Args:
        conn: SQLite database connection
        job_id: Job to update
        to_state: Target state
        allowed_from: Set of allowed previous states
        now_fn: Function returning current time

    Returns:
        True if state was advanced, False if transition invalid

    Raises:
        sqlite3.Error: On database errors
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()

    # Build WHERE clause for allowed states
    placeholders = ",".join("?" * len(allowed_from))
    allowed_list = list(allowed_from)

    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE artifact_jobs
        SET state = ?, updated_at = ?
        WHERE job_id = ? AND state IN ({placeholders})
        """,
        [to_state, now, job_id] + allowed_list,
    )

    return cursor.rowcount > 0


# ============================================================================
# Exactly-Once Effect Wrapper
# ============================================================================


def run_effect(
    conn: Any,
    op_key: str,
    job_id: str,
    kind: str,
    fn: Callable[[], Dict[str, Any]],
    now_fn: Callable[[], float] = None,
) -> Dict[str, Any]:
    """Execute a side effect exactly once.

    Algorithm:
      1. Try to INSERT op row (if key exists, returns early with cached result)
      2. Execute fn() to perform the side effect
      3. UPDATE op row with result
      4. Return result

    Args:
        conn: SQLite database connection
        op_key: Unique operation key
        job_id: Associated job ID
        kind: Operation type
        fn: Callable that performs side effect, returns {code, ...result}
        now_fn: Function returning current time

    Returns:
        Result dictionary from fn() or cached result if already executed

    Raises:
        sqlite3.Error: On database errors (not caught)
        Exception: From fn() if it fails (caught and logged)
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()

    cursor = conn.cursor()

    # Try to acquire the operation row
    try:
        cursor.execute(
            """
            INSERT INTO artifact_ops(op_key, job_id, op_type, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (op_key, job_id, kind, now),
        )
        conn.commit()
    except Exception as e:
        # Key already exists: return cached result
        if "UNIQUE constraint failed" in str(e) or "UNIQUE" in str(e):
            cursor.execute("SELECT result_json FROM artifact_ops WHERE op_key = ?", (op_key,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except Exception as parse_err:
                    LOGGER.warning(f"cached_result_parse_error: {parse_err}")
                    return {}
            return {}
        raise

    # Execute the side effect
    try:
        result = fn()
    except Exception as e:
        LOGGER.exception(f"effect_failed: {kind} {job_id}")
        result = {
            "code": "ERROR",
            "error": str(e),
            "error_type": type(e).__name__,
        }

    # Store result
    result_json = json.dumps(result)[:20000]  # Cap at 20KB
    code = result.get("code", "OK")

    cursor.execute(
        """
        UPDATE artifact_ops
        SET finished_at = ?, result_code = ?, result_json = ?
        WHERE op_key = ?
        """,
        (now_fn(), code, result_json, op_key),
    )
    conn.commit()

    return result


# ============================================================================
# Database Reconciler
# ============================================================================


def reconcile_stale_leases(
    conn: Any,
    now_fn: Callable[[], float] = None,
) -> int:
    """Clear stale leases (where lease_until < now).

    Called at startup to recover from crashed workers.

    Args:
        conn: SQLite database connection
        now_fn: Function returning current time

    Returns:
        Number of leases cleared
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()

    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE artifact_jobs
        SET lease_owner = NULL, lease_until = NULL
        WHERE lease_until < ? AND state = 'LEASED'
        """,
        (now,),
    )

    count = cursor.rowcount
    conn.commit()

    if count > 0:
        LOGGER.info(f"reconcile_stale_leases: cleared {count} leases")

    return count


def reconcile_abandoned_ops(
    conn: Any,
    abandoned_threshold_seconds: float = 600,
    now_fn: Callable[[], float] = None,
) -> int:
    """Mark operations as ABANDONED if in-flight for too long.

    Called periodically to detect crashed operations and allow retry.

    Args:
        conn: SQLite database connection
        abandoned_threshold_seconds: Time threshold (default 10 minutes)
        now_fn: Function returning current time

    Returns:
        Number of ops marked ABANDONED
    """
    import time

    if now_fn is None:
        now_fn = time.time

    now = now_fn()
    cutoff = now - abandoned_threshold_seconds

    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE artifact_ops
        SET result_code = 'ABANDONED'
        WHERE finished_at IS NULL AND started_at < ?
        """,
        (cutoff,),
    )

    count = cursor.rowcount
    conn.commit()

    if count > 0:
        LOGGER.info(f"reconcile_abandoned_ops: marked {count} ops")

    return count
