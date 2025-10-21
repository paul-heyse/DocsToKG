# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.orchestrator.queue",
#   "purpose": "SQLite-backed work queue with idempotent enqueue, crash-safe leasing, and retry logic",
#   "sections": [
#     {"id": "workqueue", "name": "WorkQueue", "anchor": "#class-workqueue", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""SQLite-backed work queue for ContentDownload orchestration.

This module provides a durable, idempotent work queue implementation using SQLite
with WAL mode for concurrent access. It guarantees:

- **Idempotence**: Duplicate enqueues are safe (artifact_id unique index)
- **Crash-safety**: Leasing with TTL enables recovery on worker crashes
- **Reliability**: Atomic state transitions prevent partial failures
- **Scalability**: SQLite WAL mode enables concurrent readers/writers

**Design:**

The queue implements a job state machine:

    QUEUED
      ↓ (lease) → IN_PROGRESS (with worker_id, lease_expires_at)
      ↓ (ack)
      ├→ DONE
      ├→ SKIPPED
      └→ ERROR (after max_attempts)

**Usage:**

    queue = WorkQueue("state/workqueue.sqlite", wal_mode=True)

    # Enqueue artifacts (idempotent)
    queue.enqueue("doi:10.1234/example", {"doi": "10.1234/example"})

    # Lease jobs for processing
    jobs = queue.lease("worker-1", limit=5, lease_ttl_sec=600)

    # Process jobs, then ack with outcome
    queue.ack(job["id"], "done", last_error=None)

    # On failure, retry
    queue.fail_and_retry(job["id"], backoff_sec=60, max_attempts=3, last_error=str(e))

    # Monitor
    stats = queue.stats()  # {"queued": 10, "in_progress": 2, ...}

**Schema:**

    CREATE TABLE jobs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      artifact_id TEXT NOT NULL UNIQUE,
      artifact_json TEXT NOT NULL,
      state TEXT NOT NULL DEFAULT 'queued',
      attempts INTEGER NOT NULL DEFAULT 0,
      last_error TEXT,
      resolver_hint TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      lease_expires_at TEXT,
      worker_id TEXT
    );

**Thread Safety:**

WAL mode allows concurrent readers; writers serialize via SQLite locking.
Multiple threads can safely call these methods on the same WorkQueue instance.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from ..orchestrator.models import JobState

__all__ = ["WorkQueue"]

logger = logging.getLogger(__name__)

# SQL schema for jobs table
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id TEXT NOT NULL UNIQUE,
  artifact_json TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'queued',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  resolver_hint TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  lease_expires_at TEXT,
  worker_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


class WorkQueue:
    """SQLite-backed work queue with idempotent enqueue and crash-safe leasing.

    Provides durable job coordination with exactly-once semantics through
    atomic state transitions and worker leasing with TTL-based recovery.
    """

    def __init__(self, path: str, wal_mode: bool = True) -> None:
        """Initialize work queue.

        Args:
            path: Path to SQLite database file
            wal_mode: Enable WAL mode for concurrent access (default True)
        """
        self.path = path
        self._connection_lock = threading.Lock()
        self._local = threading.local()  # Per-thread connection storage

        # Create parent directories if needed
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database with WAL mode if requested
        conn = sqlite3.connect(path, timeout=10.0)
        if wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        conn.close()

        logger.info(f"WorkQueue initialized at {path} (wal_mode={wal_mode})")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Implements per-thread connection pooling to reduce overhead
        of creating new connections on every operation.

        Returns:
            SQLite connection for current thread
        """
        # Check if thread-local connection exists and is still valid
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # Lazy create connection for this thread
            conn = sqlite3.connect(self.path, timeout=10.0)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def close_connection(self) -> None:
        """Close thread-local database connection.

        Should be called when a worker thread is shutting down to
        ensure proper cleanup of database resources.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._local.conn = None

    def enqueue(
        self,
        artifact_id: str,
        artifact: Mapping[str, Any],
        resolver_hint: Optional[str] = None,
    ) -> bool:
        """Idempotently enqueue an artifact for processing.

        Args:
            artifact_id: Unique artifact identifier (e.g., "doi:10.1234/example")
            artifact: Artifact payload (serialized to JSON)
            resolver_hint: Optional hint about which resolver might work

        Returns:
            True if enqueued (new), False if already exists (idempotent)

        Raises:
            sqlite3.Error: If database operation fails
        """
        conn = self._get_connection()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            artifact_json = json.dumps(dict(artifact))

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO jobs
                (artifact_id, artifact_json, state, resolver_hint, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    artifact_json,
                    JobState.QUEUED.value,
                    resolver_hint,
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()

            was_inserted = cursor.rowcount > 0
            if was_inserted:
                logger.debug(f"Enqueued artifact {artifact_id}")
            else:
                logger.debug(f"Artifact {artifact_id} already enqueued (idempotent)")

            return was_inserted
        finally:
            pass  # Don't close; keep thread-local connection alive

    def lease(
        self,
        worker_id: str,
        limit: int,
        lease_ttl_sec: int,
    ) -> list[dict[str, Any]]:
        """Atomically lease up to `limit` jobs for a worker.

        Moves jobs from QUEUED (or stale IN_PROGRESS) to IN_PROGRESS state,
        assigning them to the worker. Implements crash recovery by re-leasing
        jobs whose lease has expired.

        Args:
            worker_id: Unique worker identifier
            limit: Maximum jobs to lease
            lease_ttl_sec: Lease time-to-live in seconds

        Returns:
            List of job dicts ready for processing

        Raises:
            sqlite3.Error: If database operation fails
        """
        conn = self._get_connection()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            lease_expires = datetime.now(timezone.utc) + timedelta(seconds=lease_ttl_sec)
            lease_expires_iso = lease_expires.isoformat()

            # Atomically lease queued jobs and stale in-progress jobs
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, worker_id = ?, lease_expires_at = ?, updated_at = ?
                WHERE id IN (
                    SELECT id FROM jobs
                    WHERE (state = ? OR (state = ? AND lease_expires_at < ?))
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (
                    JobState.IN_PROGRESS.value,
                    worker_id,
                    lease_expires_iso,
                    now_iso,
                    JobState.QUEUED.value,
                    JobState.IN_PROGRESS.value,
                    now_iso,
                    limit,
                ),
            )
            conn.commit()

            # Fetch the leased jobs
            leased_jobs = conn.execute(
                """
                SELECT id, artifact_id, artifact_json, state, attempts, resolver_hint
                FROM jobs
                WHERE worker_id = ? AND state = ? AND lease_expires_at > ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (worker_id, JobState.IN_PROGRESS.value, now_iso, limit),
            ).fetchall()

            result = [
                {
                    "id": row["id"],
                    "artifact_id": row["artifact_id"],
                    "artifact_json": row["artifact_json"],
                    "state": row["state"],
                    "attempts": row["attempts"],
                    "resolver_hint": row["resolver_hint"],
                }
                for row in leased_jobs
            ]

            if result:
                logger.debug(f"Worker {worker_id} leased {len(result)} jobs")

            return result
        finally:
            pass  # Don't close; keep thread-local connection alive

    def heartbeat(self, worker_id: str, lease_ttl_sec: int = 600) -> None:
        """Extend lease for active worker.

        Updates lease_expires_at to current time + lease_ttl_sec, keeping the worker's
        jobs from being reclaimed by other workers. The lease extension duration
        is configurable to match the orchestrator's lease TTL setting.

        Args:
            worker_id: Worker identifier
            lease_ttl_sec: Seconds to extend lease (should match config.lease_ttl_seconds)

        Raises:
            sqlite3.Error: If database operation fails
        """
        conn = self._get_connection()
        try:
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()

            # Calculate lease expiration time
            lease_expires = now + timedelta(seconds=lease_ttl_sec)
            lease_expires_iso = lease_expires.isoformat()

            cursor = conn.execute(
                """
                UPDATE jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE worker_id = ? AND state = ?
                """,
                (lease_expires_iso, now_iso, worker_id, JobState.IN_PROGRESS.value),
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(
                    f"Heartbeat for worker {worker_id} extended {cursor.rowcount} leases (ttl={lease_ttl_sec}s)"
                )
        finally:
            pass  # Don't close; keep thread-local connection alive

    def ack(
        self,
        job_id: int,
        outcome: str,
        last_error: Optional[str] = None,
    ) -> None:
        """Acknowledge job completion with outcome.

        Transitions job to terminal state (DONE, SKIPPED, or ERROR).

        Args:
            job_id: Job ID
            outcome: Terminal outcome (e.g., "done", "skipped", "error")
            last_error: Optional error message if outcome is error

        Raises:
            ValueError: If outcome is not a valid JobState
            sqlite3.Error: If database operation fails
        """
        # Map outcome string to JobState
        outcome_lower = outcome.lower()
        if outcome_lower == "done":
            state = JobState.DONE
        elif outcome_lower == "skipped":
            state = JobState.SKIPPED
        elif outcome_lower == "error":
            state = JobState.ERROR
        else:
            raise ValueError(f"Invalid outcome: {outcome}")

        conn = self._get_connection()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()

            cursor = conn.execute(
                """
                UPDATE jobs
                SET state = ?, last_error = ?, worker_id = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (state.value, last_error, now_iso, job_id),
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(f"Job {job_id} acked with outcome={outcome}")
            else:
                logger.warning(f"Job {job_id} not found for ack")
        finally:
            pass  # Don't close; keep thread-local connection alive

    def fail_and_retry(
        self,
        job_id: int,
        backoff_sec: int,
        max_attempts: int,
        last_error: str,
    ) -> None:
        """Increment attempts and retry or fail job.

        On failure, either re-queues the job (with a delay) or marks it ERROR
        if max_attempts has been exceeded.

        Args:
            job_id: Job ID
            backoff_sec: Seconds to delay before retrying
            max_attempts: Maximum attempts allowed
            last_error: Error message describing the failure

        Raises:
            sqlite3.Error: If database operation fails
        """
        conn = self._get_connection()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()

            # Increment attempts
            conn.execute(
                """
                UPDATE jobs
                SET attempts = attempts + 1,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (last_error, now_iso, job_id),
            )

            # Check if we should retry or fail
            job = conn.execute("SELECT attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                logger.warning(f"Job {job_id} not found for fail_and_retry")
                return

            attempts = job["attempts"]

            if attempts < max_attempts:
                # Re-queue with exponential backoff (simple: just use backoff_sec)
                delay_expires = (
                    datetime.now(timezone.utc) + timedelta(seconds=backoff_sec)
                ).isoformat()

                conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, worker_id = NULL, lease_expires_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (JobState.QUEUED.value, delay_expires, now_iso, job_id),
                )
                logger.debug(
                    f"Job {job_id} retry scheduled (attempts={attempts}/{max_attempts}, "
                    f"backoff={backoff_sec}s)"
                )
            else:
                # Mark as error
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, worker_id = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (JobState.ERROR.value, now_iso, job_id),
                )
                logger.warning(
                    f"Job {job_id} marked ERROR after {attempts} attempts. Error: {last_error}"
                )

            conn.commit()
        finally:
            pass  # Don't close; keep thread-local connection alive

    def stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            Dict with counts for each state and total

        Raises:
            sqlite3.Error: If database operation fails
        """
        conn = self._get_connection()
        try:
            counts = conn.execute(
                """
                SELECT state, COUNT(*) as count FROM jobs GROUP BY state
                """
            ).fetchall()

            stats_dict: dict[str, int] = {
                "queued": 0,
                "in_progress": 0,
                "done": 0,
                "skipped": 0,
                "error": 0,
                "total": 0,
            }

            for row in counts:
                state = row["state"]
                count = row["count"]
                stats_dict[state] = count
                stats_dict["total"] += count

            return stats_dict
        finally:
            pass  # Don't close; keep thread-local connection alive
