# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.schema_migration",
#   "purpose": "Database schema migrations for idempotent artifact job tracking",
#   "sections": [
#     {
#       "id": "get-migration-sql",
#       "name": "get_migration_sql",
#       "anchor": "function-get-migration-sql",
#       "kind": "function"
#     },
#     {
#       "id": "apply-migration",
#       "name": "apply_migration",
#       "anchor": "function-apply-migration",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Database schema migrations for idempotent artifact job tracking.

This module provides SQL migrations to add artifact job and operation ledgers
to the manifest/telemetry database. These tables enable:
  - Exactly-once job planning and execution
  - Per-operation idempotency tracking
  - Concurrency-safe leasing for multi-worker coordination
  - Crash recovery and state reconciliation

The migration is idempotent and can be applied multiple times safely.

Example:
  ```python
  import sqlite3
  from DocsToKG.ContentDownload.schema_migration import apply_migration

  conn = sqlite3.connect("manifest.sqlite3")
  apply_migration(conn)
  conn.close()
  ```
"""

from __future__ import annotations

import sqlite3

# SQL migration: add artifact_jobs, artifact_ops, and _meta tables
_MIGRATION_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=4000;

BEGIN IMMEDIATE;

-- Meta version gate
CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO _meta(key, value) VALUES ('schema_version', '3');

-- Artifact jobs: one row per URL intended to be fetched
CREATE TABLE IF NOT EXISTS artifact_jobs (
  job_id           TEXT PRIMARY KEY,
  work_id          TEXT NOT NULL,
  artifact_id      TEXT NOT NULL,
  canonical_url    TEXT NOT NULL,
  state            TEXT NOT NULL DEFAULT 'PLANNED',
  lease_owner      TEXT,
  lease_until      REAL,
  created_at       REAL NOT NULL,
  updated_at       REAL NOT NULL,
  idempotency_key  TEXT NOT NULL,
  UNIQUE(work_id, artifact_id, canonical_url),
  UNIQUE(idempotency_key),
  CHECK (state IN ('PLANNED','LEASED','HEAD_DONE','RESUME_OK','STREAMING','FINALIZED','INDEXED','DEDUPED','FAILED','SKIPPED_DUPLICATE'))
);
CREATE INDEX IF NOT EXISTS idx_artifact_jobs_state ON artifact_jobs(state);
CREATE INDEX IF NOT EXISTS idx_artifact_jobs_lease ON artifact_jobs(lease_until);

-- Exactly-once operation ledger (per side-effect)
CREATE TABLE IF NOT EXISTS artifact_ops (
  op_key        TEXT PRIMARY KEY,
  job_id        TEXT NOT NULL,
  op_type       TEXT NOT NULL,
  started_at    REAL NOT NULL,
  finished_at   REAL,
  result_code   TEXT,
  result_json   TEXT,
  FOREIGN KEY(job_id) REFERENCES artifact_jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_artifact_ops_job ON artifact_ops(job_id);

UPDATE _meta SET value='3' WHERE key='schema_version';
COMMIT;
"""


def get_migration_sql() -> str:
    """Return the complete migration SQL as a string."""
    return _MIGRATION_SQL


def apply_migration(conn: sqlite3.Connection) -> None:
    """Apply the idempotency schema migration to the given connection.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection (should be manifest/telemetry DB)

    Raises
    ------
    sqlite3.Error
        If migration fails (e.g., schema conflicts)

    Notes
    -----
    This migration is idempotent and can be called multiple times safely.
    It enables artifact_jobs, artifact_ops, and state machine tables.
    """
    conn.executescript(_MIGRATION_SQL)


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read current schema version from _meta table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection

    Returns
    -------
    int
        Schema version (0 if _meta table doesn't exist)
    """
    try:
        row = conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
