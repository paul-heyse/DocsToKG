"""Database schema migrations for streaming architecture.

This module provides idempotent schema migrations for the artifact streaming
system, including:
  - artifact_jobs: State machine + lease management for download jobs
  - artifact_ops: Operation ledger for exactly-once effect tracking
  - Backward compatibility with existing manifest.sqlite3 schema

Version History:
  1: Initial schema (artifact_jobs, artifact_ops, indexes)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)

# Schema version constant
SCHEMA_VERSION = 1


# ============================================================================
# Schema Definitions
# ============================================================================


def _schema_v1() -> dict[str, str]:
    """Schema version 1: Initial artifact streaming tables.

    Returns:
        Dict mapping table name to CREATE TABLE statement
    """
    return {
        "artifact_jobs": """
            CREATE TABLE IF NOT EXISTS artifact_jobs (
                job_id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'PLANNED',
                lease_owner TEXT,
                lease_until REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                idempotency_key TEXT NOT NULL,
                UNIQUE(work_id, artifact_id, canonical_url),
                UNIQUE(idempotency_key),
                CHECK (state IN ('PLANNED','LEASED','HEAD_DONE','RESUME_OK','STREAMING',
                                'FINALIZED','INDEXED','DEDUPED','FAILED','SKIPPED_DUPLICATE'))
            )
        """,
        "artifact_ops": """
            CREATE TABLE IF NOT EXISTS artifact_ops (
                op_key TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                op_type TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL,
                result_code TEXT,
                result_json TEXT,
                FOREIGN KEY(job_id) REFERENCES artifact_jobs(job_id) ON DELETE CASCADE
            )
        """,
        "artifact_jobs_state_idx": """
            CREATE INDEX IF NOT EXISTS idx_artifact_jobs_state
            ON artifact_jobs(state)
        """,
        "artifact_jobs_lease_idx": """
            CREATE INDEX IF NOT EXISTS idx_artifact_jobs_lease
            ON artifact_jobs(lease_until, state)
        """,
        "artifact_jobs_idempotency_idx": """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_jobs_idempotency
            ON artifact_jobs(idempotency_key)
        """,
        "artifact_ops_job_idx": """
            CREATE INDEX IF NOT EXISTS idx_artifact_ops_job
            ON artifact_ops(job_id)
        """,
        "artifact_ops_type_idx": """
            CREATE INDEX IF NOT EXISTS idx_artifact_ops_type
            ON artifact_ops(op_type)
        """,
    }


# ============================================================================
# Migration Logic
# ============================================================================


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from database.

    Args:
        conn: SQLite database connection

    Returns:
        Schema version (0 if not initialized)
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT schema_version FROM __schema_version__")
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Set schema version in database.

    Args:
        conn: SQLite database connection
        version: Schema version to set
    """
    cursor = conn.cursor()

    # Create version table if needed
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS __schema_version__ (
            schema_version INTEGER PRIMARY KEY,
            updated_at REAL
        )
        """
    )

    # Update version
    import time

    cursor.execute(
        "INSERT OR REPLACE INTO __schema_version__(schema_version, updated_at) VALUES (?, ?)",
        (version, time.time()),
    )
    conn.commit()


def migrate_to_v1(conn: sqlite3.Connection) -> None:
    """Migrate to schema version 1.

    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    schema = _schema_v1()

    # Create tables and indexes
    for name, ddl in schema.items():
        try:
            cursor.execute(ddl)
            LOGGER.info(f"Created {name}")
        except sqlite3.OperationalError as e:
            LOGGER.warning(f"Skipping {name}: {e}")

    conn.commit()
    set_schema_version(conn, 1)
    LOGGER.info("Migration to v1 complete")


def run_migrations(conn: sqlite3.Connection, target_version: int = SCHEMA_VERSION) -> None:
    """Run all pending migrations up to target version.

    Args:
        conn: SQLite database connection
        target_version: Target schema version (default: latest)
    """
    current = get_schema_version(conn)

    if current >= target_version:
        LOGGER.info(f"Schema already at v{current}, no migrations needed")
        return

    LOGGER.info(f"Migrating from v{current} to v{target_version}")

    if current < 1 and target_version >= 1:
        migrate_to_v1(conn)

    LOGGER.info("All migrations complete")


# ============================================================================
# Initialization & Validation
# ============================================================================


def ensure_schema(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Ensure database schema is initialized.

    Args:
        db_path: Path to database file (default: manifest.sqlite3 in DOCSTOKG_DATA_ROOT)

    Returns:
        SQLite database connection with schema initialized
    """
    if db_path is None:
        from DocsToKG.ContentDownload.download import DownloadConfig

        cfg = DownloadConfig()
        db_path = cfg.manifest_db_path

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Run migrations
    run_migrations(conn, SCHEMA_VERSION)

    return conn


def validate_schema(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
    """Validate that schema is correctly initialized.

    Args:
        conn: SQLite database connection

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    cursor = conn.cursor()

    # Check tables exist
    required_tables = {"artifact_jobs", "artifact_ops"}
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'artifact_%'")
    existing = {row[0] for row in cursor.fetchall()}

    for table in required_tables:
        if table not in existing:
            errors.append(f"Table {table} not found")

    # Check key indexes
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_artifact_%'"
    )
    existing_indexes = {row[0] for row in cursor.fetchall()}

    required_indexes = {
        "idx_artifact_jobs_state",
        "idx_artifact_jobs_lease",
        "idx_artifact_ops_job",
    }

    for idx in required_indexes:
        if idx not in existing_indexes:
            errors.append(f"Index {idx} not found")

    # Check schema version
    version = get_schema_version(conn)
    if version < SCHEMA_VERSION:
        errors.append(f"Schema version {version} < required {SCHEMA_VERSION}")

    return len(errors) == 0, errors


def repair_schema(conn: sqlite3.Connection) -> None:
    """Repair schema by dropping and recreating corrupted tables.

    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()

    # Drop corrupted tables (artifact_* only, preserve existing manifest tables)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'artifact_%'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            LOGGER.info(f"Dropped corrupted table {table}")
        except Exception as e:
            LOGGER.error(f"Failed to drop {table}: {e}")

    # Drop indexes too
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_artifact_%'"
    )
    indexes = [row[0] for row in cursor.fetchall()]

    for idx in indexes:
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {idx}")
            LOGGER.info(f"Dropped corrupted index {idx}")
        except Exception as e:
            LOGGER.error(f"Failed to drop {idx}: {e}")

    conn.commit()

    # Reset schema version to force full migration
    cursor.execute("DELETE FROM __schema_version__")
    conn.commit()

    # Reinitialize schema
    run_migrations(conn, SCHEMA_VERSION)


# ============================================================================
# Helpers for Integration
# ============================================================================


def get_or_create_connection(
    db_path: Optional[str | Path] = None,
) -> sqlite3.Connection:
    """Get or create database connection with schema initialized.

    Args:
        db_path: Path to database file

    Returns:
        SQLite database connection
    """
    conn = ensure_schema(db_path)

    # Validate schema
    is_valid, errors = validate_schema(conn)
    if not is_valid:
        LOGGER.warning(f"Schema validation errors: {errors}")
        LOGGER.info("Attempting schema repair...")
        repair_schema(conn)

    return conn


def close_connection(conn: sqlite3.Connection) -> None:
    """Close database connection gracefully.

    Args:
        conn: SQLite database connection
    """
    try:
        conn.close()
    except Exception as e:
        LOGGER.error(f"Error closing connection: {e}")


# ============================================================================
# Context Manager for Transaction Management
# ============================================================================


class StreamingDatabase:
    """Context manager for streaming database operations.

    Usage:
        with StreamingDatabase() as db:
            # Perform database operations
            pass
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        """Initialize database context manager.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        """Enter context (acquire connection)."""
        self.conn = get_or_create_connection(self.db_path)
        return self.conn

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context (release connection)."""
        if self.conn:
            try:
                if exc_type is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
            finally:
                close_connection(self.conn)


# ============================================================================
# Database Health Check
# ============================================================================


def health_check(db_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Perform comprehensive database health check.

    Args:
        db_path: Path to database file

    Returns:
        Health check results dict
    """
    try:
        conn = ensure_schema(db_path)

        is_valid, errors = validate_schema(conn)

        cursor = conn.cursor()

        # Count rows in each table
        cursor.execute("SELECT COUNT(*) FROM artifact_jobs")
        jobs_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM artifact_ops")
        ops_count = cursor.fetchone()[0]

        # Get schema version
        version = get_schema_version(conn)

        conn.close()

        return {
            "status": "healthy" if is_valid else "degraded",
            "schema_version": version,
            "tables": {
                "artifact_jobs": {"row_count": jobs_count},
                "artifact_ops": {"row_count": ops_count},
            },
            "errors": errors,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
