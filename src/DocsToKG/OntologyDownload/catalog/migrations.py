# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.migrations",
#   "purpose": "Idempotent migration runner for DuckDB catalog schema",
#   "sections": [
#     {"id": "types", "name": "Data Types & Constants", "anchor": "TYP", "kind": "models"},
#     {"id": "migrations", "name": "Migration Definitions", "anchor": "MIG", "kind": "data"},
#     {"id": "runner", "name": "Migration Runner", "anchor": "RUN", "kind": "api"},
#     {"id": "queries", "name": "Schema Queries", "anchor": "QRY", "kind": "infra"}
#   ]
# }
# === /NAVMAP ===

"""Idempotent migration runner for DuckDB catalog schema.

Migrations are applied in order, once per schema version. Re-runs are safe
due to IF NOT EXISTS guards and INSERT OR IGNORE semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:  # pragma: no cover
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for catalog migrations. Ensure .venv is initialized."
    ) from exc

logger = logging.getLogger(__name__)


# ============================================================================
# DATA TYPES & CONSTANTS (TYP)
# ============================================================================


@dataclass
class MigrationResult:
    """Result of applying a migration."""

    migration_name: str
    applied: bool
    error: Optional[str] = None
    rows_affected: int = 0


# ============================================================================
# MIGRATION DEFINITIONS (MIG)
# ============================================================================

MIGRATIONS: List[Tuple[str, str]] = [
    (
        "0001_schema_version",
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            migration_name TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT now()
        );
        INSERT OR IGNORE INTO schema_version VALUES ('0001_schema_version', now());
        """,
    ),
    (
        "0002_versions",
        """
        CREATE TABLE IF NOT EXISTS versions (
            version_id TEXT PRIMARY KEY,
            service TEXT NOT NULL,
            latest_pointer BOOLEAN NOT NULL DEFAULT FALSE,
            ts TIMESTAMP NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_versions_service_latest
            ON versions(service, latest_pointer);
        CREATE INDEX IF NOT EXISTS idx_versions_ts
            ON versions(ts DESC);
        INSERT OR IGNORE INTO schema_version VALUES ('0002_versions', now());
        """,
    ),
    (
        "0003_artifacts_extracted_files",
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL REFERENCES versions(version_id),
            fs_relpath TEXT NOT NULL,
            size BIGINT NOT NULL,
            etag TEXT,
            status TEXT NOT NULL DEFAULT 'fresh',
            downloaded_at TIMESTAMP NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_version_status
            ON artifacts(version_id, status);
        CREATE INDEX IF NOT EXISTS idx_artifacts_artifact_id
            ON artifacts(artifact_id);

        CREATE TABLE IF NOT EXISTS extracted_files (
            file_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
            relpath TEXT NOT NULL,
            size BIGINT NOT NULL,
            format TEXT,
            sha256 TEXT,
            mtime TIMESTAMP,
            extracted_at TIMESTAMP NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_extracted_files_artifact_relpath
            ON extracted_files(artifact_id, relpath);
        CREATE INDEX IF NOT EXISTS idx_extracted_files_format
            ON extracted_files(format);
        CREATE INDEX IF NOT EXISTS idx_extracted_files_file_id
            ON extracted_files(file_id);

        INSERT OR IGNORE INTO schema_version VALUES ('0003_artifacts_extracted_files', now());
        """,
    ),
    (
        "0004_validations_events",
        """
        CREATE TABLE IF NOT EXISTS validations (
            validation_id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL REFERENCES extracted_files(file_id),
            validator TEXT NOT NULL,
            status TEXT NOT NULL,
            details JSON,
            validated_at TIMESTAMP NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_validations_file_validator
            ON validations(file_id, validator);
        CREATE INDEX IF NOT EXISTS idx_validations_status
            ON validations(status);
        CREATE INDEX IF NOT EXISTS idx_validations_validated_at
            ON validations(validated_at DESC);

        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            ts TIMESTAMP NOT NULL DEFAULT now(),
            type TEXT NOT NULL,
            level TEXT NOT NULL,
            payload JSON NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_run_ts
            ON events(run_id, ts DESC);

        INSERT OR IGNORE INTO schema_version VALUES ('0004_validations_events', now());
        """,
    ),
    (
        "0005_latest_pointer",
        """
        CREATE TABLE IF NOT EXISTS latest_pointer (
            version_id TEXT PRIMARY KEY REFERENCES versions(version_id),
            set_at TIMESTAMP NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_latest_pointer_set_at
            ON latest_pointer(set_at DESC);

        INSERT OR IGNORE INTO schema_version VALUES ('0005_latest_pointer', now());
        """,
    ),
]


# ============================================================================
# MIGRATION RUNNER (RUN)
# ============================================================================


def get_applied_migrations(conn: duckdb.DuckDBPyConnection) -> set[str]:
    """Get set of already-applied migration names.

    Args:
        conn: DuckDB connection

    Returns:
        Set of applied migration names
    """
    try:
        result = conn.execute("SELECT migration_name FROM schema_version").fetchall()
        return {row[0] for row in result}
    except duckdb.CatalogException:
        # schema_version table doesn't exist yet
        return set()


def apply_migrations(
    conn: duckdb.DuckDBPyConnection, dry_run: bool = False
) -> List[MigrationResult]:
    """Apply pending migrations in order.

    Migrations are idempotent: already-applied migrations are skipped.
    All pending migrations are applied in a single transaction.

    Args:
        conn: DuckDB connection (writer)
        dry_run: If True, don't commit; just report what would happen

    Returns:
        List of MigrationResult for each migration

    Raises:
        duckdb.Error: If any migration fails
    """
    applied = get_applied_migrations(conn)
    pending = [name for name, _ in MIGRATIONS if name not in applied]

    if not pending:
        logger.info("All migrations already applied (0 pending)")
        return [MigrationResult(name, False) for name, _ in MIGRATIONS if name not in applied]

    logger.info(f"Applying {len(pending)} pending migrations: {pending}")

    results: List[MigrationResult] = []

    try:
        conn.begin()

        for name, sql in MIGRATIONS:
            if name in applied:
                logger.debug(f"Skipping already-applied migration: {name}")
                continue

            try:
                logger.debug(f"Executing migration: {name}")
                conn.execute(sql)
                results.append(MigrationResult(name, True))
                logger.info(f"Applied migration: {name}")
            except duckdb.Error as exc:
                error_msg = f"Migration {name} failed: {exc}"
                logger.error(error_msg)
                results.append(MigrationResult(name, False, error=str(exc)))
                raise

        if dry_run:
            conn.rollback()
            logger.info("Dry run: rolled back all migrations")
        else:
            conn.commit()
            logger.info(f"Committed {len(results)} migrations")

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Migration transaction failed: {exc}")
        raise

    return results


def verify_schema(conn: duckdb.DuckDBPyConnection) -> bool:
    """Verify that all required tables exist.

    Args:
        conn: DuckDB connection

    Returns:
        True if all tables exist, False otherwise
    """
    required_tables = {
        "schema_version",
        "versions",
        "artifacts",
        "extracted_files",
        "validations",
        "events",
        "latest_pointer",
    }

    try:
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        existing = {row[0] for row in result}
        missing = required_tables - existing
        if missing:
            logger.warning(f"Missing tables: {missing}")
            return False
        logger.info(f"All {len(required_tables)} required tables exist")
        return True
    except duckdb.Error as exc:
        logger.error(f"Schema verification failed: {exc}")
        return False


# ============================================================================
# SCHEMA QUERIES (QRY)
# ============================================================================


def get_schema_version(conn: duckdb.DuckDBPyConnection) -> int:
    """Get the current schema version (number of applied migrations).

    Args:
        conn: DuckDB connection

    Returns:
        Number of applied migrations
    """
    try:
        result = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        return result[0] if result else 0
    except duckdb.CatalogException:
        return 0
