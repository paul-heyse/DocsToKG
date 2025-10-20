"""
Schema migration helpers for Wayback telemetry SQLite database.

This module provides a clean way to evolve the database schema over time while
maintaining backward compatibility and idempotency.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable, Dict

LOGGER = logging.getLogger(__name__)


class Migration:
    """Represents a single schema migration."""

    def __init__(
        self, version: str, description: str, upgrade_fn: Callable[[sqlite3.Connection], None]
    ):
        """Initialize a migration.

        Args:
            version: Semantic version string (e.g., "2", "1.1.0").
            description: Human-readable description of the change.
            upgrade_fn: Async-safe function that performs the migration.
        """
        self.version = version
        self.description = description
        self.upgrade_fn = upgrade_fn

    def apply(self, conn: sqlite3.Connection) -> None:
        """Apply this migration to a connection."""
        try:
            self.upgrade_fn(conn)
            LOGGER.info(f"Applied migration {self.version}: {self.description}")
        except Exception as e:
            LOGGER.error(f"Failed to apply migration {self.version}: {e}")
            raise


def migration_add_run_metrics_table(conn: sqlite3.Connection) -> None:
    """Migration: Add wayback_run_metrics roll-up table."""
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS wayback_run_metrics (
        run_id TEXT PRIMARY KEY,
        attempts INTEGER DEFAULT 0,
        emits INTEGER DEFAULT 0,
        yield_pct REAL DEFAULT 0.0,
        p95_latency_ms REAL,
        cache_hit_pct REAL DEFAULT 0.0,
        non_pdf_rate REAL DEFAULT 0.0,
        below_min_size_rate REAL DEFAULT 0.0,
        created_at TEXT,
        updated_at TEXT
    );
    """
    )
    conn.commit()


def migration_add_composite_indexes(conn: sqlite3.Connection) -> None:
    """Migration: Add composite/covering indexes for common queries."""
    c = conn.cursor()

    # These may already exist; CREATE IF NOT EXISTS prevents errors
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_attempts_run_result ON wayback_attempts(run_id, result);"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_emits_run_mode ON wayback_emits(run_id, source_mode, memento_ts);"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_discovery_stage_run ON wayback_discoveries(run_id, stage);"
    )

    # Partial index for successful attempts
    try:
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_success ON wayback_attempts(result) WHERE result LIKE 'emitted%';"
        )
    except sqlite3.OperationalError:
        pass  # Older SQLite versions don't support partial indexes

    conn.commit()


# Registry of all migrations (in order of application)
MIGRATIONS: Dict[str, Migration] = {
    "2": Migration(
        "2",
        "Add run_metrics roll-up table and composite indexes",
        lambda conn: (
            migration_add_run_metrics_table(conn),
            migration_add_composite_indexes(conn),
        )[
            1
        ],  # Execute both, return None
    ),
}


def get_current_schema_version(conn: sqlite3.Connection) -> str:
    """Get the current schema version from the database.

    Args:
        conn: SQLite connection.

    Returns:
        Version string (e.g., "1", "2"). Defaults to "1" if not set.
    """
    try:
        c = conn.cursor()
        c.execute("SELECT value FROM _meta WHERE key = 'wayback_schema_version' LIMIT 1;")
        row = c.fetchone()
        return row[0] if row else "1"
    except sqlite3.OperationalError:
        # _meta table doesn't exist yet
        return "1"


def set_schema_version(conn: sqlite3.Connection, version: str) -> None:
    """Set the schema version in the database.

    Args:
        conn: SQLite connection.
        version: Version string to set.
    """
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO _meta (key, value) VALUES ('wayback_schema_version', ?);",
        (version,),
    )
    conn.commit()


def migrate_schema(conn: sqlite3.Connection, target_version: str = "2") -> bool:
    """Migrate the database schema to a target version.

    This function is idempotent: it can be called multiple times safely.

    Args:
        conn: SQLite connection.
        target_version: Target version to migrate to (default: latest).

    Returns:
        True if migration was successful, False otherwise.
    """
    try:
        current = get_current_schema_version(conn)

        if current == target_version:
            LOGGER.debug(f"Schema already at version {current}")
            return True

        LOGGER.info(f"Migrating schema from {current} to {target_version}")

        # Apply all migrations between current and target
        versions = sorted(MIGRATIONS.keys(), key=lambda v: tuple(map(int, v.split("."))))

        for version in versions:
            if version > current and version <= target_version:
                migration = MIGRATIONS[version]
                migration.apply(conn)

        set_schema_version(conn, target_version)
        LOGGER.info(f"Schema migration to {target_version} completed successfully")
        return True

    except Exception as e:
        LOGGER.error(f"Schema migration failed: {e}")
        return False
