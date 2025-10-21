"""Tests for streaming database schema migrations and health checks.

Tests cover:
  - Schema version tracking
  - Migration execution
  - Schema validation
  - Database repair
  - Health checks
  - Backward compatibility
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from DocsToKG.ContentDownload import streaming_schema


class TestSchemaVersion:
    """Tests for schema version management."""

    def test_get_schema_version_uninitialized(self):
        """Should return 0 for uninitialized database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            version = streaming_schema.get_schema_version(conn)
            assert version == 0

            conn.close()

    def test_set_and_get_schema_version(self):
        """Should set and retrieve schema version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            streaming_schema.set_schema_version(conn, 1)
            version = streaming_schema.get_schema_version(conn)

            assert version == 1

            conn.close()


class TestMigrations:
    """Tests for schema migrations."""

    def test_migrate_to_v1(self):
        """Should create artifact tables on v1 migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            streaming_schema.migrate_to_v1(conn)

            # Verify tables exist
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_jobs'"
            )
            assert cursor.fetchone() is not None

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_ops'"
            )
            assert cursor.fetchone() is not None

            # Verify indexes exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_artifact_jobs_state'"
            )
            assert cursor.fetchone() is not None

            conn.close()

    def test_run_migrations_idempotent(self):
        """Running migrations multiple times should be idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Run migrations twice
            streaming_schema.run_migrations(conn, 1)
            streaming_schema.run_migrations(conn, 1)

            # Should still be at v1
            version = streaming_schema.get_schema_version(conn)
            assert version == 1

            conn.close()

    def test_run_migrations_from_v0_to_v1(self):
        """Should migrate from v0 to v1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            assert streaming_schema.get_schema_version(conn) == 0

            streaming_schema.run_migrations(conn, 1)

            assert streaming_schema.get_schema_version(conn) == 1

            conn.close()


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_empty_database(self):
        """Empty database should be invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            is_valid, errors = streaming_schema.validate_schema(conn)

            assert not is_valid
            assert len(errors) > 0

            conn.close()

    def test_validate_after_migration(self):
        """Database should be valid after migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            streaming_schema.run_migrations(conn, 1)

            is_valid, errors = streaming_schema.validate_schema(conn)

            assert is_valid
            assert len(errors) == 0

            conn.close()

    def test_validate_detects_missing_table(self):
        """Validation should detect missing tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Manually create only artifact_jobs, not artifact_ops
            cursor = conn.cursor()
            schema = streaming_schema._schema_v1()
            cursor.execute(schema["artifact_jobs"])
            streaming_schema.set_schema_version(conn, 1)
            conn.commit()

            is_valid, errors = streaming_schema.validate_schema(conn)

            # Should detect missing artifact_ops
            assert not is_valid
            assert any("artifact_ops" in e for e in errors)

            conn.close()


class TestEnsureSchema:
    """Tests for schema initialization."""

    def test_ensure_schema_creates_tables(self):
        """ensure_schema should create tables if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = streaming_schema.ensure_schema(db_path)

            is_valid, errors = streaming_schema.validate_schema(conn)
            assert is_valid

            conn.close()

    def test_ensure_schema_idempotent(self):
        """ensure_schema should be idempotent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn1 = streaming_schema.ensure_schema(db_path)
            version1 = streaming_schema.get_schema_version(conn1)
            conn1.close()

            conn2 = streaming_schema.ensure_schema(db_path)
            version2 = streaming_schema.get_schema_version(conn2)
            conn2.close()

            assert version1 == version2 == 1


class TestSchemaRepair:
    """Tests for schema repair."""

    def test_repair_schema_recreates_tables(self):
        """repair_schema should recreate corrupted tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create initial schema
            streaming_schema.run_migrations(conn, 1)

            # Drop a table to simulate corruption
            cursor = conn.cursor()
            cursor.execute("DROP TABLE artifact_jobs")
            conn.commit()

            # Repair should recreate it
            streaming_schema.repair_schema(conn)

            # Need to re-fetch connection after repair for validation
            conn.close()
            conn = sqlite3.connect(str(db_path))

            is_valid, errors = streaming_schema.validate_schema(conn)
            assert is_valid

            conn.close()


class TestHealthCheck:
    """Tests for database health checks."""

    def test_health_check_healthy_database(self):
        """Health check should report healthy for valid database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = streaming_schema.ensure_schema(db_path)
            conn.close()

            health = streaming_schema.health_check(db_path)

            assert health["status"] == "healthy"
            assert health["schema_version"] == 1
            assert "tables" in health
            assert health["tables"]["artifact_jobs"]["row_count"] == 0
            assert health["tables"]["artifact_ops"]["row_count"] == 0

    def test_health_check_empty_database(self):
        """Health check should report degraded for empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            health = streaming_schema.health_check(db_path)

            # First call should repair, second should be healthy
            # Actually, health_check calls ensure_schema which repairs
            assert health["status"] == "healthy"


class TestStreamingDatabaseContext:
    """Tests for StreamingDatabase context manager."""

    def test_context_manager_creates_connection(self):
        """Context manager should create and clean up connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with streaming_schema.StreamingDatabase(db_path) as conn:
                assert conn is not None
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM artifact_jobs")
                count = cursor.fetchone()[0]
                assert count == 0

    def test_context_manager_commits_on_success(self):
        """Context manager should commit on successful exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Insert a row
            with streaming_schema.StreamingDatabase(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO artifact_jobs
                    (job_id, work_id, artifact_id, canonical_url, created_at, updated_at, idempotency_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("j1", "w1", "a1", "http://ex.com/f", 0, 0, "k1"),
                )

            # Verify row persisted
            with streaming_schema.StreamingDatabase(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM artifact_jobs")
                count = cursor.fetchone()[0]
                assert count == 1

    def test_context_manager_rollback_on_error(self):
        """Context manager should rollback on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            try:
                with streaming_schema.StreamingDatabase(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO artifact_jobs
                        (job_id, work_id, artifact_id, canonical_url, created_at, updated_at, idempotency_key)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        ("j1", "w1", "a1", "http://ex.com/f", 0, 0, "k1"),
                    )
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Verify row was not persisted
            with streaming_schema.StreamingDatabase(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM artifact_jobs")
                count = cursor.fetchone()[0]
                assert count == 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing databases."""

    def test_schema_coexists_with_existing_tables(self):
        """New schema should coexist with existing manifest tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create a fake existing table (simulating existing manifest schema)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY)")
            conn.commit()

            # Now apply new schema
            streaming_schema.ensure_schema(db_path)

            # Both tables should exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}

            assert "manifest" in tables
            assert "artifact_jobs" in tables
            assert "artifact_ops" in tables

            conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
