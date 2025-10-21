"""Tests for DuckDB catalog garbage collection module."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

try:  # pragma: no cover
    import duckdb
except ImportError:  # pragma: no cover
    pytest.skip("duckdb not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.catalog.gc import (
    garbage_collect,
    identify_orphaned_artifacts,
    identify_orphaned_files,
    prune_by_retention_days,
    prune_keep_latest_n,
    vacuum_database,
)
from DocsToKG.OntologyDownload.catalog.migrations import apply_migrations


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.duckdb"


@pytest.fixture
def conn(temp_db: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and apply migrations."""
    connection = duckdb.connect(str(temp_db), read_only=False)
    apply_migrations(connection)
    yield connection
    connection.close()


class TestOrphanDetection:
    """Test orphan identification."""

    def test_identify_no_orphans_when_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test orphan detection on empty database."""
        orphans = identify_orphaned_artifacts(conn, older_than_days=1)
        assert orphans == []

    def test_identify_orphaned_artifacts(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test identification of artifacts with no extracted files."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"

        # Create old version
        old_ts = datetime.now() - timedelta(days=40)
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, old_ts],
        )

        # Create artifact with no extracted files
        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "/path/archive.zip", 1024, "fresh", old_ts],
        )

        orphans = identify_orphaned_artifacts(conn, older_than_days=30)

        assert len(orphans) == 1
        assert orphans[0].item_type == "artifact"
        assert orphans[0].item_id == a_id

    def test_identify_orphaned_files(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test identification of files with only failed validations."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"
        f_id = f"file-{uuid4().hex[:8]}"

        old_ts = datetime.now() - timedelta(days=40)

        # Create version and artifact
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, old_ts],
        )

        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "/path/archive.zip", 1024, "fresh", old_ts],
        )

        # Create file
        conn.execute(
            """INSERT INTO extracted_files
               (file_id, artifact_id, relpath, size, extracted_at)
               VALUES (?, ?, ?, ?, ?)""",
            [f_id, a_id, "data/file.ttl", 512, old_ts],
        )

        orphans = identify_orphaned_files(conn, older_than_days=30)

        assert len(orphans) == 1
        assert orphans[0].item_type == "file"
        assert orphans[0].item_id == f_id


class TestPruneByRetention:
    """Test prune by retention days."""

    def test_prune_dry_run_no_delete(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test dry-run doesn't actually delete."""
        v_id = f"v-{uuid4().hex[:8]}"
        old_ts = datetime.now() - timedelta(days=120)

        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, old_ts],
        )

        result = prune_by_retention_days(conn, keep_days=90, dry_run=True)

        assert result.dry_run is True
        assert result.items_deleted == 0
        assert result.items_identified == 1

    def test_prune_actually_delete(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test actual delete operation."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"
        old_ts = datetime.now() - timedelta(days=120)

        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, old_ts],
        )

        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "/path/archive.zip", 1024, "fresh", old_ts],
        )

        result = prune_by_retention_days(conn, keep_days=90, dry_run=False)

        assert result.items_deleted == 1
        assert result.bytes_freed == 1024
        assert result.dry_run is False

        # Verify deletion
        count = conn.execute("SELECT COUNT(*) FROM versions").fetchone()
        assert count[0] == 0


class TestPruneKeepLatestN:
    """Test prune keeping latest N versions."""

    def test_keep_latest_5_versions(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test keeping latest 5 versions."""
        # Create 10 versions
        for i in range(10):
            v_id = f"v{i:02d}-{uuid4().hex[:4]}"
            ts = datetime.now() - timedelta(days=10 - i)
            conn.execute(
                "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
                [v_id, "TEST", False, ts],
            )

        result = prune_keep_latest_n(conn, keep_count=5, dry_run=True)

        assert result.items_identified == 5  # Should delete 5 old ones

    def test_keep_latest_dry_run(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test dry-run doesn't delete."""
        for i in range(3):
            v_id = f"v{i:02d}-{uuid4().hex[:4]}"
            ts = datetime.now() - timedelta(days=3 - i)
            conn.execute(
                "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
                [v_id, "TEST", False, ts],
            )

        result = prune_keep_latest_n(conn, keep_count=2, dry_run=True)

        assert result.items_deleted == 0
        assert result.items_identified == 1


class TestVacuum:
    """Test vacuum operations."""

    def test_vacuum_success(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test VACUUM operation."""
        # Insert some data
        v_id = f"v-{uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, datetime.now()],
        )

        result = vacuum_database(conn)

        assert result.tables_vacuumed == 5
        assert result.duration_ms >= 0


class TestFullGarbageCollection:
    """Test full GC pipeline."""

    def test_full_gc_dry_run(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test full GC with dry-run."""
        # Create some versions
        for i in range(5):
            v_id = f"v{i:02d}-{uuid4().hex[:4]}"
            ts = datetime.now() - timedelta(days=i)
            conn.execute(
                "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
                [v_id, "TEST", False, ts],
            )

        prune_result, vacuum_result = garbage_collect(conn, keep_latest_n=2, dry_run=True)

        assert prune_result.dry_run is True
        assert prune_result.items_deleted == 0
        assert vacuum_result.tables_vacuumed == 0

    def test_full_gc_execute(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test full GC execution."""
        for i in range(5):
            v_id = f"v{i:02d}-{uuid4().hex[:4]}"
            ts = datetime.now() - timedelta(days=i)
            conn.execute(
                "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
                [v_id, "TEST", False, ts],
            )

        prune_result, vacuum_result = garbage_collect(conn, keep_latest_n=2, dry_run=False)

        assert prune_result.items_deleted >= 0
        assert vacuum_result.tables_vacuumed == 5
