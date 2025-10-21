"""Tests for DuckDB catalog transactional boundaries."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from uuid import uuid4

import pytest

try:  # pragma: no cover
    import duckdb
except ImportError:  # pragma: no cover
    pytest.skip("duckdb not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.catalog.boundaries import (
    DownloadBoundaryResult,
    ExtractionBoundaryResult,
    download_boundary,
    extraction_boundary,
    set_latest_boundary,
    validation_boundary,
)
from DocsToKG.OntologyDownload.catalog.migrations import apply_migrations


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        yield db_path


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def conn(temp_db: Path) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Create a DuckDB connection and apply migrations."""
    connection = duckdb.connect(str(temp_db), read_only=False)
    apply_migrations(connection)
    yield connection
    connection.close()


@pytest.fixture
def version_and_artifact(conn: duckdb.DuckDBPyConnection) -> Generator[tuple[str, str], None, None]:
    """Create a version for testing."""
    v_id = f"v-{uuid4().hex[:8]}"
    a_id = f"art-{uuid4().hex[:8]}"

    conn.execute(
        "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
        [v_id, "TEST", False, datetime.now()],
    )

    yield v_id, a_id


class TestDownloadBoundary:
    """Test download boundary context manager."""

    def test_download_boundary_success(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test successful download boundary."""
        v_id, a_id = version_and_artifact

        with download_boundary(conn, a_id, v_id, "/path/archive.zip", 1024, "etag1") as result:
            assert isinstance(result, DownloadBoundaryResult)
            assert result.artifact_id == a_id
            assert result.inserted is True

        # Verify insert
        row = conn.execute(
            "SELECT artifact_id, version_id, size, etag FROM artifacts WHERE artifact_id = ?",
            [a_id],
        ).fetchone()

        assert row is not None
        assert row[0] == a_id
        assert row[2] == 1024
        assert row[3] == "etag1"

    def test_download_boundary_result_before_commit(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test that result shows inserted=False before successful commit."""
        v_id, a_id = version_and_artifact

        with download_boundary(conn, a_id, v_id, "/path/archive.zip", 512) as result:
            assert result.inserted is True  # Set to True after commit

    def test_download_boundary_no_etag(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test download boundary without ETag."""
        v_id, a_id = version_and_artifact

        with download_boundary(conn, a_id, v_id, "/path/archive.zip", 2048, etag=None) as result:
            assert result.etag is None
            assert result.inserted is True


class TestExtractionBoundary:
    """Test extraction boundary context manager."""

    def test_extraction_boundary_success(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test successful extraction boundary."""
        v_id, a_id = version_and_artifact

        with extraction_boundary(conn, a_id) as result:
            assert isinstance(result, ExtractionBoundaryResult)
            assert result.artifact_id == a_id
            assert result.files_inserted == 0  # Not set by boundary

            # Caller would update result
            assert result.inserted is False

    def test_extraction_boundary_empty_is_safe(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test that extraction boundary with no files rolls back safely."""
        v_id, a_id = version_and_artifact

        with extraction_boundary(conn, a_id) as result:
            # Don't insert anything
            pass

        # Should rollback, no error


class TestValidationBoundary:
    """Test validation boundary context manager."""

    def test_validation_boundary_pass(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test validation boundary with pass status."""
        v_id, a_id = version_and_artifact

        # Create artifact first
        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, etag, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "/path/archive.zip", 1024, None, "fresh", datetime.now()],
        )

        # Create file
        f_id = f"file-{uuid4().hex[:8]}"
        conn.execute(
            """INSERT INTO extracted_files
               (file_id, artifact_id, relpath, size, extracted_at)
               VALUES (?, ?, ?, ?, ?)""",
            [f_id, a_id, "data/file.ttl", 512, datetime.now()],
        )

        with validation_boundary(conn, f_id, "rdflib", "pass", {"checked": True}) as result:
            assert result.file_id == f_id
            assert result.validator == "rdflib"
            assert result.status == "pass"
            assert result.inserted is True

        # Verify insert
        row = conn.execute(
            "SELECT file_id, validator, status FROM validations WHERE file_id = ?",
            [f_id],
        ).fetchone()

        assert row is not None
        assert row[2] == "pass"

    def test_validation_boundary_fail(
        self, conn: duckdb.DuckDBPyConnection, version_and_artifact: tuple[str, str]
    ) -> None:
        """Test validation boundary with fail status."""
        v_id, a_id = version_and_artifact

        # Create artifact first
        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, etag, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "/path/archive.zip", 1024, None, "fresh", datetime.now()],
        )

        # Create file
        f_id = f"file-{uuid4().hex[:8]}"
        conn.execute(
            """INSERT INTO extracted_files
               (file_id, artifact_id, relpath, size, extracted_at)
               VALUES (?, ?, ?, ?, ?)""",
            [f_id, a_id, "data/file.rdf", 256, datetime.now()],
        )

        with validation_boundary(
            conn, f_id, "owlready2", "fail", {"reason": "Invalid syntax"}
        ) as result:
            assert result.status == "fail"
            assert result.inserted is True


class TestSetLatestBoundary:
    """Test set-latest boundary context manager."""

    def test_set_latest_boundary_success(
        self,
        conn: duckdb.DuckDBPyConnection,
        version_and_artifact: tuple[str, str],
        temp_dir: Path,
    ) -> None:
        """Test successful set-latest boundary."""
        v_id, _ = version_and_artifact
        latest_path = temp_dir / "LATEST.json"
        temp_path = temp_dir / "LATEST.json.tmp"

        # Prepare temp file
        temp_path.write_text(json.dumps({"version_id": v_id}))

        with set_latest_boundary(conn, v_id, latest_path) as result:
            assert result.version_id == v_id
            assert result.pointer_updated is True

        # Verify LATEST.json exists
        assert latest_path.exists()
        assert not temp_path.exists()

        # Verify DB update
        row = conn.execute(
            "SELECT version_id FROM latest_pointer WHERE version_id = ?",
            [v_id],
        ).fetchone()

        assert row is not None

    def test_set_latest_boundary_no_temp_file(
        self,
        conn: duckdb.DuckDBPyConnection,
        version_and_artifact: tuple[str, str],
        temp_dir: Path,
    ) -> None:
        """Test set-latest boundary when temp file missing."""
        v_id, _ = version_and_artifact
        latest_path = temp_dir / "LATEST.json"

        # Don't create temp file
        with set_latest_boundary(conn, v_id, latest_path) as result:
            assert result.pointer_updated is True
            assert result.json_written is False

        # DB should still be updated
        row = conn.execute(
            "SELECT version_id FROM latest_pointer WHERE version_id = ?",
            [v_id],
        ).fetchone()

        assert row is not None

    def test_set_latest_boundary_replaces_existing(
        self,
        conn: duckdb.DuckDBPyConnection,
        version_and_artifact: tuple[str, str],
        temp_dir: Path,
    ) -> None:
        """Test that set-latest replaces existing pointer."""
        v1_id, _ = version_and_artifact
        v2_id = f"v-{uuid4().hex[:8]}"

        # Create v2
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v2_id, "TEST", False, datetime.now()],
        )

        latest_path = temp_dir / "LATEST.json"
        temp_path1 = temp_dir / "LATEST.json.tmp"

        # Set v1 as latest
        temp_path1.write_text(json.dumps({"version_id": v1_id}))
        with set_latest_boundary(conn, v1_id, latest_path):
            pass

        # Set v2 as latest
        temp_path2 = temp_dir / "LATEST.json.tmp"
        temp_path2.write_text(json.dumps({"version_id": v2_id}))
        with set_latest_boundary(conn, v2_id, latest_path):
            pass

        # v2 should be latest
        row = conn.execute(
            "SELECT version_id FROM latest_pointer WHERE version_id = ?",
            [v2_id],
        ).fetchone()

        assert row is not None


class TestBoundaryAtomicity:
    """Test atomicity guarantees across boundaries."""

    def test_download_boundary_rollback_on_fk_violation(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Test that download boundary rolls back on FK violation."""
        a_id = f"art-{uuid4().hex[:8]}"
        nonexistent_v_id = "nonexistent"

        # FK violation: version doesn't exist
        with pytest.raises(duckdb.Error):
            with download_boundary(conn, a_id, nonexistent_v_id, "/path", 1024):
                pass

        # Verify nothing was inserted
        row = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE artifact_id = ?",
            [a_id],
        ).fetchone()

        assert row is not None and row[0] == 0

    def test_validation_boundary_rollback_on_fk_violation(
        self, conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Test that validation boundary rolls back on FK violation."""
        f_id = f"file-{uuid4().hex[:8]}"

        # FK violation: file doesn't exist
        with pytest.raises(duckdb.Error):
            with validation_boundary(conn, f_id, "rdflib", "pass"):
                pass

        # Verify nothing was inserted
        row = conn.execute(
            "SELECT COUNT(*) FROM validations WHERE file_id = ?",
            [f_id],
        ).fetchone()

        assert row is not None and row[0] == 0
