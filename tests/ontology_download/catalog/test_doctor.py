"""Tests for DuckDB catalog doctor module."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

try:  # pragma: no cover
    import duckdb
except ImportError:  # pragma: no cover
    pytest.skip("duckdb not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.catalog.doctor import (
    DoctorReport,
    HealthCheckResult,
    detect_db_fs_drifts,
    generate_doctor_report,
    quick_health_check,
    scan_filesystem_artifacts,
    scan_filesystem_files,
)
from DocsToKG.OntologyDownload.catalog.migrations import apply_migrations


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.duckdb"


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary artifact and extract directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def conn(temp_db: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and apply migrations."""
    connection = duckdb.connect(str(temp_db), read_only=False)
    apply_migrations(connection)
    yield connection
    connection.close()


class TestHealthCheck:
    """Test quick health check."""

    def test_health_check_empty_database(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test health check on empty database."""
        result = quick_health_check(conn)

        assert isinstance(result, HealthCheckResult)
        assert result.is_healthy is True
        assert result.artifact_count == 0
        assert result.file_count == 0
        assert "migrations applied" in result.message or "schema v5" in result.message

    def test_health_check_with_data(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test health check with populated data."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"

        # Insert version and artifact
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, datetime.now()],
        )

        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "/path/archive.zip", 1024, "fresh", datetime.now()],
        )

        result = quick_health_check(conn)

        assert result.is_healthy is True
        assert result.artifact_count == 1
        assert result.file_count == 0


class TestFilesystemScanning:
    """Test filesystem scanning functions."""

    def test_scan_artifacts_empty_directory(self, temp_dir: Path) -> None:
        """Test scanning empty artifacts directory."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()

        result = scan_filesystem_artifacts(artifacts_dir)

        assert result == []

    def test_scan_artifacts_with_files(self, temp_dir: Path) -> None:
        """Test scanning artifacts directory with files."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()

        # Create test artifact files
        (artifacts_dir / "archive1.zip").write_text("archive1")
        (artifacts_dir / "archive2.zip").write_text("archive2")

        result = scan_filesystem_artifacts(artifacts_dir)

        assert len(result) == 2
        assert all(p[1] > 0 for p in result)  # All have size

    def test_scan_artifacts_nonexistent_directory(self, temp_dir: Path) -> None:
        """Test scanning nonexistent directory."""
        result = scan_filesystem_artifacts(temp_dir / "nonexistent")

        assert result == []

    def test_scan_files_empty_directory(self, temp_dir: Path) -> None:
        """Test scanning empty extracted files directory."""
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        result = scan_filesystem_files(extracted_dir)

        assert result == []

    def test_scan_files_with_nested_files(self, temp_dir: Path) -> None:
        """Test scanning extracted files with nested directories."""
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        # Create nested structure
        (extracted_dir / "subdir").mkdir()
        (extracted_dir / "file1.ttl").write_text("file1")
        (extracted_dir / "subdir" / "file2.rdf").write_text("file2")

        result = scan_filesystem_files(extracted_dir)

        assert len(result) == 2


class TestDriftDetection:
    """Test drift detection between DB and filesystem."""

    def test_detect_missing_fs_file(self, conn: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
        """Test detection of missing filesystem file."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"

        # Setup DB with artifact
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, datetime.now()],
        )

        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "archive.zip", 1024, "fresh", datetime.now()],
        )

        # Create directories but don't create the file
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        issues = detect_db_fs_drifts(conn, artifacts_dir, extracted_dir)

        assert len(issues) == 1
        assert issues[0].issue_type == "missing_fs_file"
        assert issues[0].artifact_id == a_id
        assert issues[0].severity == "error"

    def test_no_drifts_when_consistent(
        self, conn: duckdb.DuckDBPyConnection, temp_dir: Path
    ) -> None:
        """Test no drifts detected when DB and FS are consistent."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"

        # Setup DB
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, datetime.now()],
        )

        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        # Create the artifact file that DB expects
        (artifacts_dir / "archive.zip").write_text("content")

        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "archive.zip", 7, "fresh", datetime.now()],
        )

        issues = detect_db_fs_drifts(conn, artifacts_dir, extracted_dir)

        # No missing file issue (the file exists now)
        assert len(issues) == 0


class TestDoctorReport:
    """Test doctor report generation."""

    def test_generate_report_empty_database(
        self, conn: duckdb.DuckDBPyConnection, temp_dir: Path
    ) -> None:
        """Test report generation on empty database."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        report = generate_doctor_report(conn, artifacts_dir, extracted_dir)

        assert isinstance(report, DoctorReport)
        assert report.total_artifacts == 0
        assert report.total_files == 0
        assert report.issues_found == 0
        assert report.critical_issues == 0

    def test_report_with_issues(self, conn: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
        """Test report generation with detected issues."""
        v_id = f"v-{uuid4().hex[:8]}"
        a_id = f"art-{uuid4().hex[:8]}"

        # Setup DB with artifact
        conn.execute(
            "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
            [v_id, "TEST", False, datetime.now()],
        )

        conn.execute(
            """INSERT INTO artifacts
               (artifact_id, version_id, fs_relpath, size, status, downloaded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [a_id, v_id, "archive.zip", 1024, "fresh", datetime.now()],
        )

        # Create directories but no files
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        report = generate_doctor_report(conn, artifacts_dir, extracted_dir)

        assert report.total_artifacts == 1
        assert report.issues_found >= 1
        assert report.critical_issues >= 1

    def test_report_timestamps(self, conn: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
        """Test that report includes timestamp."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        extracted_dir = temp_dir / "extracted"
        extracted_dir.mkdir()

        before = datetime.now()
        report = generate_doctor_report(conn, artifacts_dir, extracted_dir)
        after = datetime.now()

        assert before <= report.timestamp <= after
