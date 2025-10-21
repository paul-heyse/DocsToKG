"""Tests for DuckDB catalog query façades."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

try:  # pragma: no cover
    import duckdb
except ImportError:  # pragma: no cover
    pytest.skip("duckdb not installed", allow_module_level=True)

from DocsToKG.OntologyDownload.catalog.migrations import apply_migrations
from DocsToKG.OntologyDownload.catalog.queries import (
    ArtifactRow,
    FileRow,
    ValidationRow,
    VersionRow,
    get_artifact,
    get_artifact_stats,
    get_file,
    get_file_stats,
    get_latest,
    get_validation,
    get_validation_stats,
    get_version,
    list_artifacts,
    list_files,
    list_files_by_format,
    list_validations,
    list_validations_by_status,
    list_versions,
)


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        yield db_path


@pytest.fixture
def conn(temp_db: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and apply migrations."""
    connection = duckdb.connect(str(temp_db), read_only=False)
    apply_migrations(connection)
    yield connection
    connection.close()


@pytest.fixture
def populated_conn(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Populate database with test data."""
    # Insert versions
    v1_id = f"v1-{uuid4().hex[:8]}"
    v2_id = f"v2-{uuid4().hex[:8]}"

    now = datetime.now()
    conn.execute(
        "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
        [v1_id, "OLS", True, now],
    )
    time.sleep(0.01)

    conn.execute(
        "INSERT INTO versions (version_id, service, latest_pointer, ts) VALUES (?, ?, ?, ?)",
        [v2_id, "OBO", False, datetime.now()],
    )
    time.sleep(0.01)

    # Insert artifact for v1 only (size 1024)
    a1_id = f"art-{uuid4().hex[:8]}"
    conn.execute(
        """INSERT INTO artifacts
           (artifact_id, version_id, fs_relpath, size, etag, status, downloaded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [a1_id, v1_id, "/path1", 1024, "etag1", "fresh", datetime.now()],
    )
    time.sleep(0.01)

    # Insert files into a1 (artifact for v1)
    f1_id = f"file-{uuid4().hex[:8]}"
    f2_id = f"file-{uuid4().hex[:8]}"

    conn.execute(
        """INSERT INTO extracted_files
           (file_id, artifact_id, relpath, size, format, sha256, mtime, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [f1_id, a1_id, "data/file1.ttl", 512, "ttl", "sha256-1", datetime.now(), datetime.now()],
    )
    time.sleep(0.01)

    conn.execute(
        """INSERT INTO extracted_files
           (file_id, artifact_id, relpath, size, format, sha256, mtime, extracted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [f2_id, a1_id, "data/file2.rdf", 256, "rdf", "sha256-2", datetime.now(), datetime.now()],
    )
    time.sleep(0.01)

    # Insert validation for f1
    v_id = f"val-{uuid4().hex[:8]}"
    conn.execute(
        """INSERT INTO validations
           (validation_id, file_id, validator, status, details, validated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [v_id, f1_id, "rdflib", "pass", '{"checked": true}', datetime.now()],
    )

    yield conn


class TestVersionQueries:
    """Test version query façades."""

    def test_list_versions_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing versions on empty database."""
        versions = list_versions(conn)
        assert versions == []

    def test_list_versions_populated(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing versions with data."""
        versions = list_versions(populated_conn)
        assert len(versions) == 2
        assert all(isinstance(v, VersionRow) for v in versions)
        # Second version was inserted last so should be first (ORDER BY ts DESC)
        assert versions[0].service == "OBO"
        # Find the one with latest_pointer
        latest_versions = [v for v in versions if v.latest_pointer]
        assert len(latest_versions) == 1
        assert latest_versions[0].service == "OLS"

    def test_get_latest_with_pointer(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test getting latest version marked with pointer."""
        latest = get_latest(populated_conn)
        assert latest is not None
        assert latest.latest_pointer is True
        assert latest.service == "OLS"

    def test_get_latest_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test getting latest from empty database."""
        latest = get_latest(conn)
        assert latest is None

    def test_get_version_by_id(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test getting specific version by ID."""
        all_versions = list_versions(populated_conn)
        v = get_version(populated_conn, all_versions[0].version_id)
        assert v is not None
        assert v.version_id == all_versions[0].version_id


class TestArtifactQueries:
    """Test artifact query façades."""

    def test_list_artifacts_empty_version(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing artifacts for non-existent version."""
        artifacts = list_artifacts(conn, "nonexistent")
        assert artifacts == []

    def test_list_artifacts_populated(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing artifacts for a version."""
        versions = list_versions(populated_conn)
        # Get the version with artifacts (v1 has service='OLS')
        v1 = [v for v in versions if v.service == "OLS"][0]
        artifacts = list_artifacts(populated_conn, v1.version_id)
        assert len(artifacts) == 1
        assert isinstance(artifacts[0], ArtifactRow)
        assert artifacts[0].size == 1024

    def test_get_artifact_by_id(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test getting specific artifact by ID."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        artifacts = list_artifacts(populated_conn, v1.version_id)
        art = get_artifact(populated_conn, artifacts[0].artifact_id)
        assert art is not None
        assert art.artifact_id == artifacts[0].artifact_id


class TestFileQueries:
    """Test file query façades."""

    def test_list_files_empty_version(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing files for non-existent version."""
        files = list_files(conn, "nonexistent")
        assert files == []

    def test_list_files_populated(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing files for a version."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        files = list_files(populated_conn, v1.version_id)
        assert len(files) == 2
        assert all(isinstance(f, FileRow) for f in files)
        assert {f.format for f in files} == {"ttl", "rdf"}

    def test_list_files_by_format(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing files filtered by format."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        ttl_files = list_files_by_format(populated_conn, v1.version_id, "ttl")
        assert len(ttl_files) == 1
        assert ttl_files[0].format == "ttl"
        assert ttl_files[0].relpath == "data/file1.ttl"

    def test_get_file_by_id(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test getting specific file by ID."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        files = list_files(populated_conn, v1.version_id)
        file = get_file(populated_conn, files[0].file_id)
        assert file is not None
        assert file.file_id == files[0].file_id


class TestValidationQueries:
    """Test validation query façades."""

    def test_list_validations_empty_version(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing validations for non-existent version."""
        validations = list_validations(conn, "nonexistent")
        assert validations == []

    def test_list_validations_populated(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing validations for a version."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        validations = list_validations(populated_conn, v1.version_id)
        assert len(validations) == 1
        assert isinstance(validations[0], ValidationRow)
        assert validations[0].status == "pass"

    def test_list_validations_by_status(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test listing validations filtered by status."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        passed = list_validations_by_status(populated_conn, v1.version_id, "pass")
        assert len(passed) == 1
        assert passed[0].status == "pass"

        failed = list_validations_by_status(populated_conn, v1.version_id, "fail")
        assert len(failed) == 0

    def test_get_validation_by_id(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test getting specific validation by ID."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        validations = list_validations(populated_conn, v1.version_id)
        val = get_validation(populated_conn, validations[0].validation_id)
        assert val is not None
        assert val.validation_id == validations[0].validation_id


class TestStatisticsQueries:
    """Test statistics query façades."""

    def test_get_artifact_stats_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test artifact stats on empty version."""
        stats = get_artifact_stats(conn, "nonexistent")
        assert stats == {"total_artifacts": 0, "total_size": 0, "avg_size": 0}

    def test_get_artifact_stats_populated(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test artifact stats with data."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        stats = get_artifact_stats(populated_conn, v1.version_id)
        assert stats["total_artifacts"] == 1
        assert stats["total_size"] == 1024
        assert stats["avg_size"] == 1024

    def test_get_file_stats_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test file stats on empty version."""
        stats = get_file_stats(conn, "nonexistent")
        assert stats == {"total_files": 0, "total_size": 0, "avg_size": 0}

    def test_get_file_stats_populated(self, populated_conn: duckdb.DuckDBPyConnection) -> None:
        """Test file stats with data."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        stats = get_file_stats(populated_conn, v1.version_id)
        assert stats["total_files"] == 2
        assert stats["total_size"] == 768  # 512 + 256
        assert stats["avg_size"] == 384  # 768 / 2

    def test_get_validation_stats_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Test validation stats on empty version."""
        stats = get_validation_stats(conn, "nonexistent")
        assert stats == {"total_validations": 0, "passed": 0, "failed": 0, "timeout": 0}

    def test_get_validation_stats_populated(
        self, populated_conn: duckdb.DuckDBPyConnection
    ) -> None:
        """Test validation stats with data."""
        versions = list_versions(populated_conn)
        v1 = [v for v in versions if v.service == "OLS"][0]
        stats = get_validation_stats(populated_conn, v1.version_id)
        assert stats["total_validations"] == 1
        assert stats["passed"] == 1
        assert stats["failed"] == 0
        assert stats["timeout"] == 0
