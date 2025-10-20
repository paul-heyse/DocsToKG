"""Tests for the DuckDB ontology metadata catalog."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from DocsToKG.OntologyDownload.database import (
    ArtifactRow,
    Database,
    DatabaseConfiguration,
    FileRow,
    ValidationRow,
    VersionRow,
    VersionStats,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        yield db_path


@pytest.fixture
def db(temp_db_path):
    """Create and bootstrap a test database."""
    config = DatabaseConfiguration(
        db_path=temp_db_path,
        readonly=False,
        enable_locks=False,  # Disable locks in tests for simplicity
    )
    database = Database(config)
    database.bootstrap()
    yield database
    database.close()


@pytest.fixture
def readonly_db(temp_db_path):
    """Create a read-only database connection (for testing reader path)."""
    # First create and populate the DB
    config = DatabaseConfiguration(db_path=temp_db_path, readonly=False, enable_locks=False)
    database = Database(config)
    database.bootstrap()

    # Insert some test data
    database.upsert_version("v1.0", "OLS", "hash123")
    database.close()

    # Now open read-only
    readonly_config = DatabaseConfiguration(db_path=temp_db_path, readonly=True)
    readonly_db = Database(readonly_config)
    readonly_db.bootstrap()
    yield readonly_db
    readonly_db.close()


class TestDatabaseBootstrap:
    """Test database initialization and schema creation."""

    def test_bootstrap_creates_schema(self, db):
        """Verify that bootstrap creates all necessary tables."""
        # Query for schema_version table
        result = db._connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        assert result[0] > 0, "schema_version table should have migration records"

        # Verify tables exist
        tables = [
            "versions",
            "artifacts",
            "extracted_files",
            "validations",
            "latest_pointer",
        ]
        for table in tables:
            result = db._connection.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table}'"
            ).fetchone()
            assert result[0] > 0, f"Table {table} should exist"

    def test_bootstrap_applies_migrations_once(self, temp_db_path):
        """Verify that migrations are idempotent (applied once)."""
        config = DatabaseConfiguration(db_path=temp_db_path, readonly=False, enable_locks=False)

        # First bootstrap
        db1 = Database(config)
        db1.bootstrap()
        version_count_1 = db1._connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[
            0
        ]
        db1.close()

        # Second bootstrap (should be idempotent)
        db2 = Database(config)
        db2.bootstrap()
        version_count_2 = db2._connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[
            0
        ]
        db2.close()

        assert version_count_1 == version_count_2, "Migrations should be applied only once"


class TestVersions:
    """Test version management operations."""

    def test_upsert_version(self, db):
        """Verify version upsert works correctly."""
        db.upsert_version("v1.0", "OLS", "hash123")

        versions = db.list_versions()
        assert len(versions) == 1
        assert versions[0].version_id == "v1.0"
        assert versions[0].service == "OLS"
        assert versions[0].plan_hash == "hash123"

    def test_set_and_get_latest_version(self, db):
        """Verify latest version pointer works."""
        db.upsert_version("v1.0", "OLS")
        db.set_latest_version("v1.0", by="test")

        latest = db.get_latest_version()
        assert latest is not None
        assert latest.version_id == "v1.0"

    def test_list_versions_by_service(self, db):
        """Verify filtering versions by service."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_version("v2.0", "BioPortal")
        db.upsert_version("v3.0", "OLS")

        ols_versions = db.list_versions(service="OLS")
        assert len(ols_versions) == 2

        bioportal_versions = db.list_versions(service="BioPortal")
        assert len(bioportal_versions) == 1

    def test_list_versions_respects_limit(self, db):
        """Verify limit parameter works correctly."""
        for i in range(5):
            db.upsert_version(f"v{i}", "OLS")

        versions = db.list_versions(limit=2)
        assert len(versions) == 2


class TestArtifacts:
    """Test artifact management operations."""

    def test_upsert_artifact(self, db):
        """Verify artifact upsert works correctly."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            artifact_id="sha256_abc",
            version_id="v1.0",
            service="OLS",
            source_url="https://example.com/file.tar.gz",
            size_bytes=1024,
            fs_relpath="OLS/v1.0/file.tar.gz",
            status="fresh",
            etag="etag123",
        )

        artifacts = db.list_artifacts("v1.0")
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == "sha256_abc"
        assert artifacts[0].status == "fresh"

    def test_list_artifacts_by_status(self, db):
        """Verify filtering artifacts by status."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1", "v1.0", "OLS", "https://example.com/1.tar.gz", 1024, "OLS/v1.0/1.tar.gz", "fresh"
        )
        db.upsert_artifact(
            "id2",
            "v1.0",
            "OLS",
            "https://example.com/2.tar.gz",
            2048,
            "OLS/v1.0/2.tar.gz",
            "cached",
        )

        fresh = db.list_artifacts("v1.0", status="fresh")
        assert len(fresh) == 1
        assert fresh[0].artifact_id == "id1"

        cached = db.list_artifacts("v1.0", status="cached")
        assert len(cached) == 1
        assert cached[0].artifact_id == "id2"


class TestExtractedFiles:
    """Test extracted file management."""

    def test_insert_extracted_files(self, db):
        """Verify extracted file insertion works."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            1024,
            "OLS/v1.0/file.tar.gz",
            "fresh",
        )

        files = [
            FileRow(
                file_id="file_sha256_1",
                artifact_id="id1",
                version_id="v1.0",
                relpath_in_version="data/ontology.ttl",
                format="ttl",
                size_bytes=512,
            ),
            FileRow(
                file_id="file_sha256_2",
                artifact_id="id1",
                version_id="v1.0",
                relpath_in_version="data/ontology.rdf",
                format="rdf",
                size_bytes=600,
            ),
        ]
        db.insert_extracted_files(files)

        extracted = db.list_extracted_files("v1.0")
        assert len(extracted) == 2
        assert extracted[0].format in {"ttl", "rdf"}

    def test_list_extracted_files_by_format(self, db):
        """Verify filtering extracted files by format."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            1024,
            "OLS/v1.0/file.tar.gz",
            "fresh",
        )

        files = [
            FileRow(
                "fid1",
                "id1",
                "v1.0",
                "data/1.ttl",
                "ttl",
                512,
            ),
            FileRow(
                "fid2",
                "id1",
                "v1.0",
                "data/2.rdf",
                "rdf",
                600,
            ),
            FileRow(
                "fid3",
                "id1",
                "v1.0",
                "data/3.ttl",
                "ttl",
                400,
            ),
        ]
        db.insert_extracted_files(files)

        ttl_files = db.list_extracted_files("v1.0", format_filter="ttl")
        assert len(ttl_files) == 2
        for f in ttl_files:
            assert f.format == "ttl"


class TestValidations:
    """Test validation recording."""

    def test_insert_validations(self, db):
        """Verify validation insertion works."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            1024,
            "OLS/v1.0/file.tar.gz",
            "fresh",
        )
        db.insert_extracted_files([FileRow("fid1", "id1", "v1.0", "data/1.ttl", "ttl", 512)])

        validations = [
            ValidationRow(
                validation_id="vid1",
                file_id="fid1",
                validator="rdflib",
                passed=True,
                run_at=datetime.now(timezone.utc),
                details_json={"message": "Valid RDF"},
            )
        ]
        db.insert_validations(validations)

        failures = db.get_validation_failures("v1.0")
        assert len(failures) == 0

    def test_get_validation_failures(self, db):
        """Verify failure retrieval works."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            1024,
            "OLS/v1.0/file.tar.gz",
            "fresh",
        )
        db.insert_extracted_files([FileRow("fid1", "id1", "v1.0", "data/1.ttl", "ttl", 512)])

        validations = [
            ValidationRow(
                validation_id="vid1",
                file_id="fid1",
                validator="rdflib",
                passed=False,
                run_at=datetime.now(timezone.utc),
                details_json={"error": "Invalid namespace"},
            ),
            ValidationRow(
                validation_id="vid2",
                file_id="fid1",
                validator="owlready2",
                passed=True,
                run_at=datetime.now(timezone.utc),
            ),
        ]
        db.insert_validations(validations)

        failures = db.get_validation_failures("v1.0")
        assert len(failures) == 1
        assert failures[0].validator == "rdflib"
        assert failures[0].passed is False


class TestStatistics:
    """Test statistics and reporting."""

    def test_get_version_stats(self, db):
        """Verify version statistics computation."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            2048,
            "OLS/v1.0/file.tar.gz",
            "fresh",
        )

        files = [
            FileRow("fid1", "id1", "v1.0", "data/1.ttl", "ttl", 512),
            FileRow("fid2", "id1", "v1.0", "data/2.rdf", "rdf", 600),
        ]
        db.insert_extracted_files(files)

        validations = [
            ValidationRow("vid1", "fid1", "rdflib", True, datetime.now(timezone.utc)),
            ValidationRow("vid2", "fid2", "rdflib", False, datetime.now(timezone.utc)),
        ]
        db.insert_validations(validations)

        stats = db.get_version_stats("v1.0")
        assert stats is not None
        assert stats.version_id == "v1.0"
        assert stats.files == 2
        assert stats.bytes == 1112  # 512 + 600
        assert stats.validations_passed == 1
        assert stats.validations_failed == 1


class TestTransactions:
    """Test transaction support."""

    def test_transaction_commits(self, db):
        """Verify successful transaction commits."""
        with db.transaction():
            db.upsert_version("v1.0", "OLS")
            db.set_latest_version("v1.0")

        latest = db.get_latest_version()
        assert latest is not None
        assert latest.version_id == "v1.0"

    def test_transaction_rollback_on_error(self, db):
        """Verify rollback on transaction error."""
        try:
            with db.transaction():
                db.upsert_version("v1.0", "OLS")
                # Simulate an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Version should not be saved due to rollback
        versions = db.list_versions()
        assert len(versions) == 0

    def test_readonly_transaction_raises(self, readonly_db):
        """Verify read-only mode prevents writes."""
        with pytest.raises(Exception):
            with readonly_db.transaction():
                readonly_db.upsert_version("v2.0", "OLS")


class TestContextManager:
    """Test context manager interface."""

    def test_context_manager_bootstrap_and_close(self, temp_db_path):
        """Verify context manager handles bootstrap and close."""
        config = DatabaseConfiguration(db_path=temp_db_path, readonly=False, enable_locks=False)

        with Database(config) as db:
            assert db._connection is not None
            db.upsert_version("v1.0", "OLS")

        # Connection should be closed after exiting context
        assert db._connection is None


class TestIdempotence:
    """Test idempotent operations."""

    def test_upsert_version_idempotent(self, db):
        """Verify version upsert is idempotent."""
        db.upsert_version("v1.0", "OLS", "hash1")
        db.upsert_version("v1.0", "OLS", "hash2")

        versions = db.list_versions()
        assert len(versions) == 1
        assert versions[0].plan_hash == "hash2"

    def test_upsert_artifact_idempotent(self, db):
        """Verify artifact upsert is idempotent."""
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            1024,
            "OLS/v1.0/file.tar.gz",
            "fresh",
        )
        db.upsert_artifact(
            "id1",
            "v1.0",
            "OLS",
            "https://example.com/file.tar.gz",
            2048,
            "OLS/v1.0/file.tar.gz",
            "cached",
        )

        artifacts = db.list_artifacts("v1.0")
        assert len(artifacts) == 1
        assert artifacts[0].size_bytes == 2048
        assert artifacts[0].status == "cached"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
