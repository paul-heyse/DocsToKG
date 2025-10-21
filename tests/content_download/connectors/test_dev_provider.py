"""
Unit Tests for Development Provider (SQLite)

Tests cover:
- Initialization and schema creation
- Register/get operations (idempotency)
- Queries by artifact and SHA-256
- Deduplication detection
- Statistics and health checks
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.catalog.connectors import (  # type: ignore[import-untyped]
    CatalogConnector,
    ProviderOperationError,
)


class TestDevelopmentProvider:
    """Tests for DevelopmentProvider (SQLite)."""

    def test_dev_provider_memory_database(self) -> None:
        """Development provider works with in-memory SQLite."""
        with CatalogConnector("development", {"db_path": ":memory:"}) as cat:
            assert cat.provider_type == "development"
            health = cat.health_check()
            assert health.status.value == "healthy"

    def test_dev_provider_file_database(self) -> None:
        """Development provider works with file-based SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.sqlite")
            with CatalogConnector("development", {"db_path": db_path}) as cat:
                # Check file was created
                assert Path(db_path).exists()

    def test_dev_provider_not_opened_raises_error(self) -> None:
        """Operations on unopened provider raise error."""
        connector = CatalogConnector("development", {})
        with pytest.raises(RuntimeError, match="Connector not opened"):
            connector.register_or_get("test:001", "http://example.com", "test")

    def test_dev_provider_registers_record(self) -> None:
        """Record can be registered and retrieved."""
        with CatalogConnector("development", {}) as cat:
            record = cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com",
                resolver="test",
                content_type="application/pdf",
                bytes=1000,
                sha256="abc123def456",
                storage_uri="file:///tmp/test.pdf",
                run_id="run-001",
            )

            assert record.id > 0
            assert record.artifact_id == "test:001"
            assert record.source_url == "http://example.com"
            assert record.resolver == "test"
            assert record.bytes == 1000

    def test_dev_provider_idempotent_register(self) -> None:
        """register_or_get is idempotent - returns same record on duplicate."""
        with CatalogConnector("development", {}) as cat:
            # Register first time
            record1 = cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com",
                resolver="test",
                content_type="application/pdf",
                bytes=1000,
                sha256="abc123",
                storage_uri="file:///tmp/test.pdf",
            )

            # Register same artifact/url/resolver again
            record2 = cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com",
                resolver="test",
                content_type="application/pdf",
                bytes=1000,
                sha256="abc123",
                storage_uri="file:///tmp/test.pdf",
            )

            # Should be same record
            assert record1.id == record2.id
            assert record1.artifact_id == record2.artifact_id

    def test_dev_provider_get_by_artifact(self) -> None:
        """Can query records by artifact ID."""
        with CatalogConnector("development", {}) as cat:
            # Register multiple records for same artifact
            cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com/1",
                resolver="resolver1",
                bytes=100,
                storage_uri="file:///tmp/1.pdf",
            )
            cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com/2",
                resolver="resolver2",
                bytes=200,
                storage_uri="file:///tmp/2.pdf",
            )

            # Query by artifact
            records = cat.get_by_artifact("test:001")
            assert len(records) == 2
            assert all(r.artifact_id == "test:001" for r in records)

    def test_dev_provider_get_by_sha256(self) -> None:
        """Can query records by SHA-256 hash."""
        with CatalogConnector("development", {}) as cat:
            sha = "abc123def456"

            # Register multiple records with same SHA
            cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com/1",
                resolver="test",
                sha256=sha,
                bytes=100,
                storage_uri="file:///tmp/1.pdf",
            )
            cat.register_or_get(
                artifact_id="test:002",
                source_url="http://example.com/2",
                resolver="test",
                sha256=sha,
                bytes=100,
                storage_uri="file:///tmp/2.pdf",
            )

            # Query by SHA
            records = cat.get_by_sha256(sha)
            assert len(records) == 2
            assert all(r.sha256 == sha for r in records)

    def test_dev_provider_find_duplicates(self) -> None:
        """Can find SHA-256 hashes with multiple records."""
        with CatalogConnector("development", {}) as cat:
            sha1 = "abc123"
            sha2 = "def456"

            # Register duplicates
            cat.register_or_get(
                artifact_id="a:1",
                source_url="http://example.com/1",
                resolver="test",
                sha256=sha1,
                bytes=100,
                storage_uri="file:///tmp/1.pdf",
            )
            cat.register_or_get(
                artifact_id="a:2",
                source_url="http://example.com/2",
                resolver="test",
                sha256=sha1,
                bytes=100,
                storage_uri="file:///tmp/2.pdf",
            )

            # Register non-duplicate
            cat.register_or_get(
                artifact_id="a:3",
                source_url="http://example.com/3",
                resolver="test",
                sha256=sha2,
                bytes=200,
                storage_uri="file:///tmp/3.pdf",
            )

            # Find duplicates
            duplicates = cat.find_duplicates()
            assert len(duplicates) == 1
            assert duplicates[0] == (sha1, 2)

    def test_dev_provider_stats(self) -> None:
        """Can retrieve catalog statistics."""
        with CatalogConnector("development", {}) as cat:
            # Register some records
            cat.register_or_get(
                artifact_id="a:1",
                source_url="http://example.com/1",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=1000,
                sha256="sha1",
                storage_uri="file:///tmp/1.pdf",
            )
            cat.register_or_get(
                artifact_id="a:2",
                source_url="http://example.com/2",
                resolver="resolver2",
                content_type="text/html",
                bytes=2000,
                sha256="sha2",
                storage_uri="file:///tmp/2.html",
            )

            # Get stats
            stats = cat.stats()
            assert stats["total_records"] == 2
            assert stats["total_bytes"] == 3000
            assert stats["unique_sha256"] == 2
            assert stats["duplicates"] == 0
            assert "resolver1" in stats["by_resolver"]
            assert "resolver2" in stats["by_resolver"]

    def test_dev_provider_health_check(self) -> None:
        """Can perform health check."""
        with CatalogConnector("development", {}) as cat:
            health = cat.health_check()
            assert health.status.value == "healthy"
            assert health.latency_ms >= 0
            assert health.message == "Development provider OK"

    def test_dev_provider_verify_returns_true_for_no_hash(self) -> None:
        """Verify returns true when record has no SHA-256 hash."""
        with CatalogConnector("development", {}) as cat:
            record = cat.register_or_get(
                artifact_id="test:001",
                source_url="http://example.com",
                resolver="test",
                sha256=None,  # No hash
                storage_uri="file:///tmp/test.pdf",
            )

            result = cat.verify(record.id)
            assert result is True

    def test_dev_provider_context_manager(self) -> None:
        """Provider can be used with context manager."""
        connector = CatalogConnector("development", {})
        
        # Initially not opened
        with pytest.raises(RuntimeError):
            connector.stats()
        
        # Open with context manager
        with connector as cat:
            stats = cat.stats()
            assert isinstance(stats, dict)
        
        # After exiting, connection is closed
        with pytest.raises(RuntimeError):
            connector.stats()

    def test_dev_provider_concurrent_register(self) -> None:
        """Multiple registrations are thread-safe."""
        import threading

        with CatalogConnector("development", {}) as cat:
            results = []

            def register_record(idx: int) -> None:
                record = cat.register_or_get(
                    artifact_id=f"test:{idx}",
                    source_url=f"http://example.com/{idx}",
                    resolver="test",
                    bytes=100 * idx,
                    storage_uri=f"file:///tmp/{idx}.pdf",
                )
                results.append(record.id)

            # Register from multiple threads
            threads = [threading.Thread(target=register_record, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All should be registered
            assert len(results) == 5
            assert len(set(results)) == 5  # All unique IDs

    def test_dev_provider_wal_mode(self) -> None:
        """WAL mode is enabled for better concurrency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.sqlite")
            with CatalogConnector("development", {
                "db_path": db_path,
                "enable_wal": True
            }) as cat:
                # WAL files should exist
                wal_file = Path(db_path + "-wal")
                assert wal_file.exists()
