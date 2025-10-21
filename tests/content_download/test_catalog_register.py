"""Tests for catalog registration and CRUD operations.

Tests idempotent insertion, lookup operations, and thread safety.
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.catalog.models import DocumentRecord
from DocsToKG.ContentDownload.catalog.store import SQLiteCatalog


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def catalog(temp_db):
    """Create a test catalog."""
    cat = SQLiteCatalog(path=temp_db, wal_mode=False)
    yield cat
    cat.close()


class TestRegisterOrGet:
    """Test idempotent registration."""
    
    def test_register_new_record(self, catalog):
        """Test registering a new record."""
        result = catalog.register_or_get(
            artifact_id="doi:10.1234/test",
            source_url="https://example.com/paper.pdf",
            resolver="test_resolver",
            content_type="application/pdf",
            bytes=12345,
            sha256="abc123def456",
            storage_uri="file:///tmp/paper.pdf",
            run_id="run-001",
        )
        
        assert isinstance(result, DocumentRecord)
        assert result.artifact_id == "doi:10.1234/test"
        assert result.resolver == "test_resolver"
        assert result.bytes == 12345
        assert result.sha256 == "abc123def456"
    
    def test_idempotent_insertion(self, catalog):
        """Test that registering twice returns same record."""
        params = {
            "artifact_id": "doi:10.1234/idempotent",
            "source_url": "https://example.com/paper.pdf",
            "resolver": "unpaywall",
            "content_type": "application/pdf",
            "bytes": 99999,
            "sha256": "idempotent123",
            "storage_uri": "file:///data/paper.pdf",
            "run_id": "run-002",
        }
        
        # First registration
        record1 = catalog.register_or_get(**params)
        
        # Second registration (same params)
        record2 = catalog.register_or_get(**params)
        
        # Should return same ID
        assert record1.id == record2.id
        assert record1.artifact_id == record2.artifact_id
    
    def test_different_resolver_same_artifact_different_record(self, catalog):
        """Test that same artifact via different resolver creates different record."""
        artifact_id = "doi:10.1234/multi"
        url = "https://example.com/paper.pdf"
        
        # Register via resolver 1
        record1 = catalog.register_or_get(
            artifact_id=artifact_id,
            source_url=url,
            resolver="resolver1",
            content_type="application/pdf",
            bytes=1000,
            sha256="hash1",
            storage_uri="file:///data/1.pdf",
            run_id="run-003",
        )
        
        # Register same artifact via resolver 2
        record2 = catalog.register_or_get(
            artifact_id=artifact_id,
            source_url=url,
            resolver="resolver2",
            content_type="application/pdf",
            bytes=1000,
            sha256="hash1",
            storage_uri="file:///data/2.pdf",
            run_id="run-003",
        )
        
        # Should be different records (different resolvers)
        assert record1.id != record2.id
        assert record1.resolver != record2.resolver


class TestGetByArtifact:
    """Test artifact lookup."""
    
    def test_get_by_artifact_single_record(self, catalog):
        """Test retrieving single record by artifact ID."""
        artifact_id = "doi:10.1234/single"
        
        catalog.register_or_get(
            artifact_id=artifact_id,
            source_url="https://example.com/paper.pdf",
            resolver="unpaywall",
            content_type="application/pdf",
            bytes=5000,
            sha256="single123",
            storage_uri="file:///data/paper.pdf",
            run_id="run-004",
        )
        
        records = catalog.get_by_artifact(artifact_id)
        
        assert len(records) == 1
        assert records[0].artifact_id == artifact_id
    
    def test_get_by_artifact_multiple_resolvers(self, catalog):
        """Test retrieving multiple records for same artifact."""
        artifact_id = "doi:10.1234/multi-resolver"
        
        for i in range(3):
            catalog.register_or_get(
                artifact_id=artifact_id,
                source_url=f"https://example.com/paper-v{i}.pdf",
                resolver=f"resolver-{i}",
                content_type="application/pdf",
                bytes=1000 * (i + 1),
                sha256=f"hash{i}",
                storage_uri=f"file:///data/paper-{i}.pdf",
                run_id="run-005",
            )
        
        records = catalog.get_by_artifact(artifact_id)
        
        assert len(records) == 3
        assert all(r.artifact_id == artifact_id for r in records)
    
    def test_get_by_artifact_nonexistent(self, catalog):
        """Test retrieving nonexistent artifact."""
        records = catalog.get_by_artifact("doi:10.1234/nonexistent")
        
        assert len(records) == 0


class TestGetBySha256:
    """Test SHA-256 lookup."""
    
    def test_get_by_sha256_single_record(self, catalog):
        """Test retrieving by SHA-256."""
        sha256 = "exactlyonehash"
        
        catalog.register_or_get(
            artifact_id="doi:10.1234/sha256-single",
            source_url="https://example.com/paper.pdf",
            resolver="unpaywall",
            content_type="application/pdf",
            bytes=1000,
            sha256=sha256,
            storage_uri="file:///data/paper.pdf",
            run_id="run-006",
        )
        
        records = catalog.get_by_sha256(sha256)
        
        assert len(records) == 1
        assert records[0].sha256 == sha256
    
    def test_get_by_sha256_multiple_artifacts(self, catalog):
        """Test retrieving multiple artifacts with same hash (dedup)."""
        sha256 = "shared_hash_123"
        
        for i in range(2):
            catalog.register_or_get(
                artifact_id=f"doi:10.1234/dedup-{i}",
                source_url=f"https://example{i}.com/paper.pdf",
                resolver=f"resolver-{i}",
                content_type="application/pdf",
                bytes=5000,
                sha256=sha256,
                storage_uri=f"file:///data/paper-{i}.pdf",
                run_id="run-007",
            )
        
        records = catalog.get_by_sha256(sha256)
        
        assert len(records) == 2
        assert all(r.sha256 == sha256 for r in records)


class TestFindDuplicates:
    """Test duplicate detection."""
    
    def test_find_duplicates_empty(self, catalog):
        """Test find_duplicates on empty catalog."""
        duplicates = catalog.find_duplicates()
        
        assert len(duplicates) == 0
    
    def test_find_duplicates_no_dups(self, catalog):
        """Test find_duplicates with unique hashes."""
        for i in range(3):
            catalog.register_or_get(
                artifact_id=f"doi:10.1234/unique-{i}",
                source_url=f"https://example.com/paper-{i}.pdf",
                resolver="unpaywall",
                content_type="application/pdf",
                bytes=1000,
                sha256=f"hash-{i}",
                storage_uri=f"file:///data/{i}.pdf",
                run_id="run-008",
            )
        
        duplicates = catalog.find_duplicates()
        
        assert len(duplicates) == 0
    
    def test_find_duplicates_with_dups(self, catalog):
        """Test finding actual duplicates."""
        shared_hash = "duplicate_content"
        
        # Register 3 records with same hash
        for i in range(3):
            catalog.register_or_get(
                artifact_id=f"doi:10.1234/dup-{i}",
                source_url=f"https://example.com/same-{i}.pdf",
                resolver="unpaywall",
                content_type="application/pdf",
                bytes=7777,
                sha256=shared_hash,
                storage_uri=f"file:///data/same-{i}.pdf",
                run_id="run-009",
            )
        
        duplicates = catalog.find_duplicates()
        
        assert len(duplicates) == 1
        assert duplicates[0][0] == shared_hash
        assert duplicates[0][1] == 3


class TestStats:
    """Test statistics."""
    
    def test_stats_empty_catalog(self, catalog):
        """Test stats on empty catalog."""
        stats = catalog.stats()
        
        assert stats["total_documents"] == 0
        assert stats["unique_hashes"] == 0
        assert stats["unique_artifacts"] == 0
        assert stats["total_bytes"] == 0
    
    def test_stats_with_records(self, catalog):
        """Test stats with multiple records."""
        # Register 2 artifacts with 3 records total
        catalog.register_or_get(
            artifact_id="doi:10.1234/stats1",
            source_url="https://example.com/1.pdf",
            resolver="resolver1",
            content_type="application/pdf",
            bytes=1000,
            sha256="hash1",
            storage_uri="file:///data/1.pdf",
            run_id="run-010",
        )
        
        catalog.register_or_get(
            artifact_id="doi:10.1234/stats1",
            source_url="https://example.com/1b.pdf",
            resolver="resolver2",
            content_type="application/pdf",
            bytes=1000,
            sha256="hash1",
            storage_uri="file:///data/1b.pdf",
            run_id="run-010",
        )
        
        catalog.register_or_get(
            artifact_id="doi:10.1234/stats2",
            source_url="https://example.com/2.pdf",
            resolver="resolver1",
            content_type="application/pdf",
            bytes=2000,
            sha256="hash2",
            storage_uri="file:///data/2.pdf",
            run_id="run-010",
        )
        
        stats = catalog.stats()
        
        assert stats["total_documents"] == 3
        assert stats["unique_hashes"] == 2
        assert stats["unique_artifacts"] == 2
        assert stats["total_bytes"] == 4000


class TestThreadSafety:
    """Test concurrent access."""
    
    def test_concurrent_registration(self, catalog):
        """Test concurrent registrations."""
        errors = []
        
        def register_record(i):
            try:
                catalog.register_or_get(
                    artifact_id=f"doi:10.1234/concurrent-{i % 5}",
                    source_url=f"https://example.com/{i}.pdf",
                    resolver=f"resolver-{i % 3}",
                    content_type="application/pdf",
                    bytes=i * 100,
                    sha256=f"hash-{i}",
                    storage_uri=f"file:///data/{i}.pdf",
                    run_id=f"run-{i}",
                )
            except Exception as e:
                errors.append(e)
        
        # Spawn 10 concurrent threads
        threads = [threading.Thread(target=register_record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        stats = catalog.stats()
        assert stats["total_documents"] == 10
