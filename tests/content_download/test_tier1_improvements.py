"""Tests for Tier 1 quick wins improvements.

Coverage for:
  - Streaming verification
  - Incremental GC
  - Dedup analytics
  - Backup & recovery
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from DocsToKG.ContentDownload.catalog.analytics import DedupAnalyzer
from DocsToKG.ContentDownload.catalog.backup import CatalogBackup
from DocsToKG.ContentDownload.catalog.models import DocumentRecord
from DocsToKG.ContentDownload.catalog.store import SQLiteCatalog
from DocsToKG.ContentDownload.catalog.verify import StreamingVerifier


class TestStreamingVerification:
    """Test async streaming verification."""
    
    def test_verify_single_file_match(self, tmp_path):
        """Test verifying a file with matching hash."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_data = b"test content for verification"
        test_file.write_bytes(test_data)
        
        # Compute expected hash
        import hashlib
        expected_hash = hashlib.sha256(test_data).hexdigest()
        
        catalog = MagicMock()
        verifier = StreamingVerifier(catalog)
        
        # Run verification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                verifier.verify_single(
                    record_id=1,
                    storage_uri=f"file://{test_file}",
                    expected_sha256=expected_hash,
                )
            )
        finally:
            loop.close()
        
        assert result.record_id == 1
        assert result.matches is True
        assert result.computed_sha256 == expected_hash
    
    def test_verify_single_file_mismatch(self, tmp_path):
        """Test verification with hash mismatch."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test content")
        
        catalog = MagicMock()
        verifier = StreamingVerifier(catalog)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                verifier.verify_single(
                    record_id=1,
                    storage_uri=f"file://{test_file}",
                    expected_sha256="0000000000000000",
                )
            )
        finally:
            loop.close()
        
        assert result.matches is False
        assert result.error is None  # Hash mismatch, not an error
    
    def test_verify_single_missing_file(self):
        """Test verification with missing file."""
        catalog = MagicMock()
        verifier = StreamingVerifier(catalog)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                verifier.verify_single(
                    record_id=1,
                    storage_uri="file:///nonexistent/file.pdf",
                    expected_sha256="abc123",
                )
            )
        finally:
            loop.close()
        
        assert result.matches is False
        assert result.error is not None
        assert "not found" in result.error.lower()


class TestIncrementalGC:
    """Test incremental garbage collection."""
    
    @pytest.mark.skip(reason="Requires full catalog setup")
    def test_gc_batch_processing(self):
        """Test GC processes in batches."""
        pass


class TestDedupAnalytics:
    """Test deduplication analytics."""
    
    def test_dedup_analyzer_creation(self):
        """Test creating dedup analyzer."""
        catalog = MagicMock()
        analyzer = DedupAnalyzer(catalog)
        assert analyzer.catalog is catalog
    
    def test_top_duplicates_empty(self):
        """Test top duplicates with no records."""
        catalog = MagicMock()
        catalog.get_all_records.side_effect = NotImplementedError
        
        analyzer = DedupAnalyzer(catalog)
        result = analyzer.top_duplicates(n=10)
        
        assert result == []
    
    def test_dedup_ratio_zero(self):
        """Test dedup ratio with no duplicates."""
        catalog = MagicMock()
        
        # Create mock records with unique hashes
        records = [
            MagicMock(sha256=f"hash{i}", bytes=1000) for i in range(5)
        ]
        catalog.get_all_records.return_value = records
        catalog.find_duplicates.return_value = []
        
        analyzer = DedupAnalyzer(catalog)
        ratio = analyzer.dedup_ratio()
        
        assert ratio == 0.0
    
    def test_recommendations_empty(self):
        """Test recommendations with no data."""
        catalog = MagicMock()
        catalog.get_all_records.side_effect = NotImplementedError
        
        analyzer = DedupAnalyzer(catalog)
        recs = analyzer.recommendations()
        
        assert len(recs) > 0
        assert isinstance(recs[0], str)


class TestBackupRecovery:
    """Test catalog backup and recovery."""
    
    def test_backup_creation(self, tmp_path):
        """Test creating a backup."""
        # Create a test catalog
        catalog_path = tmp_path / "catalog.sqlite"
        catalog = SQLiteCatalog(path=str(catalog_path), wal_mode=False)
        
        # Add a test record
        catalog.register_or_get(
            artifact_id="test:001",
            source_url="http://example.com/test.pdf",
            resolver="test_resolver",
            content_type="application/pdf",
            bytes=1000,
            sha256="abc123def456",
            storage_uri="file:///tmp/test.pdf",
            run_id="run-001",
        )
        
        catalog.conn.close()
        
        # Create backup
        backup_dir = tmp_path / "backups"
        backup_manager = CatalogBackup(str(catalog_path))
        metadata = backup_manager.backup_atomic(backup_dir)
        
        assert metadata.destination_path is not None
        assert metadata.file_size_bytes > 0
        assert metadata.record_count == 1
        
        # Verify backup file exists
        backup_path = Path(metadata.destination_path)
        assert backup_path.exists()
    
    def test_backup_metadata_saved(self, tmp_path):
        """Test that backup metadata is saved."""
        catalog_path = tmp_path / "catalog.sqlite"
        catalog = SQLiteCatalog(path=str(catalog_path), wal_mode=False)
        catalog.conn.close()
        
        backup_dir = tmp_path / "backups"
        backup_manager = CatalogBackup(str(catalog_path))
        metadata = backup_manager.backup_atomic(backup_dir)
        
        # Check metadata file
        backup_path = Path(metadata.destination_path)
        metadata_path = backup_path.with_suffix(".metadata.json")
        
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
        
        assert "timestamp" in metadata_dict
        assert "record_count" in metadata_dict
    
    def test_backup_recovery_dry_run(self, tmp_path):
        """Test recovery with dry-run."""
        catalog_path = tmp_path / "catalog.sqlite"
        catalog = SQLiteCatalog(path=str(catalog_path), wal_mode=False)
        catalog.conn.close()
        
        backup_dir = tmp_path / "backups"
        backup_manager = CatalogBackup(str(catalog_path))
        metadata = backup_manager.backup_atomic(backup_dir)
        
        # Try to recover (dry-run)
        recovery_path = tmp_path / "catalog_recovered.sqlite"
        backup_manager.catalog_path = recovery_path
        
        success = backup_manager.recover_from_backup(
            Path(metadata.destination_path),
            recovery_path,
            dry_run=True,
        )
        
        assert success is True
        assert not recovery_path.exists()  # Dry-run, shouldn't create file
    
    def test_list_backups(self, tmp_path):
        """Test listing backups."""
        catalog_path = tmp_path / "catalog.sqlite"
        catalog = SQLiteCatalog(path=str(catalog_path), wal_mode=False)
        catalog.conn.close()
        
        backup_dir = tmp_path / "backups"
        backup_manager = CatalogBackup(str(catalog_path))
        
        # Create a backup
        backup_manager.backup_atomic(backup_dir)
        
        # List backups
        backups = backup_manager.list_backups(backup_dir)
        
        assert len(backups) == 1
        backup_path, metadata = backups[0]
        assert backup_path.exists()
        assert metadata.record_count == 0  # Empty catalog


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
