"""Tests for garbage collection and retention.

Tests orphan finding, retention filtering, and safe deletion.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.catalog.gc import (
    collect_referenced_paths,
    delete_orphan_files,
    find_orphans,
    retention_filter,
)
from DocsToKG.ContentDownload.catalog.models import DocumentRecord
from DocsToKG.ContentDownload.catalog.store import SQLiteCatalog


class TestFindOrphans:
    """Test orphan file detection."""
    
    def test_find_orphans_no_files(self):
        """Test finding orphans in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orphans = find_orphans(tmpdir, set())
            
            assert len(orphans) == 0
    
    def test_find_orphans_no_refs(self):
        """Test all files are orphans when no refs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.pdf").write_bytes(b"content1")
            (Path(tmpdir) / "file2.pdf").write_bytes(b"content2")
            
            orphans = find_orphans(tmpdir, set())
            
            assert len(orphans) == 2
    
    def test_find_orphans_with_refs(self):
        """Test finding orphans with some referenced files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.pdf"
            file2 = Path(tmpdir) / "file2.pdf"
            
            file1.write_bytes(b"content1")
            file2.write_bytes(b"content2")
            
            # Reference only file1
            refs = {str(file1.resolve())}
            
            orphans = find_orphans(tmpdir, refs)
            
            assert len(orphans) == 1
            assert str(file2.resolve()) in orphans
    
    def test_find_orphans_nested_dirs(self):
        """Test finding orphans in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subdir = root / "subdir"
            subdir.mkdir()
            
            (subdir / "file1.pdf").write_bytes(b"content")
            (root / "file2.pdf").write_bytes(b"content")
            
            orphans = find_orphans(tmpdir, set())
            
            assert len(orphans) == 2


class TestRetentionFilter:
    """Test retention filtering."""
    
    def test_retention_filter_disabled(self):
        """Test retention filtering disabled (0 days)."""
        now = datetime.utcnow()
        records = [
            DocumentRecord(
                id=1,
                artifact_id="doi:1",
                source_url="url1",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=1000,
                sha256="hash1",
                storage_uri="file:///data/1.pdf",
                created_at=now - timedelta(days=10),
                updated_at=now,
                run_id="run1",
            ),
        ]
        
        filtered = retention_filter(records, retention_days=0)
        
        assert len(filtered) == 0
    
    def test_retention_filter_no_expired(self):
        """Test retention filter with no expired records."""
        now = datetime.utcnow()
        records = [
            DocumentRecord(
                id=1,
                artifact_id="doi:1",
                source_url="url1",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=1000,
                sha256="hash1",
                storage_uri="file:///data/1.pdf",
                created_at=now - timedelta(days=5),
                updated_at=now,
                run_id="run1",
            ),
        ]
        
        filtered = retention_filter(records, retention_days=30)
        
        assert len(filtered) == 0
    
    def test_retention_filter_with_expired(self):
        """Test retention filter finds expired records."""
        now = datetime.utcnow()
        old = now - timedelta(days=40)
        
        records = [
            DocumentRecord(
                id=1,
                artifact_id="doi:1",
                source_url="url1",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=1000,
                sha256="hash1",
                storage_uri="file:///data/1.pdf",
                created_at=old,
                updated_at=now,
                run_id="run1",
            ),
        ]
        
        filtered = retention_filter(records, retention_days=30)
        
        assert len(filtered) == 1


class TestCollectReferencedPaths:
    """Test reference path collection."""
    
    def test_collect_file_uris(self):
        """Test collecting file:// URIs."""
        records = [
            DocumentRecord(
                id=1,
                artifact_id="doi:1",
                source_url="url1",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=1000,
                sha256="hash1",
                storage_uri="file:///data/file1.pdf",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                run_id="run1",
            ),
            DocumentRecord(
                id=2,
                artifact_id="doi:2",
                source_url="url2",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=2000,
                sha256="hash2",
                storage_uri="file:///data/file2.pdf",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                run_id="run1",
            ),
        ]
        
        paths = collect_referenced_paths(records)
        
        assert len(paths) == 2
    
    def test_collect_skips_s3(self):
        """Test that S3 URIs are skipped."""
        records = [
            DocumentRecord(
                id=1,
                artifact_id="doi:1",
                source_url="url1",
                resolver="resolver1",
                content_type="application/pdf",
                bytes=1000,
                sha256="hash1",
                storage_uri="s3://bucket/file.pdf",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                run_id="run1",
            ),
        ]
        
        paths = collect_referenced_paths(records)
        
        assert len(paths) == 0


class TestDeleteOrphanFiles:
    """Test safe deletion."""
    
    def test_delete_orphans_dry_run(self):
        """Test dry-run deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "orphan1.pdf"
            file1.write_bytes(b"content")
            
            deleted = delete_orphan_files([str(file1)], dry_run=True)
            
            assert deleted == 1
            assert file1.exists()  # Not actually deleted in dry-run
    
    def test_delete_orphans_apply(self):
        """Test actual deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "orphan1.pdf"
            file1.write_bytes(b"content")
            
            deleted = delete_orphan_files([str(file1)], dry_run=False)
            
            assert deleted == 1
            assert not file1.exists()
    
    def test_delete_orphans_already_gone(self):
        """Test deletion of already-missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.pdf"
            
            deleted = delete_orphan_files([str(missing)], dry_run=False)
            
            assert deleted == 1
    
    def test_delete_orphans_partial_failure(self):
        """Test deletion continues on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "orphan1.pdf"
            file1.write_bytes(b"content")
            missing = Path(tmpdir) / "missing.pdf"
            
            deleted = delete_orphan_files([str(file1), str(missing)], dry_run=False)
            
            # Both counted as deleted (one actually, one already gone)
            assert deleted == 2
            assert not file1.exists()


class TestIntegrationGcPipeline:
    """Integration tests for GC pipeline."""
    
    def test_gc_pipeline_find_and_delete(self):
        """Test complete GC pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            file1 = Path(tmpdir) / "ref.pdf"
            file1.write_bytes(b"referenced")
            orphan = Path(tmpdir) / "orphan.pdf"
            orphan.write_bytes(b"unreferenced")
            
            # Create records for referenced file only
            records = [
                DocumentRecord(
                    id=1,
                    artifact_id="doi:1",
                    source_url="url1",
                    resolver="resolver1",
                    content_type="application/pdf",
                    bytes=1000,
                    sha256="hash1",
                    storage_uri=f"file://{file1}",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    run_id="run1",
                ),
            ]
            
            # Collect refs
            refs = collect_referenced_paths(records)
            
            # Find orphans
            orphans = find_orphans(tmpdir, refs)
            
            assert len(orphans) == 1
            
            # Delete
            deleted = delete_orphan_files(orphans, dry_run=False)
            
            assert deleted == 1
            assert file1.exists()
            assert not orphan.exists()
