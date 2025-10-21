"""Tests for storage layouts and deduplication.

Tests CAS paths, policy paths, and hardlink deduplication.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from DocsToKG.ContentDownload.catalog.fs_layout import (
    cas_path,
    choose_final_path,
    dedup_hardlink_or_copy,
    extract_basename_from_url,
    policy_path,
)


class TestCasPath:
    """Test CAS path generation."""
    
    def test_cas_path_generation(self):
        """Test CAS path structure."""
        root = "/data/cas"
        sha256 = "e3b0c44298fc1c14e44ee83e48873be5"
        
        path = cas_path(root, sha256)
        
        # Should use two-level fan-out
        assert "/cas/" in path
        assert "/sha256/" in path
        assert path.endswith("e83e48873be5")
        assert "/e3/" in path
    
    def test_cas_path_invalid_hash(self):
        """Test CAS path with invalid hash."""
        with pytest.raises(ValueError):
            cas_path("/data", "abc")
    
    def test_cas_path_directory_creation(self):
        """Test CAS path creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sha256 = "a" * 64
            path = cas_path(tmpdir, sha256)
            
            # Parent directory should exist
            assert Path(path).parent.exists()


class TestPolicyPath:
    """Test policy path generation."""
    
    def test_policy_path_generation(self):
        """Test policy path structure."""
        root = "/data/docs"
        
        path = policy_path(
            root,
            artifact_id="doi:10.1234/abc",
            url_basename="paper.pdf",
        )
        
        assert path == f"{root}/paper.pdf"
    
    def test_policy_path_directory_creation(self):
        """Test policy path creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = policy_path(
                tmpdir,
                artifact_id="doi:test",
                url_basename="papers/paper.pdf",
            )
            
            # Parent directory should exist
            assert Path(path).parent.exists()


class TestChooseFinalPath:
    """Test layout strategy selection."""
    
    def test_choose_cas_layout(self):
        """Test choosing CAS layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = choose_final_path(
                root_dir=tmpdir,
                layout="cas",
                sha256_hex="abc" * 21 + "abc",  # 64 chars
                artifact_id="doi:test",
                url_basename="paper.pdf",
            )
            
            assert "/cas/" in path
    
    def test_choose_policy_layout(self):
        """Test choosing policy path layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = choose_final_path(
                root_dir=tmpdir,
                layout="policy_path",
                sha256_hex=None,
                artifact_id="doi:test",
                url_basename="paper.pdf",
            )
            
            assert path.endswith("paper.pdf")
    
    def test_cas_requires_sha256(self):
        """Test that CAS layout requires SHA-256."""
        with pytest.raises(ValueError):
            choose_final_path(
                root_dir="/data",
                layout="cas",
                sha256_hex=None,
                artifact_id="doi:test",
                url_basename="paper.pdf",
            )
    
    def test_invalid_layout(self):
        """Test invalid layout."""
        with pytest.raises(ValueError):
            choose_final_path(
                root_dir="/data",
                layout="invalid_layout",
                sha256_hex="abc123",
                artifact_id="doi:test",
                url_basename="paper.pdf",
            )


class TestDedupHardlinkOrCopy:
    """Test deduplication logic."""
    
    def test_new_file_no_dedup(self):
        """Test new file (no dedup)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.pdf"
            dst = Path(tmpdir) / "dest.pdf"
            
            # Create source file
            src.write_bytes(b"test content")
            
            # Perform dedup
            is_dedup = dedup_hardlink_or_copy(str(src), str(dst))
            
            assert is_dedup is False
            assert dst.exists()
            assert not src.exists()  # Moved
    
    def test_dedup_hardlink_success(self):
        """Test successful hardlink dedup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            dst = Path(tmpdir) / "final.pdf"
            dst.write_bytes(b"duplicated content")
            
            # Create temp file with same content
            src = Path(tmpdir) / "temp.pdf"
            src.write_bytes(b"duplicated content")
            
            # Perform dedup
            is_dedup = dedup_hardlink_or_copy(str(src), str(dst), hardlink=True)
            
            assert is_dedup is True
            assert not src.exists()  # Temp removed
            assert dst.exists()
    
    def test_dedup_fallback_to_copy(self):
        """Test fallback to copy on cross-filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            dst = Path(tmpdir) / "final.pdf"
            dst.write_bytes(b"content")
            
            # Create temp file
            src = Path(tmpdir) / "temp.pdf"
            src.write_bytes(b"content")
            
            # Try hardlink dedup (may fail on some systems)
            is_dedup = dedup_hardlink_or_copy(str(src), str(dst), hardlink=True)
            
            # Either hardlink succeeded or fallback to copy succeeded
            assert is_dedup is True
            assert not src.exists()
            assert dst.exists()
    
    def test_dedup_disabled_uses_copy(self):
        """Test copy when hardlink dedup disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "final.pdf"
            dst.write_bytes(b"content")
            
            src = Path(tmpdir) / "temp.pdf"
            src.write_bytes(b"content")
            
            is_dedup = dedup_hardlink_or_copy(str(src), str(dst), hardlink=False)
            
            assert is_dedup is True
            assert not src.exists()
            assert dst.exists()


class TestExtractBasenameFromUrl:
    """Test URL basename extraction."""
    
    def test_extract_pdf_basename(self):
        """Test extracting PDF basename."""
        url = "https://example.com/papers/2023/important_paper.pdf"
        
        basename = extract_basename_from_url(url)
        
        assert basename == "important_paper.pdf"
    
    def test_extract_with_query_params(self):
        """Test extracting basename with query params."""
        url = "https://example.com/download?id=123&format=pdf&token=xyz"
        
        basename = extract_basename_from_url(url)
        
        # Should use MD5 fallback
        assert "artifact_" in basename
        assert basename.endswith(".bin")
    
    def test_extract_root_path(self):
        """Test extracting from root URL."""
        url = "https://example.com/"
        
        basename = extract_basename_from_url(url)
        
        # Should use MD5 fallback
        assert "artifact_" in basename
    
    def test_extract_malformed_url(self):
        """Test extracting from malformed URL."""
        url = "not a real url"
        
        basename = extract_basename_from_url(url)
        
        # Should use MD5 fallback
        assert "artifact_" in basename


class TestIntegrationLayoutPipeline:
    """Integration tests for layout pipeline."""
    
    def test_cas_pipeline_end_to_end(self):
        """Test full CAS pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temp file
            src = Path(tmpdir) / "temp.pdf"
            src.write_bytes(b"important content")
            
            sha256 = "e3b0c44298fc1c149c89ee83e48873be5"
            
            # Choose CAS path
            final = choose_final_path(
                root_dir=tmpdir,
                layout="cas",
                sha256_hex=sha256,
                artifact_id="doi:test",
                url_basename="paper.pdf",
            )
            
            # Finalize
            dedup_hardlink_or_copy(str(src), final, hardlink=True)
            
            # Verify
            assert Path(final).exists()
            assert not src.exists()
    
    def test_policy_pipeline_end_to_end(self):
        """Test full policy path pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "temp.pdf"
            src.write_bytes(b"content")
            
            # Choose policy path
            final = choose_final_path(
                root_dir=tmpdir,
                layout="policy_path",
                sha256_hex=None,
                artifact_id="doi:test",
                url_basename="papers/2023/paper.pdf",
            )
            
            # Finalize
            dedup_hardlink_or_copy(str(src), final, hardlink=False)
            
            # Verify
            assert Path(final).exists()
            assert not src.exists()
