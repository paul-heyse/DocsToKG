"""Unit tests for LocalDuckDBStorage.

Tests cover:
- Basic operations (put_file, put_bytes, delete, exists, stat, list)
- Atomicity guarantees
- Error handling
- Path safety
- Integration with DuckDB
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from DocsToKG.OntologyDownload.storage.base import StoredObject, StoredStat
from DocsToKG.OntologyDownload.storage.localfs_duckdb import LocalDuckDBStorage


@dataclass(frozen=True)
class MockDuckDBConfig:
    """Mock DuckDB configuration for testing."""

    path: Path
    threads: int = 4
    readonly: bool = False
    writer_lock: bool = True


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_duckdb_config(temp_storage):
    """Create mock DuckDB config."""
    return MockDuckDBConfig(
        path=temp_storage / "test.duckdb",
        threads=4,
        readonly=False,
        writer_lock=True,
    )


@pytest.fixture
def storage(temp_storage, mock_duckdb_config):
    """Create LocalDuckDBStorage instance."""
    return LocalDuckDBStorage(
        root=temp_storage,
        db=mock_duckdb_config,
        latest_name="LATEST.json",
        write_latest_mirror=True,
    )


class TestBasicOperations:
    """Test basic storage operations."""

    def test_put_file(self, storage, temp_storage):
        """Test putting a file into storage."""
        # Create source file
        src = temp_storage / "source.txt"
        src.write_text("test content")

        # Put file
        result = storage.put_file(src, "data/file.txt")

        # Verify result
        assert result.path_rel == "data/file.txt"
        assert result.size == 12
        assert result.url.endswith("data/file.txt")

        # Verify file exists in storage
        stored_path = temp_storage / "data" / "file.txt"
        assert stored_path.exists()
        assert stored_path.read_text() == "test content"

    def test_put_bytes(self, storage, temp_storage):
        """Test putting bytes into storage."""
        data = b"test data"
        result = storage.put_bytes(data, "data/bytes.dat")

        assert result.path_rel == "data/bytes.dat"
        assert result.size == 9

        stored_path = temp_storage / "data" / "bytes.dat"
        assert stored_path.read_bytes() == data

    def test_delete_single_file(self, storage, temp_storage):
        """Test deleting a single file."""
        # Create file
        storage.put_bytes(b"data", "file.txt")

        # Delete it
        storage.delete("file.txt")

        # Verify it's gone
        assert not (temp_storage / "file.txt").exists()

    def test_delete_multiple_files(self, storage, temp_storage):
        """Test deleting multiple files."""
        # Create files
        storage.put_bytes(b"data1", "file1.txt")
        storage.put_bytes(b"data2", "file2.txt")

        # Delete both
        storage.delete(["file1.txt", "file2.txt"])

        # Verify both are gone
        assert not (temp_storage / "file1.txt").exists()
        assert not (temp_storage / "file2.txt").exists()

    def test_delete_missing_file(self, storage):
        """Test deleting missing file (should not raise)."""
        # Should not raise
        storage.delete("nonexistent.txt")

    def test_exists(self, storage):
        """Test exists check."""
        storage.put_bytes(b"data", "file.txt")

        assert storage.exists("file.txt")
        assert not storage.exists("missing.txt")

    def test_stat(self, storage, temp_storage):
        """Test stat operation."""
        storage.put_bytes(b"12345", "file.dat")

        stat = storage.stat("file.dat")

        assert stat.size == 5
        assert stat.etag is None
        assert stat.last_modified is not None

    def test_stat_missing_file(self, storage):
        """Test stat on missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            storage.stat("missing.txt")

    def test_list_empty(self, storage):
        """Test listing empty storage."""
        result = storage.list()
        assert result == []

    def test_list_all_files(self, storage):
        """Test listing all files."""
        storage.put_bytes(b"data1", "file1.txt")
        storage.put_bytes(b"data2", "subdir/file2.txt")

        result = storage.list()

        assert len(result) == 2
        assert "file1.txt" in result
        assert "subdir/file2.txt" in result

    def test_list_with_prefix(self, storage):
        """Test listing with prefix filter."""
        storage.put_bytes(b"data1", "subdir/file1.txt")
        storage.put_bytes(b"data2", "subdir/file2.txt")
        storage.put_bytes(b"data3", "other/file3.txt")

        result = storage.list("subdir")

        assert len(result) == 2
        assert all("subdir" in p for p in result)

    def test_resolve_url(self, storage, temp_storage):
        """Test resolving URL."""
        url = storage.resolve_url("data/file.txt")

        assert str(temp_storage) in url
        assert "data/file.txt" in url

    def test_base_url(self, storage, temp_storage):
        """Test getting base URL."""
        base = storage.base_url()

        assert base == str(temp_storage.resolve())


class TestAtomicity:
    """Test atomicity guarantees."""

    def test_put_file_atomic(self, storage, temp_storage):
        """Test put_file is atomic (no partial files)."""
        src = temp_storage / "source.txt"
        src.write_text("x" * 1000)

        storage.put_file(src, "file.txt")

        dest = temp_storage / "file.txt"
        assert dest.exists()
        assert dest.read_text() == "x" * 1000

    def test_put_file_cleanup_on_error(self, storage, temp_storage):
        """Test cleanup of temp file on error."""
        # Create a non-existent source to trigger error
        with pytest.raises(FileNotFoundError):
            storage.put_file(temp_storage / "nonexistent.txt", "dest.txt")

        # Verify no tmp files left
        tmp_files = [
            f for f in temp_storage.glob("**/*") if ".tmp-" in f.name
        ]
        assert len(tmp_files) == 0

    def test_put_bytes_atomic(self, storage, temp_storage):
        """Test put_bytes is atomic."""
        data = b"x" * 1000
        storage.put_bytes(data, "file.dat")

        dest = temp_storage / "file.dat"
        assert dest.read_bytes() == data

    def test_put_bytes_cleanup_on_error(self, storage, temp_storage):
        """Test cleanup on write error."""
        # Use mock to simulate write error
        with patch("builtins.open", side_effect=OSError("Disk error")):
            with pytest.raises(OSError):
                storage.put_bytes(b"data", "file.txt")

        # Verify no tmp files left
        tmp_files = [
            f for f in temp_storage.glob("**/*") if ".tmp-" in f.name
        ]
        assert len(tmp_files) == 0

    def test_rename_atomic(self, storage, temp_storage):
        """Test rename is atomic."""
        storage.put_bytes(b"data", "src.txt")
        storage.rename("src.txt", "dst.txt")

        assert (temp_storage / "dst.txt").exists()
        assert not (temp_storage / "src.txt").exists()


class TestPathSafety:
    """Test path safety checks."""

    def test_reject_absolute_path(self, storage):
        """Test rejection of absolute paths."""
        with pytest.raises(ValueError):
            storage.put_bytes(b"data", "/absolute/path.txt")

    def test_reject_path_traversal(self, storage):
        """Test rejection of path traversal."""
        with pytest.raises(ValueError):
            storage.put_bytes(b"data", "../../../etc/passwd")

    def test_reject_backslash(self, storage):
        """Test rejection of backslash in paths."""
        with pytest.raises(ValueError):
            storage.put_bytes(b"data", "dir\\file.txt")

    def test_safe_nested_path(self, storage):
        """Test safe nested paths work."""
        result = storage.put_bytes(b"data", "a/b/c/d/file.txt")
        assert result.path_rel == "a/b/c/d/file.txt"


class TestErrorHandling:
    """Test error handling."""

    def test_stat_nonexistent_raises(self, storage):
        """Test stat on nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            storage.stat("missing.txt")

    def test_rename_missing_source_raises(self, storage):
        """Test rename with missing source raises."""
        with pytest.raises(FileNotFoundError):
            storage.rename("missing.txt", "dest.txt")

    def test_delete_handles_permission_error(self, storage, temp_storage):
        """Test delete handles permission errors gracefully."""
        # Create file
        storage.put_bytes(b"data", "file.txt")

        # Make it read-only and parent read-only
        path = temp_storage / "file.txt"
        os.chmod(path, 0o444)
        os.chmod(temp_storage, 0o555)

        try:
            # Should not raise (permission error handled)
            storage.delete("file.txt")
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_storage, 0o755)
            os.chmod(path, 0o644)


class TestVersionPointer:
    """Test version pointer operations."""

    def test_set_latest_version_creates_mirror(self, storage, temp_storage):
        """Test set_latest_version creates JSON mirror."""
        with patch(
            "DocsToKG.OntologyDownload.storage.localfs_duckdb.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            storage.set_latest_version("v1.0", extra={"by": "test"})

            # Verify Repo was called
            mock_repo.set_latest.assert_called_once_with("v1.0", by="test")

            # Verify JSON mirror was created
            mirror = temp_storage / "LATEST.json"
            assert mirror.exists()

            data = json.loads(mirror.read_text())
            assert data["latest"] == "v1.0"
            assert data["by"] == "test"

    def test_get_latest_version(self, storage):
        """Test get_latest_version returns from DB."""
        with patch(
            "DocsToKG.OntologyDownload.storage.localfs_duckdb.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_latest.return_value = "v2.0"
            mock_repo_class.return_value = mock_repo

            result = storage.get_latest_version()

            assert result == "v2.0"
            mock_repo.get_latest.assert_called_once()

    def test_get_latest_version_returns_none(self, storage):
        """Test get_latest_version returns None when not set."""
        with patch(
            "DocsToKG.OntologyDownload.storage.localfs_duckdb.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_latest.return_value = None
            mock_repo_class.return_value = mock_repo

            result = storage.get_latest_version()

            assert result is None


class TestIntegration:
    """Integration tests."""

    def test_complete_workflow(self, storage, temp_storage):
        """Test complete storage workflow."""
        # Upload file
        src = temp_storage / "source.txt"
        src.write_text("content")
        storage.put_file(src, "data/file.txt")

        # Write bytes
        storage.put_bytes(b"more data", "data/other.bin")

        # List files (excluding source.txt which is outside storage)
        files = storage.list()
        storage_files = [f for f in files if not f.startswith("source")]
        assert len(storage_files) == 2
        assert "data/file.txt" in storage_files
        assert "data/other.bin" in storage_files

        # Stat file
        stat = storage.stat("data/file.txt")
        assert stat.size == 7

        # Delete one
        storage.delete("data/other.bin")

        # Verify
        assert storage.exists("data/file.txt")
        assert not storage.exists("data/other.bin")

    def test_version_pointer_atomic(self, storage, temp_storage):
        """Test version pointer update is atomic."""
        with patch(
            "DocsToKG.OntologyDownload.storage.localfs_duckdb.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            storage.set_latest_version("v1.0")

            # Verify mirror is created atomically
            mirror = temp_storage / "LATEST.json"
            assert mirror.exists()

            # Verify no tmp files left
            tmp_files = [
                f
                for f in temp_storage.glob("**/*")
                if ".tmp-" in f.name
            ]
            assert len(tmp_files) == 0
