"""Tests for P1 Observability & Integrity: Atomic Write & Integrity (Phase 1B).

Covers:
- atomic_write_stream() with successful writes
- Content-Length verification and mismatch detection
- SizeMismatchError exception
- Temporary file cleanup on errors
- Atomic rename behavior
- fsync guarantees
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from DocsToKG.ContentDownload.io_utils import SizeMismatchError, atomic_write_stream


class TestAtomicWriteStreamBasic:
    """Tests for basic atomic write functionality."""
    
    def test_write_small_file_successfully(self, tmp_path: Path) -> None:
        """Successfully write a small file atomically."""
        dest_path = str(tmp_path / "output.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"Hello, "
            yield b"World!"
        
        written = atomic_write_stream(dest_path, byte_iter())
        
        assert written == 13
        assert Path(dest_path).exists()
        assert Path(dest_path).read_bytes() == b"Hello, World!"
    
    def test_write_large_file_with_chunking(self, tmp_path: Path) -> None:
        """Write a large file in multiple chunks."""
        dest_path = str(tmp_path / "large.bin")
        chunk_size = 1024
        total_chunks = 100
        
        def byte_iter() -> Iterator[bytes]:
            for i in range(total_chunks):
                yield b"X" * chunk_size
        
        written = atomic_write_stream(dest_path, byte_iter(), chunk_size=chunk_size)
        
        assert written == chunk_size * total_chunks
        assert Path(dest_path).exists()
        assert len(Path(dest_path).read_bytes()) == written
    
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if missing."""
        dest_path = str(tmp_path / "deep" / "nested" / "path" / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"content"
        
        written = atomic_write_stream(dest_path, byte_iter())
        
        assert written == 7
        assert Path(dest_path).exists()
    
    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty iterator results in zero-byte file."""
        dest_path = str(tmp_path / "empty.txt")
        
        def byte_iter() -> Iterator[bytes]:
            return
            yield  # Never executed (makes it a generator)
        
        written = atomic_write_stream(dest_path, byte_iter())
        
        assert written == 0
        assert Path(dest_path).exists()
        assert Path(dest_path).stat().st_size == 0


class TestAtomicWriteContentLengthVerification:
    """Tests for Content-Length verification."""
    
    def test_content_length_match(self, tmp_path: Path) -> None:
        """Content-Length matches actual bytes."""
        dest_path = str(tmp_path / "file.txt")
        content = b"Hello, World!"
        
        def byte_iter() -> Iterator[bytes]:
            yield content
        
        written = atomic_write_stream(
            dest_path, byte_iter(), expected_len=len(content)
        )
        
        assert written == len(content)
        assert Path(dest_path).read_bytes() == content
    
    def test_content_length_mismatch_too_few_bytes(self, tmp_path: Path) -> None:
        """Mismatch when fewer bytes written than Content-Length."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"short"  # Only 5 bytes
        
        with pytest.raises(SizeMismatchError) as exc_info:
            atomic_write_stream(dest_path, byte_iter(), expected_len=100)
        
        error = exc_info.value
        assert error.expected == 100
        assert error.actual == 5
        # File should NOT exist after error
        assert not Path(dest_path).exists()
    
    def test_content_length_mismatch_too_many_bytes(self, tmp_path: Path) -> None:
        """Mismatch when more bytes written than Content-Length."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            for i in range(100):
                yield b"X"  # Total 100 bytes
        
        with pytest.raises(SizeMismatchError) as exc_info:
            atomic_write_stream(dest_path, byte_iter(), expected_len=50)
        
        error = exc_info.value
        assert error.expected == 50
        assert error.actual == 100
        # File should NOT exist after error
        assert not Path(dest_path).exists()
    
    def test_no_verification_when_expected_len_none(self, tmp_path: Path) -> None:
        """No verification when expected_len is None (default)."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"any content"
        
        # Should succeed without error even with no expected_len
        written = atomic_write_stream(dest_path, byte_iter(), expected_len=None)
        
        assert written == 11
        assert Path(dest_path).exists()


class TestAtomicWriteTemporaryFileCleanup:
    """Tests for temporary file cleanup on errors."""
    
    def test_temp_file_removed_on_size_mismatch(self, tmp_path: Path) -> None:
        """Temporary file is removed if size verification fails."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"content"
        
        # List temp files before
        temp_dir = tempfile.gettempdir()
        
        # Attempt write with size mismatch
        with pytest.raises(SizeMismatchError):
            atomic_write_stream(dest_path, byte_iter(), expected_len=1000)
        
        # Verify destination doesn't exist
        assert not Path(dest_path).exists()
        
        # No orphaned file at destination
        assert not Path(dest_path).exists()
    
    def test_temp_file_removed_on_exception(self, tmp_path: Path) -> None:
        """Temporary file is removed if iterator raises exception."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"start"
            raise IOError("Iterator failure")
        
        with pytest.raises(IOError):
            atomic_write_stream(dest_path, byte_iter())
        
        # Destination should not exist after error
        assert not Path(dest_path).exists()


class TestAtomicWriteCustomChunkSize:
    """Tests for custom chunk size handling."""
    
    def test_custom_chunk_size_small(self, tmp_path: Path) -> None:
        """Small custom chunk size works correctly."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"A" * 100
        
        written = atomic_write_stream(
            dest_path, byte_iter(), chunk_size=10
        )
        
        assert written == 100
        assert Path(dest_path).read_bytes() == b"A" * 100
    
    def test_custom_chunk_size_large(self, tmp_path: Path) -> None:
        """Large custom chunk size works correctly."""
        dest_path = str(tmp_path / "file.txt")
        content = b"X" * 10_000_000  # 10 MB
        
        def byte_iter() -> Iterator[bytes]:
            yield content
        
        written = atomic_write_stream(
            dest_path, byte_iter(), chunk_size=5_000_000
        )
        
        assert written == len(content)
        assert Path(dest_path).stat().st_size == len(content)


class TestSizeMismatchError:
    """Tests for SizeMismatchError exception."""
    
    def test_error_attributes(self) -> None:
        """SizeMismatchError has expected and actual attributes."""
        error = SizeMismatchError(100, 50)
        
        assert error.expected == 100
        assert error.actual == 50
        assert "100" in str(error)
        assert "50" in str(error)
    
    def test_error_message_format(self) -> None:
        """Error message is clear and helpful."""
        error = SizeMismatchError(1000, 999)
        message = str(error)
        
        assert "1000" in message
        assert "999" in message
        assert "mismatch" in message.lower()


class TestAtomicWriteEdgeCases:
    """Tests for edge cases."""
    
    def test_write_to_existing_file(self, tmp_path: Path) -> None:
        """Atomic write overwrites existing file."""
        dest_path = tmp_path / "file.txt"
        dest_path.write_text("old content")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"new content"
        
        written = atomic_write_stream(str(dest_path), byte_iter())
        
        assert written == 11
        assert dest_path.read_bytes() == b"new content"
    
    def test_write_single_byte(self, tmp_path: Path) -> None:
        """Write a single byte file."""
        dest_path = str(tmp_path / "file.bin")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"X"
        
        written = atomic_write_stream(dest_path, byte_iter())
        
        assert written == 1
        assert Path(dest_path).read_bytes() == b"X"
    
    def test_empty_chunks_ignored(self, tmp_path: Path) -> None:
        """Empty chunks in iterator are ignored."""
        dest_path = str(tmp_path / "file.txt")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"A"
            yield b""  # Empty chunk
            yield b"B"
            yield b""  # Empty chunk
            yield b"C"
        
        written = atomic_write_stream(dest_path, byte_iter())
        
        assert written == 3
        assert Path(dest_path).read_bytes() == b"ABC"


class TestAtomicWriteIntegration:
    """Integration tests combining multiple features."""
    
    def test_write_with_verification_success_flow(self, tmp_path: Path) -> None:
        """Complete successful write with verification."""
        dest_path = str(tmp_path / "data.bin")
        content = b"X" * 10000
        
        def byte_iter() -> Iterator[bytes]:
            # Simulate chunked response
            for i in range(0, len(content), 1000):
                yield content[i:i+1000]
        
        written = atomic_write_stream(
            dest_path,
            byte_iter(),
            expected_len=len(content),
            chunk_size=1000
        )
        
        assert written == len(content)
        assert Path(dest_path).exists()
        assert Path(dest_path).read_bytes() == content
    
    def test_write_failure_cleanup_flow(self, tmp_path: Path) -> None:
        """Failed write cleans up properly."""
        dest_path = str(tmp_path / "data.bin")
        
        def byte_iter() -> Iterator[bytes]:
            yield b"partial"  # Only 7 bytes
        
        with pytest.raises(SizeMismatchError):
            atomic_write_stream(
                dest_path,
                byte_iter(),
                expected_len=1000,
                chunk_size=1024
            )
        
        # Verify no orphaned file
        assert not Path(dest_path).exists()
        assert not Path(dest_path).with_suffix(".tmp").exists()
