"""Atomic file write utilities and integrity verification (P1 scope).

This module provides robust I/O primitives for the download pipeline:

- :func:`atomic_write_stream` - Write response stream to disk atomically with
  Content-Length verification, ensuring no partial/truncated files.
- :class:`SizeMismatchError` - Raised when actual bytes written don't match
  Content-Length header.

All operations use fsync and directory-level atomicity to prevent data loss
on crashes.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Iterator, Optional

__all__ = ["SizeMismatchError", "atomic_write_stream"]

logger = logging.getLogger(__name__)


class SizeMismatchError(Exception):
    """Raised when downloaded bytes don't match Content-Length header.
    
    Attributes:
        expected: Expected bytes (from Content-Length).
        actual: Actual bytes written.
    """
    
    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Size mismatch: expected {expected} bytes, got {actual} bytes"
        )


def atomic_write_stream(
    dest_path: str,
    byte_iter: Iterator[bytes],
    *,
    expected_len: Optional[int] = None,
    chunk_size: int = 1 << 20,  # 1 MiB default
) -> int:
    """Write response stream to destination path atomically.
    
    Uses a temporary file + fsync + rename pattern to guarantee atomicity.
    If expected_len is provided and actual bytes written don't match, the
    temporary file is removed and SizeMismatchError is raised.
    
    Args:
        dest_path: Absolute path where file should be written.
        byte_iter: Iterator yielding chunks of bytes (e.g., from httpx.Response.iter_bytes).
        expected_len: Expected file size (from Content-Length header); 
                     None means no verification.
        chunk_size: Buffer size for reads/writes (default 1 MiB).
        
    Returns:
        Number of bytes written.
        
    Raises:
        SizeMismatchError: If expected_len is provided and actual != expected.
        OSError: If file I/O fails.
        
    Examples:
        >>> # Use with httpx response
        >>> with httpx.stream("GET", "https://example.com/file") as resp:
        ...     written = atomic_write_stream(
        ...         "/tmp/file.pdf",
        ...         resp.iter_bytes(chunk_size=1024*1024),
        ...         expected_len=int(resp.headers.get("Content-Length", 0) or 0)
        ...     )
        >>> print(f"Wrote {written} bytes")
    """
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)
    
    # Create temporary file in same directory (atomic rename works cross-device)
    fd, tmp_path = tempfile.mkstemp(
        dir=dest_dir,
        prefix=".part-",
        suffix=".tmp"
    )
    bytes_written = 0
    
    try:
        with os.fdopen(fd, "wb", buffering=0) as f:
            for chunk in byte_iter:
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
            
            # Ensure all data is written to disk
            f.flush()
            os.fsync(f.fileno())
        
        # Verify size if Content-Length was provided
        if expected_len is not None and bytes_written != expected_len:
            os.unlink(tmp_path)
            raise SizeMismatchError(expected_len, bytes_written)
        
        # Atomic rename to final location
        os.replace(tmp_path, dest_path)
        
        # Fsync directory to ensure rename is durable
        dir_fd = os.open(dest_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        
        return bytes_written
        
    except SizeMismatchError:
        # Re-raise without cleanup (already removed tmp_path above)
        raise
    except Exception:
        # Clean up temporary file on any error
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
