# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.io_utils",
#   "purpose": "Atomic file write utilities and Content-Length verification for download integrity",
#   "sections": [
#     {
#       "id": "sizemismatcherror",
#       "name": "SizeMismatchError",
#       "anchor": "class-sizemismatcherror",
#       "kind": "class"
#     },
#     {
#       "id": "atomic-write-stream",
#       "name": "atomic_write_stream",
#       "anchor": "function-atomic-write-stream",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Atomic file write utilities and integrity verification (P1 Observability & Integrity).

**Purpose**
-----------
This module provides production-grade I/O primitives for the ContentDownload pipeline,
ensuring downloaded artifacts are persisted atomically with Content-Length verification
to prevent partial/corrupted files on crashes or network failures.

**Responsibilities**
--------------------
- Write response streams to disk atomically using temporary file + fsync + rename pattern
- Verify downloaded byte count matches Content-Length header (when provided)
- Clean up temporary files on failures
- Provide deterministic, crash-safe file operations
- Support efficient streaming with configurable chunk sizes

**Key Classes & Functions**
---------------------------

:class:`SizeMismatchError`
  Exception raised when downloaded bytes don't match Content-Length header.
  Attributes: expected (int), actual (int) for diagnostics.

:func:`atomic_write_stream`
  Write response stream to disk atomically with integrity verification.
  Returns byte count written. Raises SizeMismatchError on size mismatch.

**Integration Points**
----------------------
- Called from streaming.py::stream_to_part() after HTTP GET
- Used by download execution helpers for payload persistence
- Raises SizeMismatchError which triggers telemetry.log_io_attempt()

**Safety & Reliability**
------------------------
- **Atomic writes**: Temporary file + fsync + os.replace ensures no partial files
- **Directory fsync**: Ensures rename is durable on crashes
- **Error recovery**: Cleans up temporary files on any failure
- **Content verification**: Matches bytes to Content-Length header
- **Cross-filesystem support**: Uses same directory for atomic rename

**Performance**
---------------
- Default 1 MiB chunk size balances memory and throughput
- Configurable chunk_size parameter for tuning
- No extra copies; streams directly to disk
- fsync overhead only at stream end (not per-chunk)

**Design Pattern**
------------------
This module follows P1's principle of **non-breaking observability**:
- Pure functions (no hidden state)
- Clear error semantics (SizeMismatchError is explicit)
- Optional verification (expected_len=None skips check)
- Thread-safe (no shared state between calls)
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

    This exception indicates that the number of bytes actually written to disk
    does not match the Content-Length header provided by the server. This
    typically indicates network corruption, truncation, or server misconfiguration.

    Attributes:
        expected: Expected bytes (from Content-Length header).
        actual: Actual bytes successfully written to disk.

    Example:
        >>> try:
        ...     atomic_write_stream("/path/file", byte_iter, expected_len=1000000)
        ... except SizeMismatchError as e:
        ...     logger.error(f"Download corrupted: expected {e.expected}, got {e.actual}")
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Size mismatch: expected {expected} bytes, got {actual} bytes")


def atomic_write_stream(
    dest_path: str,
    byte_iter: Iterator[bytes],
    *,
    expected_len: Optional[int] = None,
    chunk_size: int = 1 << 20,  # 1 MiB default
) -> int:
    """Write response stream to destination path atomically with integrity verification.

    Uses a temporary file + fsync + atomic rename pattern to guarantee that either
    the entire file is written successfully or no partial file remains on disk.
    If expected_len is provided and actual bytes written don't match, the temporary
    file is removed and SizeMismatchError is raised.

    **Atomicity guarantee:**
    - Writes to temporary file in same directory (ensures atomic rename)
    - Calls fsync on both file descriptor and directory
    - Uses os.replace for atomic rename
    - On error, temporary file is cleaned up

    Args:
        dest_path: Absolute path where file should be written. Parent directories
                  are created if they don't exist.
        byte_iter: Iterator yielding chunks of bytes (e.g., from httpx.Response.iter_bytes
                  or any generator yielding bytes objects).
        expected_len: Expected file size from Content-Length header; None means no
                     verification. When provided, raises SizeMismatchError if actual
                     bytes written don't match.
        chunk_size: Buffer size for reads/writes in bytes (default 1 MiB = 2^20).
                   Larger values may improve throughput on high-latency networks;
                   smaller values reduce memory usage.

    Returns:
        Number of bytes written to the file.

    Raises:
        SizeMismatchError: If expected_len is provided and actual bytes written
                          don't equal expected_len. Temporary file is removed before
                          raising.
        OSError: If file I/O fails (permission denied, disk full, etc.).

    Examples:
        >>> # Stream HTTP response to file with verification
        >>> import httpx
        >>> with httpx.stream("GET", "https://example.com/document.pdf") as resp:
        ...     written = atomic_write_stream(
        ...         "/data/downloaded.pdf",
        ...         resp.iter_bytes(chunk_size=1024*1024),
        ...         expected_len=int(resp.headers.get("Content-Length", 0) or 0)
        ...     )
        >>> print(f"Successfully wrote {written} bytes")

        >>> # Skip verification when Content-Length not available
        >>> written = atomic_write_stream(
        ...     "/data/output.txt",
        ...     generate_data_chunks()
        ...     # expected_len not provided, so no verification
        ... )

    Notes:
        - Parent directories are created automatically with exist_ok=True
        - Empty chunks in the iterator are skipped (not written)
        - fsync is called to ensure durability on crash
        - Temporary files use .tmp suffix and .part- prefix for identification
    """
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    # Create temporary file in same directory (atomic rename works cross-device)
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".part-", suffix=".tmp")
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
