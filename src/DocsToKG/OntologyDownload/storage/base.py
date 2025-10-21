"""Storage backend abstraction for DuckDB catalog.

Provides a unified interface for storage operations across different backends
(local filesystem, cloud storage, etc.). All operations are designed to work
with atomic writes and transactional guarantees.

NAVMAP:
  - StoredObject: Result type for file operations
  - StoredStat: File metadata (size, etag, mtime)
  - StorageBackend: Abstract protocol defining storage interface
  - Core Methods:
    * Writes: put_file, put_bytes (atomic)
    * Deletes: delete (safe)
    * Reads: exists, stat, list, resolve_url
    * Version Control: set_latest_version, get_latest_version
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class StoredObject:
    """Result of a successful storage operation.

    Attributes:
        path_rel: Relative path in storage backend
        size: File size in bytes (None if unknown)
        etag: Entity tag for cache validation (None if not supported)
        url: Absolute URL to access the object
    """

    path_rel: str
    size: Optional[int]
    etag: Optional[str]
    url: str


@dataclass(frozen=True)
class StoredStat:
    """File metadata from stat operation.

    Attributes:
        size: File size in bytes
        etag: Entity tag for cache validation (None if not supported)
        last_modified: Unix epoch timestamp (None if not available)
    """

    size: int
    etag: Optional[str]
    last_modified: Optional[float]


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage backend interface.

    Defines the contract for all storage backends. Implementations should
    ensure atomic writes, proper error handling, and consistency with
    DuckDB transactions.

    Implementation Notes:
      - All writes should be atomic (use tmp files + rename)
      - All deletes should be safe (handle missing files gracefully)
      - fsync() should be called to ensure durability
      - Paths should be validated (no path traversal)
    """

    def base_url(self) -> str:
        """Get base URL for this storage backend.

        Returns:
            Base URL or path as string
        """
        ...

    def put_file(
        self, local: Path, remote_rel: str, *, meta: dict | None = None
    ) -> StoredObject:
        """Upload a local file to storage.

        Performs atomic upload with fsync() to ensure durability.

        Args:
            local: Local file path to upload
            remote_rel: Relative path in storage backend
            meta: Optional metadata dict (etag, content-type, etc.)

        Returns:
            StoredObject with result information

        Raises:
            ValueError: If path is unsafe (e.g., path traversal)
            OSError: On file I/O errors
        """
        ...

    def put_bytes(
        self, data: bytes, remote_rel: str, *, meta: dict | None = None
    ) -> StoredObject:
        """Write bytes to storage.

        Performs atomic write with fsync() to ensure durability.

        Args:
            data: Bytes to write
            remote_rel: Relative path in storage backend
            meta: Optional metadata dict

        Returns:
            StoredObject with result information

        Raises:
            ValueError: If path is unsafe
            OSError: On write errors
        """
        ...

    def rename(self, src_rel: str, dst_rel: str) -> None:
        """Atomically rename/move a stored object.

        Args:
            src_rel: Source relative path
            dst_rel: Destination relative path

        Raises:
            ValueError: If paths are unsafe
            FileNotFoundError: If source doesn't exist
            OSError: On rename errors
        """
        ...

    def delete(self, path_rel_or_list: str | list[str]) -> None:
        """Delete object(s) from storage.

        Deletes safely - missing files are silently ignored.

        Args:
            path_rel_or_list: Single path or list of relative paths

        Raises:
            ValueError: If any path is unsafe
        """
        ...

    def exists(self, remote_rel: str) -> bool:
        """Check if object exists in storage.

        Args:
            remote_rel: Relative path to check

        Returns:
            True if object exists, False otherwise
        """
        ...

    def stat(self, remote_rel: str) -> StoredStat:
        """Get metadata for stored object.

        Args:
            remote_rel: Relative path to stat

        Returns:
            StoredStat with size, etag, and mtime

        Raises:
            FileNotFoundError: If object doesn't exist
        """
        ...

    def list(self, prefix_rel: str = "") -> list[str]:
        """List all objects with optional prefix filter.

        Args:
            prefix_rel: Prefix to filter by (empty = all objects)

        Returns:
            List of relative paths matching prefix

        Raises:
            ValueError: If prefix is unsafe
        """
        ...

    def resolve_url(self, remote_rel: str) -> str:
        """Get absolute URL for stored object.

        Args:
            remote_rel: Relative path

        Returns:
            Absolute URL to access object

        Raises:
            ValueError: If path is unsafe
        """
        ...

    def set_latest_version(self, version: str, extra: dict | None = None) -> None:
        """Set the latest version pointer.

        Updates the version pointer in storage backend. The "latest" version
        pointer is typically DB-authoritative, but may be mirrored to a
        JSON file for convenience.

        Args:
            version: Version identifier to set as latest
            extra: Optional extra metadata (by, timestamp, etc.)

        Raises:
            OSError: On write errors
        """
        ...

    def get_latest_version(self) -> Optional[str]:
        """Get the latest version pointer.

        Retrieves the version pointer from storage backend. Returns None
        if no latest version is set.

        Returns:
            Latest version identifier, or None if not set
        """
        ...
