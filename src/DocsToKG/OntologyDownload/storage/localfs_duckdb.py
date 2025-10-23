# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.storage.localfs_duckdb",
#   "purpose": "Local filesystem + DuckDB storage backend implementation.",
#   "sections": [
#     {
#       "id": "localduckdbstorage",
#       "name": "LocalDuckDBStorage",
#       "anchor": "class-localduckdbstorage",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Local filesystem + DuckDB storage backend implementation.

Implements the StorageBackend protocol for local filesystem storage with
DuckDB catalog integration. Provides atomic writes, safe deletes, and
DB-authoritative latest version pointer.

NAVMAP:
  - LocalDuckDBStorage: Main implementation class
  - Core Features:
    * Atomic file writes with fsync()
    * Safe deletes (missing files ignored)
    * Path validation (no traversal)
    * DuckDB latest pointer integration
    * JSON mirror for convenience
    * Comprehensive error handling
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..observability.events import emit_event
from .base import StorageBackend, StoredObject, StoredStat

if TYPE_CHECKING:
    from ..catalog.connection import DuckDBConfig
    from ..catalog.repo import Repo
else:
    # Fallback for when modules don't exist yet
    DuckDBConfig = Any  # type: ignore[assignment, misc]
    Repo = Any  # type: ignore[assignment, misc]


@dataclass(frozen=True)
class LocalDuckDBStorage(StorageBackend):
    """Local filesystem storage with DuckDB catalog backend.

    Stores all artifacts on local filesystem while maintaining
    DB-authoritative latest version pointer. Optionally mirrors
    latest pointer to JSON file for convenience.

    Attributes:
        root: Root directory for storage
        db: DuckDB configuration
        latest_name: JSON filename for latest mirror
        write_latest_mirror: Whether to write JSON mirror
    """

    root: Path
    db: DuckDBConfig
    latest_name: str = "LATEST.json"
    write_latest_mirror: bool = True

    def base_url(self) -> str:
        """Get base URL (root path as string)."""
        return str(self.root.resolve())

    def _abs(self, rel: str) -> Path:
        """Convert relative path to absolute, with safety checks.

        Args:
            rel: Relative path in storage

        Returns:
            Absolute Path object

        Raises:
            ValueError: If path is unsafe (traversal, absolute, backslash)
        """
        if rel.startswith("/") or ".." in Path(rel).parts or "\\" in rel:
            msg = f"unsafe storage relpath: {rel}"
            raise ValueError(msg)
        return (self.root / Path(rel)).resolve()

    def put_file(self, local: Path, remote_rel: str, *, meta: dict | None = None) -> StoredObject:
        """Upload local file to storage with atomic write.

        Args:
            local: Local file path to upload
            remote_rel: Relative path in storage
            meta: Optional metadata (unused in local backend)

        Returns:
            StoredObject with upload result

        Raises:
            ValueError: If remote_rel is unsafe
            OSError: On file I/O errors
        """
        dest = self._abs(remote_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")

        # Stream copy with fsync
        try:
            with open(local, "rb") as rf, open(tmp, "wb") as wf:
                while True:
                    chunk = rf.read(1024 * 1024)
                    if not chunk:
                        break
                    wf.write(chunk)
                wf.flush()
                os.fsync(wf.fileno())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        # Atomic rename
        os.replace(tmp, dest)

        # Fsync directory to ensure rename is durable
        fd = os.open(str(dest.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        st = os.stat(dest)
        result = StoredObject(path_rel=remote_rel, size=st.st_size, etag=None, url=str(dest))

        # Emit storage.put event per acceptance criteria
        emit_event(
            type="storage.put",
            level="INFO",
            payload={
                "operation": "put_file",
                "path_rel": remote_rel,
                "size_bytes": result.size,
                "source": str(local),
            },
        )

        return result

    def put_bytes(self, data: bytes, remote_rel: str, *, meta: dict | None = None) -> StoredObject:
        """Write bytes to storage with atomic write.

        Args:
            data: Bytes to write
            remote_rel: Relative path in storage
            meta: Optional metadata (unused in local backend)

        Returns:
            StoredObject with write result

        Raises:
            ValueError: If remote_rel is unsafe
            OSError: On write errors
        """
        dest = self._abs(remote_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")

        # Atomic write with fsync
        try:
            with open(tmp, "wb") as wf:
                wf.write(data)
                wf.flush()
                os.fsync(wf.fileno())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        # Atomic rename
        os.replace(tmp, dest)

        # Fsync directory
        fd = os.open(str(dest.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        st = os.stat(dest)
        result = StoredObject(path_rel=remote_rel, size=st.st_size, etag=None, url=str(dest))

        # Emit storage.put event per acceptance criteria
        emit_event(
            type="storage.put",
            level="INFO",
            payload={
                "operation": "put_bytes",
                "path_rel": remote_rel,
                "size_bytes": result.size,
            },
        )

        return result

    def rename(self, src_rel: str, dst_rel: str) -> None:
        """Atomically rename/move stored object.

        Args:
            src_rel: Source relative path
            dst_rel: Destination relative path

        Raises:
            ValueError: If paths are unsafe
            FileNotFoundError: If source doesn't exist
        """
        src = self._abs(src_rel)
        dst = self._abs(dst_rel)

        if not src.exists():
            msg = f"source not found: {src_rel}"
            raise FileNotFoundError(msg)

        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)

        # Fsync directory
        fd = os.open(str(dst.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        # Emit storage.mv event per acceptance criteria
        emit_event(
            type="storage.mv",
            level="INFO",
            payload={
                "src_rel": src_rel,
                "dst_rel": dst_rel,
            },
        )

    def delete(self, path_rel_or_list: str | list[str]) -> None:
        """Delete object(s) from storage (safe - missing files ignored).

        Args:
            path_rel_or_list: Single path or list of paths to delete

        Raises:
            ValueError: If any path is unsafe
        """
        rels = [path_rel_or_list] if isinstance(path_rel_or_list, str) else path_rel_or_list

        deleted_count = 0
        for rel in rels:
            p = self._abs(rel)
            try:
                p.unlink()
                deleted_count += 1
            except FileNotFoundError:
                pass

        # Emit storage.delete event per acceptance criteria
        if deleted_count > 0:
            emit_event(
                type="storage.delete",
                level="INFO",
                payload={
                    "count": deleted_count,
                    "paths": rels if isinstance(path_rel_or_list, str) else None,
                },
            )

    def exists(self, remote_rel: str) -> bool:
        """Check if object exists in storage.

        Args:
            remote_rel: Relative path to check

        Returns:
            True if exists, False otherwise
        """
        return self._abs(remote_rel).exists()

    def stat(self, remote_rel: str) -> StoredStat:
        """Get file metadata.

        Args:
            remote_rel: Relative path to stat

        Returns:
            StoredStat with size, etag, and mtime

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        p = self._abs(remote_rel)
        st = os.stat(p)  # Raises FileNotFoundError if missing
        return StoredStat(size=st.st_size, etag=None, last_modified=st.st_mtime)

    def list(self, prefix_rel: str = "") -> list[str]:
        """List objects with optional prefix filter.

        Args:
            prefix_rel: Prefix to filter by (empty = all)

        Returns:
            List of relative paths

        Raises:
            ValueError: If prefix is unsafe
        """
        if prefix_rel:
            self._abs(prefix_rel)  # Validate prefix

        base = self._abs(prefix_rel) if prefix_rel else self.root.resolve()
        out: list[str] = []

        try:
            for root, _dirs, files in os.walk(base):
                for fname in files:
                    fpath = Path(root) / fname
                    rel = fpath.resolve().relative_to(self.root.resolve())
                    out.append(rel.as_posix())
        except FileNotFoundError:
            # Directory doesn't exist yet - return empty list
            pass

        return out

    def resolve_url(self, remote_rel: str) -> str:
        """Get absolute URL for object.

        Args:
            remote_rel: Relative path

        Returns:
            Absolute file path as URL

        Raises:
            ValueError: If path is unsafe
        """
        return str(self._abs(remote_rel))

    def set_latest_version(self, version: str, extra: dict | None = None) -> None:
        """Set latest version pointer.

        Updates DB-authoritative pointer via Repo, and optionally
        writes JSON mirror for convenience.

        Args:
            version: Version ID to set as latest
            extra: Optional metadata (by, timestamp, etc.)

        Raises:
            OSError: On write errors
        """
        # Update DB (authoritative)
        repo = Repo(self.db)
        repo.set_latest(version, by=extra.get("by") if extra else None)

        # Optional JSON mirror
        if self.write_latest_mirror:
            mirror = self.root / self.latest_name
            tmp = mirror.with_name(mirror.name + f".tmp-{os.getpid()}")
            mirror.parent.mkdir(parents=True, exist_ok=True)

            body = {
                "latest": version,
                "ts": None,
                "storage": str(self.root.resolve()),
                "by": extra.get("by") if extra else None,
                "db": str(self.db.path.resolve()),
            }

            try:
                with open(tmp, "w", encoding="utf-8") as wf:
                    json.dump(body, wf, separators=(",", ":"), sort_keys=True)
                    wf.flush()
                    os.fsync(wf.fileno())
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

            os.replace(tmp, mirror)

            # Fsync directory
            fd = os.open(str(mirror.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        # Emit storage.latest.set event per acceptance criteria
        emit_event(
            type="storage.latest.set",
            level="INFO",
            payload={
                "version_id": version,
                "mirror_written": self.write_latest_mirror,
                "by": extra.get("by") if extra else None,
            },
            version_id=version,
        )

    def get_latest_version(self) -> Optional[str]:
        """Get latest version pointer from DB.

        Returns DB-authoritative pointer. JSON mirror is not consulted.

        Returns:
            Latest version ID, or None if not set
        """
        repo = Repo(self.db)
        return repo.get_latest()
