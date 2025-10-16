"""Storage backend implementations for the ontology downloader."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import List, Protocol, Tuple

try:  # pragma: no cover - optional dependency
    import fsspec  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when not installed
    fsspec = None  # type: ignore[assignment]

from .config import ConfigError
from .io_safe import sanitize_filename
from .optdeps import get_pystow

__all__ = [
    "DATA_ROOT",
    "CONFIG_DIR",
    "CACHE_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "StorageBackend",
    "LocalStorageBackend",
    "FsspecStorageBackend",
    "get_storage_backend",
    "STORAGE",
]


pystow = get_pystow()

DATA_ROOT = pystow.join("ontology-fetcher")
CONFIG_DIR = DATA_ROOT / "configs"
CACHE_DIR = DATA_ROOT / "cache"
LOG_DIR = DATA_ROOT / "logs"
LOCAL_ONTOLOGY_DIR = DATA_ROOT / "ontologies"

for directory in (CONFIG_DIR, CACHE_DIR, LOG_DIR, LOCAL_ONTOLOGY_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class StorageBackend(Protocol):
    """Protocol describing storage operations required by the pipeline."""

    root: Path

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Return a workspace for the given ontology/version combination."""

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure a local directory exists for the requested ontology version."""

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted version identifiers available for *ontology_id*."""

    def available_ontologies(self) -> List[str]:
        """Return sorted ontology identifiers known to the backend."""

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Persist processed artefacts for *ontology_id*/*version*."""

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the canonical path for *ontology_id*/*version*."""

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete stored data for *ontology_id*/*version*, returning bytes reclaimed."""

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Record the latest processed version for *ontology_id*."""


def _safe_identifiers(ontology_id: str, version: str) -> Tuple[str, str]:
    """Return identifiers sanitised for filesystem usage."""

    safe_id = sanitize_filename(ontology_id)
    safe_version = sanitize_filename(version)
    return safe_id, safe_version


def _directory_size(path: Path) -> int:
    """Return the total size in bytes for files rooted at *path*."""

    total = 0
    for entry in path.rglob("*"):
        try:
            info = entry.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


class LocalStorageBackend:
    """Storage backend that keeps artefacts on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root: Path = root

    def _version_dir(self, ontology_id: str, version: str) -> Path:
        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return self.root / safe_id / safe_version

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Create a workspace for ``ontology_id``/``version`` with required subdirs."""

        base = self.ensure_local_version(ontology_id, version)
        for subdir in ("original", "normalized", "validation"):
            (base / subdir).mkdir(parents=True, exist_ok=True)
        return base

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure the version directory exists and return its path."""

        base = self._version_dir(ontology_id, version)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def available_versions(self, ontology_id: str) -> List[str]:
        """List version identifiers currently stored for ``ontology_id``."""

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        if not base.exists():
            return []
        versions = [entry.name for entry in base.iterdir() if entry.is_dir()]
        return sorted(versions)

    def available_ontologies(self) -> List[str]:
        """List ontology identifiers known to the local backend."""

        if not self.root.exists():
            return []
        return sorted([entry.name for entry in self.root.iterdir() if entry.is_dir()])

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Hook for subclasses; local backend writes occur in-place already."""

        _ = (ontology_id, version, local_dir)  # pragma: no cover - intentional no-op

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the filesystem path holding ``ontology_id``/``version`` data."""

        return self._version_dir(ontology_id, version)

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Remove stored data for ``ontology_id``/``version`` and return bytes reclaimed."""

        path = self._version_dir(ontology_id, version)
        if not path.exists():
            return 0
        reclaimed = _directory_size(path)
        shutil.rmtree(path)
        return reclaimed

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update symbolic links/markers to highlight the latest processed version."""

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        base.mkdir(parents=True, exist_ok=True)
        link = base / "latest"
        marker = base / "latest.txt"
        target = Path(version)

        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if marker.exists():
                marker.unlink()
            marker.write_text(version)
        else:
            if marker.exists():
                marker.unlink()


class FsspecStorageBackend(LocalStorageBackend):
    """Hybrid backend that mirrors artefacts to an fsspec location."""

    def __init__(self, url: str) -> None:
        if fsspec is None:  # pragma: no cover - dependency missing
            raise ConfigError(
                "fsspec required for remote storage. Install it via 'pip install fsspec'."
            )
        fs, path = fsspec.core.url_to_fs(url)  # type: ignore[attr-defined]
        self.fs = fs
        self.base_path = PurePosixPath(path)
        super().__init__(LOCAL_ONTOLOGY_DIR)

    def _remote_version_path(self, ontology_id: str, version: str) -> PurePosixPath:
        """Return the remote storage path for ``ontology_id``/``version``."""

        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return (self.base_path / safe_id / safe_version).with_suffix("")

    def available_versions(self, ontology_id: str) -> List[str]:
        """Combine local and remote version identifiers for ``ontology_id``."""

        local_versions = super().available_versions(ontology_id)
        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        remote_dir = self.base_path / safe_id
        try:
            entries = self.fs.ls(str(remote_dir), detail=False)
        except FileNotFoundError:
            entries = []
        remote_versions = [
            PurePosixPath(entry).name for entry in entries if entry and not entry.endswith(".tmp")
        ]
        return sorted({*local_versions, *remote_versions})

    def available_ontologies(self) -> List[str]:
        """Return the union of ontology ids present locally and in remote storage."""

        local = set(super().available_ontologies())
        try:
            entries = self.fs.ls(str(self.base_path), detail=False)
        except FileNotFoundError:
            entries = []
        remote = {PurePosixPath(entry).name for entry in entries if entry}
        return sorted(local | remote)

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Mirror processed artefacts to remote storage after local completion."""

        remote_dir = self._remote_version_path(ontology_id, version)
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(local_dir)
            remote_path = remote_dir / PurePosixPath(str(relative).replace("\\", "/"))
            self.fs.makedirs(str(remote_path.parent), exist_ok=True)
            self.fs.put_file(str(path), str(remote_path))

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Remove local and remote artefacts for ``ontology_id``/``version``."""

        reclaimed = super().delete_version(ontology_id, version)
        remote_dir = self._remote_version_path(ontology_id, version)
        if not self.fs.exists(str(remote_dir)):
            return reclaimed

        try:
            remote_files = self.fs.find(str(remote_dir))
        except FileNotFoundError:
            remote_files = []
        for remote_file in remote_files:
            try:
                info = self.fs.info(remote_file)
            except FileNotFoundError:
                continue
            size = info.get("size") if isinstance(info, dict) else None
            if isinstance(size, (int, float)):
                reclaimed += int(size)
        self.fs.rm(str(remote_dir), recursive=True)
        return reclaimed


def get_storage_backend() -> StorageBackend:
    """Instantiate the storage backend based on environment configuration."""

    storage_url = os.getenv("ONTOFETCH_STORAGE_URL")
    if storage_url:
        return FsspecStorageBackend(storage_url)
    return LocalStorageBackend(LOCAL_ONTOLOGY_DIR)


STORAGE: StorageBackend = get_storage_backend()
