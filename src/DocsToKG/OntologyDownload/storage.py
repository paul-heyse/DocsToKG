"""
Storage backends for ontology artifacts.

This module centralizes access to ontology storage locations, supporting both
the default local filesystem layout (managed via :mod:`pystow`) and optional
remote backends powered by :mod:`fsspec`. Callers interact with the storage
backend abstractly, allowing the downloader to mirror versioned ontology
artifacts to remote object stores while maintaining a local working copy for
streaming and validation.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Protocol, Tuple

from .config import ConfigError
from .download import sanitize_filename
from .optdeps import get_pystow

try:  # pragma: no cover - guarded import for optional dependency
    import fsspec  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled dynamically
    fsspec = None  # type: ignore


pystow = get_pystow()

DATA_ROOT = pystow.join("ontology-fetcher")
CONFIG_DIR = DATA_ROOT / "configs"
CACHE_DIR = DATA_ROOT / "cache"
LOG_DIR = DATA_ROOT / "logs"
LOCAL_ONTOLOGY_DIR = DATA_ROOT / "ontologies"

for directory in (CONFIG_DIR, CACHE_DIR, LOG_DIR, LOCAL_ONTOLOGY_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class StorageBackend(Protocol):
    """Protocol describing storage backend operations.

    Implementations may store data locally, mirror to remote backends, or both.
    Callers rely on the common interface to prepare working directories,
    enumerate versions, and finalize artefacts after downloads complete.

    Attributes:
        prepare_version: Callable producing a working directory path.
        ensure_local_version: Callable that guarantees a local working copy.
        available_versions: Callable returning stored version identifiers.
        finalize_version: Callable persisting local results to durable storage.

    Examples:
        >>> backend = LocalStorageBackend(Path("/tmp/ontologies"))
        >>> isinstance(backend.available_versions("hp"), list)
        True
    """

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Return a local directory prepared for the given ontology/version.

        Args:
            ontology_id: Stable ontology identifier requested by the caller.
            version: Canonical version string resolved for the ontology.

        Returns:
            Path pointing to a base directory containing all working sub-folders.
        """

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure the specified version exists locally, syncing from remote if needed.

        Args:
            ontology_id: Ontology identifier whose artefacts are required.
            version: Version string that must be present locally.

        Returns:
            Path to the local directory where the ontology version resides.
        """

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted list of available versions from storage.

        Args:
            ontology_id: Ontology identifier to inspect.

        Returns:
            Sorted list of version strings available to the backend.
        """

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Persist local version directory to remote storage when applicable.

        Args:
            ontology_id: Ontology identifier that completed processing.
            version: Version string corresponding to the processed ontology.
            local_dir: Local directory tree ready for publication.

        Returns:
            None
        """

    def available_ontologies(self) -> List[str]:
        """Return ontology identifiers managed by the backend."""

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the local filesystem location for a stored version."""

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Remove a stored version and return number of bytes reclaimed."""

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update latest version marker for an ontology."""


def _safe_identifiers(ontology_id: str, version: str) -> Tuple[str, str]:
    """Return sanitized identifiers suitable for filesystem usage."""

    safe_id = sanitize_filename(ontology_id)
    safe_version = sanitize_filename(version)
    return safe_id, safe_version


class LocalStorageBackend:
    """Storage backend that keeps ontology artifacts on the local filesystem.

    Attributes:
        root: Root directory under which ontology artifacts are stored. Each
            ontology/version pair receives a dedicated subdirectory.

    Examples:
        >>> backend = LocalStorageBackend(Path("/tmp/ontologies"))
        >>> workspace = backend.prepare_version("hp", "2024-01-01")
        >>> (workspace / "original").exists()
        True
    """

    def __init__(self, root: Path) -> None:
        """Create a local storage backend rooted at ``root``.

        Args:
            root: Base directory that stores ontology versions.

        Returns:
            None
        """

        self.root = root

    def _version_dir(self, ontology_id: str, version: str) -> Path:
        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return self.root / safe_id / safe_version

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Create a local working directory structure for an ontology version.

        Args:
            ontology_id: Identifier of the ontology being processed.
            version: Canonical version string for the ontology.

        Returns:
            Path to the prepared base directory containing ``original``,
            ``normalized``, and ``validation`` subdirectories.
        """

        base = self.ensure_local_version(ontology_id, version)
        for subdir in ("original", "normalized", "validation"):
            (base / subdir).mkdir(parents=True, exist_ok=True)
        return base

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure a local directory exists for the requested ontology version.

        Args:
            ontology_id: Identifier of the ontology being processed.
            version: Version string that should exist locally.

        Returns:
            Path to the local directory for the ontology version.
        """

        base = self._version_dir(ontology_id, version)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def available_versions(self, ontology_id: str) -> List[str]:
        """List versions already present on disk for an ontology.

        Args:
            ontology_id: Identifier of the ontology whose versions are requested.

        Returns:
            Sorted list of version strings discovered in the local store.
        """

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        if not base.exists():
            return []
        versions = [entry.name for entry in base.iterdir() if entry.is_dir()]
        return sorted(versions)

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Finalize a local version directory (no-op for purely local storage).

        Args:
            ontology_id: Identifier of the ontology that finished processing.
            version: Version string associated with the processed ontology.
            local_dir: Path to the local directory ready for consumption.

        Returns:
            None
        """

        # Local backend already operates in-place; nothing further needed.
        _ = (ontology_id, version, local_dir)  # pragma: no cover - intentional no-op

    def available_ontologies(self) -> List[str]:
        """Return ontology identifiers present on the local filesystem."""

        if not self.root.exists():
            return []
        return sorted([entry.name for entry in self.root.iterdir() if entry.is_dir()])

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the directory path for a stored ontology version."""

        return self._version_dir(ontology_id, version)

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete a stored ontology version and return reclaimed bytes."""

        path = self.version_path(ontology_id, version)
        if not path.exists():
            return 0

        reclaimed = _directory_size(path)
        shutil.rmtree(path)
        return reclaimed

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update the latest version marker for an ontology."""

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


def _directory_size(path: Path) -> int:
    """Return the total size of all regular files within ``path``."""

    total = 0
    for entry in path.rglob("*"):
        try:
            info = entry.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


class FsspecStorageBackend(LocalStorageBackend):
    """Storage backend that mirrors ontology artifacts to a remote location via fsspec.

    Attributes:
        fs: fsspec filesystem instance used for remote interactions.
        base_path: Root path within the remote filesystem where artefacts live.

    Examples:
        >>> backend = FsspecStorageBackend("memory://ontologies")
        Traceback (most recent call last):
        ...
        ConfigError: fsspec required for remote storage. Install it via 'pip install fsspec'.
    """

    def __init__(self, url: str) -> None:
        """Create a hybrid storage backend backed by an fsspec URL.

        Args:
            url: Remote fsspec URL (e.g., ``s3://bucket/prefix``) identifying the store.

        Raises:
            ConfigError: If :mod:`fsspec` is not installed or the URL cannot be parsed.

        Returns:
            None
        """

        if fsspec is None:
            raise ConfigError(
                "fsspec required for remote storage. Install it via 'pip install fsspec'."
            )
        fs, path = fsspec.core.url_to_fs(url)  # type: ignore[attr-defined]
        self.fs = fs
        self.base_path = PurePosixPath(path)
        super().__init__(LOCAL_ONTOLOGY_DIR)

    def _remote_version_path(self, ontology_id: str, version: str) -> PurePosixPath:
        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return (self.base_path / safe_id / safe_version).with_suffix("")

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return unique versions aggregated from local cache and remote storage.

        Args:
            ontology_id: Identifier of the ontology whose versions are requested.

        Returns:
            Sorted list of versions available either locally or remotely.
        """

        local_versions = super().available_versions(ontology_id)
        remote_versions: Iterable[str] = []
        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        remote_dir = self.base_path / safe_id
        try:
            entries = self.fs.ls(str(remote_dir), detail=False)
        except FileNotFoundError:
            entries = []
        remote_versions = [
            PurePosixPath(entry).name for entry in entries if entry and not entry.endswith(".tmp")
        ]
        merged = sorted({*local_versions, *remote_versions})
        return merged

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Mirror a remote ontology version into the local cache if necessary.

        Args:
            ontology_id: Identifier of the ontology being requested.
            version: Version string to ensure locally.

        Returns:
            Path to the local directory containing the requested version.
        """

        base = super().ensure_local_version(ontology_id, version)
        remote_dir = self._remote_version_path(ontology_id, version)
        manifest_path = base / "manifest.json"
        if manifest_path.exists():
            return base
        if self.fs.exists(str(remote_dir)):
            try:
                remote_files = self.fs.find(str(remote_dir))
            except FileNotFoundError:
                remote_files = []
            for remote_file in remote_files:
                remote_path = PurePosixPath(remote_file)
                relative = remote_path.relative_to(remote_dir)
                local_path = base / Path(str(relative))
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.fs.get_file(str(remote_path), str(local_path))
        return base

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Upload the finalized local version directory to the remote store.

        Args:
            ontology_id: Identifier of the ontology ready for publication.
            version: Version string for the finalized ontology artifacts.
            local_dir: Local directory tree containing normalized outputs.

        Returns:
            None
        """

        remote_dir = self._remote_version_path(ontology_id, version)
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(local_dir)
            remote_path = remote_dir / PurePosixPath(str(relative).replace("\\", "/"))
            self.fs.makedirs(str(remote_path.parent), exist_ok=True)
            self.fs.put_file(str(path), str(remote_path))

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete both local and remote copies for a stored version."""

        reclaimed = super().delete_version(ontology_id, version)
        remote_dir = self._remote_version_path(ontology_id, version)
        try:
            self.fs.rm(str(remote_dir), recursive=True)
        except FileNotFoundError:
            pass
        return reclaimed


def get_storage_backend() -> StorageBackend:
    """Instantiate the appropriate storage backend based on environment settings.

    Args:
        None

    Returns:
        StorageBackend implementation bound to either local or remote storage.
    """

    storage_url = os.getenv("ONTOFETCH_STORAGE_URL")
    if storage_url:
        return FsspecStorageBackend(storage_url)
    return LocalStorageBackend(LOCAL_ONTOLOGY_DIR)


STORAGE: StorageBackend = get_storage_backend()
