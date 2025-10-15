"""Storage backends for ontology artifacts.

The refactored ontology downloader expects durable, versioned storage that can
support dry-run planning, streaming normalization, and version pruning. This
module centralizes local and optional remote (fsspec) storage implementations,
ensuring that metadata such as latest version markers and reclaimed space
reports remain consistent with the openspec storage management requirements.
"""

from __future__ import annotations

import importlib
import os
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .foundation import sanitize_filename
from .settings import ConfigError

try:  # pragma: no cover - optional dependency
    import fsspec  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - allow local-only mode
    fsspec = None  # type: ignore

__all__ = [
    "CONFIG_DIR",
    "CACHE_DIR",
    "LOG_DIR",
    "LOCAL_ONTOLOGY_DIR",
    "StorageBackend",
    "LocalStorageBackend",
    "FsspecStorageBackend",
    "get_storage_backend",
    "get_pystow",
    "get_rdflib",
    "get_pronto",
    "get_owlready2",
    "STORAGE",
]

_pystow: Optional[Any] = None
_rdflib: Optional[Any] = None
_pronto: Optional[Any] = None
_owlready2: Optional[Any] = None

_STUB_ATTR = "_ontofetch_stub"
_BNODE_COUNTER = 0


def _create_stub_module(name: str, attrs: Dict[str, Any]) -> ModuleType:
    """Return a :class:`ModuleType` populated with *attrs*."""

    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    setattr(module, _STUB_ATTR, True)
    return module


def _create_stub_bnode(value: Optional[str] = None) -> str:
    """Return a deterministic blank node identifier."""

    global _BNODE_COUNTER
    if value is not None:
        return value
    _BNODE_COUNTER += 1
    return f"_:b{_BNODE_COUNTER}"


def _create_stub_literal(value: Any = None) -> str:
    """Return a simple literal representation."""

    if value is None:
        return '""'
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _create_stub_uri(value: Optional[str] = None) -> str:
    """Return a URI reference wrapped in angle brackets."""

    if value is None:
        return "<>"
    if value.startswith("<") and value.endswith(">"):  # pragma: no cover - defensive
        return value
    return f"<{value}>"


class _StubNamespace:
    """Minimal replacement mimicking rdflib Namespace behaviour."""

    def __init__(self, base: str):
        self._base = base

    def __getitem__(self, key: str) -> str:
        return f"{self._base}{key}"


class _StubNamespaceManager:
    """Provide a namespaces() method compatible with rdflib."""

    def namespaces(self):  # pragma: no cover - simple empty iterator
        return []


def _import_module(name: str) -> Any:
    """Import a module by name using :mod:`importlib`.

    The indirection makes it trivial to monkeypatch the import logic in unit
    tests without modifying global interpreter state.
    """

    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, _STUB_ATTR, False):
        sys.modules.pop(name, None)
    return importlib.import_module(name)


class _PystowFallback:
    """Minimal ``pystow`` replacement used when the dependency is absent."""

    def __init__(self) -> None:
        root = os.environ.get("PYSTOW_HOME")
        self._root = Path(root) if root is not None else Path.home() / ".data"

    def join(self, *segments: str) -> Path:
        """Build a path relative to the fallback root directory."""

        return self._root.joinpath(*segments)


def get_pystow() -> Any:
    """Return the real :mod:`pystow` module or a fallback stub."""

    global _pystow
    if _pystow is not None:
        return _pystow
    try:
        _pystow = _import_module("pystow")
    except ModuleNotFoundError:  # pragma: no cover - allow minimal envs
        _pystow = _PystowFallback()
    return _pystow


def get_rdflib() -> Any:
    """Return :mod:`rdflib` or a stub supporting limited graph operations."""

    global _rdflib
    if _rdflib is not None:
        return _rdflib
    try:
        _rdflib = _import_module("rdflib")
    except ModuleNotFoundError:  # pragma: no cover - exercised in tests
        graph_cls = type(
            "Graph",
            (),
            {
                "_ontofetch_stub": True,
                "__init__": lambda self: None,
                "parse": lambda self, *_args, **_kwargs: None,
                "serialize": lambda self, destination=None, format=None: None,
                "namespaces": lambda self: [],
            },
        )
        namespace_manager = _StubNamespaceManager()
        fallback = _create_stub_module(
            "rdflib",
            {
                "Graph": graph_cls,
                "Namespace": _StubNamespace,
                "BNode": lambda value=None: _create_stub_bnode(value),
                "Literal": lambda value=None: _create_stub_literal(value),
                "URIRef": lambda value=None: _create_stub_uri(value),
                "namespace": _create_stub_module("namespace", {"NamespaceManager": namespace_manager}),
            },
        )
        fallback.namespace = _create_stub_module("namespace", {"NamespaceManager": _StubNamespaceManager})
        _rdflib = fallback
    return _rdflib


def get_pronto() -> Any:
    """Return :mod:`pronto` or a stub with minimal ontology behaviour."""

    global _pronto
    if _pronto is not None:
        return _pronto
    try:
        _pronto = _import_module("pronto")
    except ModuleNotFoundError:  # pragma: no cover - test fallback
        ontology_cls = type(
            "Ontology",
            (),
            {
                "_ontofetch_stub": True,
                "__init__": lambda self, _path=None: None,
                "terms": lambda self: [],
                "dump": lambda self, *_args, **_kwargs: None,
            },
        )
        _pronto = _create_stub_module("pronto", {"Ontology": ontology_cls})
    return _pronto


def get_owlready2() -> Any:
    """Return :mod:`owlready2` or a stub matching the API used in validators."""

    global _owlready2
    if _owlready2 is not None:
        return _owlready2
    try:
        _owlready2 = _import_module("owlready2")
    except ModuleNotFoundError:  # pragma: no cover - test fallback
        class _Ontology:
            _ontofetch_stub = True

            def __init__(self, iri: str) -> None:
                self.iri = iri

            def load(self) -> "_Ontology":
                return self

            def classes(self) -> Iterable[str]:
                return []

        def _get_ontology(iri: str) -> _Ontology:
            return _Ontology(iri)

        _owlready2 = _create_stub_module("owlready2", {"get_ontology": _get_ontology})
    return _owlready2


pystow = get_pystow()

DATA_ROOT = pystow.join("ontology-fetcher")
CONFIG_DIR = DATA_ROOT / "configs"
CACHE_DIR = DATA_ROOT / "cache"
LOG_DIR = DATA_ROOT / "logs"
LOCAL_ONTOLOGY_DIR = DATA_ROOT / "ontologies"

for directory in (CONFIG_DIR, CACHE_DIR, LOG_DIR, LOCAL_ONTOLOGY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

class StorageBackend(Protocol):
    """Protocol describing the operations required by the downloader pipeline.

    Attributes:
        root_path: Canonical base path that implementations expose for disk
            storage.  Remote-only backends can synthesize this attribute for
            instrumentation purposes.

    Examples:
        >>> class MemoryBackend(StorageBackend):
        ...     root_path = Path(\"/tmp\")  # pragma: no cover - illustrative stub
        ...     def prepare_version(self, ontology_id: str, version: str) -> Path:
        ...         ...
    """

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Return a working directory prepared for the given ontology version.

        Args:
            ontology_id: Identifier of the ontology being downloaded.
            version: Version string representing the in-flight download.

        Returns:
            Path to a freshly prepared directory tree ready for population.
        """

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Ensure the requested version is present locally and return its path.

        Args:
            ontology_id: Identifier whose version must be present.
            version: Version string that should exist on local storage.

        Returns:
            Path to the local directory containing the requested version.
        """

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted version identifiers currently stored for an ontology.

        Args:
            ontology_id: Identifier whose known versions are requested.

        Returns:
            Sorted list of version strings recognised by the backend.
        """

    def available_ontologies(self) -> List[str]:
        """Return sorted ontology identifiers known to the backend.

        Args:
            None

        Returns:
            Alphabetically sorted list of ontology identifiers the backend can
            service.
        """

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Persist the working directory after validation succeeds.

        Args:
            ontology_id: Identifier of the ontology that completed processing.
            version: Version string associated with the finalized artifacts.
            local_dir: Directory containing the validated ontology payload.

        Returns:
            None.
        """

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the canonical storage path for ``ontology_id``/``version``.

        Args:
            ontology_id: Identifier of the ontology being queried.
            version: Version string for which a canonical path is needed.

        Returns:
            Path pointing to the storage location for the requested version.
        """

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete a stored version returning the number of bytes reclaimed.

        Args:
            ontology_id: Identifier whose version should be removed.
            version: Version string targeted for deletion.

        Returns:
            Number of bytes reclaimed by removing the stored version.

        Raises:
            OSError: If the underlying storage provider fails to delete data.
        """

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update the latest version marker for operators and CLI tooling.

        Args:
            ontology_id: Identifier whose latest marker requires updating.
            version: Version string to record as the active release.

        Returns:
            None.
        """


def _safe_identifiers(ontology_id: str, version: str) -> Tuple[str, str]:
    """Return identifiers sanitized for filesystem usage."""

    safe_id = sanitize_filename(ontology_id)
    safe_version = sanitize_filename(version)
    return safe_id, safe_version


def _directory_size(path: Path) -> int:
    """Return the total size in bytes for all regular files under ``path``."""

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
    """Storage backend that keeps ontology artifacts on the local filesystem.

    Attributes:
        root: Base directory that stores ontology versions grouped by identifier.

    Examples:
        >>> backend = LocalStorageBackend(LOCAL_ONTOLOGY_DIR)
        >>> backend.available_ontologies()
        []
    """

    def __init__(self, root: Path) -> None:
        """Initialise the backend with a given storage root.

        Args:
            root: Directory used to persist ontology artifacts.

        Returns:
            None
        """

        self.root = root

    def _version_dir(self, ontology_id: str, version: str) -> Path:
        safe_id, safe_version = _safe_identifiers(ontology_id, version)
        return self.root / safe_id / safe_version

    def prepare_version(self, ontology_id: str, version: str) -> Path:
        """Create the working directory structure for a download run.

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
        """Ensure a local workspace exists for ``ontology_id``/``version``.

        Args:
            ontology_id: Identifier whose workspace must exist.
            version: Version string that should map to a directory.

        Returns:
            Path to the local directory for the ontology version.
        """

        base = self._version_dir(ontology_id, version)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def available_versions(self, ontology_id: str) -> List[str]:
        """Return sorted versions already present for an ontology.

        Args:
            ontology_id: Identifier whose stored versions should be listed.

        Returns:
            Sorted list of version strings found under the storage root.
        """

        safe_id, _ = _safe_identifiers(ontology_id, "unused")
        base = self.root / safe_id
        if not base.exists():
            return []
        versions = [entry.name for entry in base.iterdir() if entry.is_dir()]
        return sorted(versions)

    def available_ontologies(self) -> List[str]:
        """Return ontology identifiers discovered under ``root``.

        Args:
            None

        Returns:
            Sorted list of ontology identifiers available locally.
        """

        if not self.root.exists():
            return []
        return sorted([entry.name for entry in self.root.iterdir() if entry.is_dir()])

    def finalize_version(self, ontology_id: str, version: str, local_dir: Path) -> None:
        """Finalize a local version directory (no-op for purely local storage).

        Args:
            ontology_id: Identifier that finished processing.
            version: Version string associated with the processed ontology.
            local_dir: Directory containing the ready-to-serve ontology.

        Returns:
            None.
        """

        _ = (ontology_id, version, local_dir)  # pragma: no cover - intentional no-op

    def version_path(self, ontology_id: str, version: str) -> Path:
        """Return the local storage directory for the requested version.

        Args:
            ontology_id: Identifier being queried.
            version: Version string whose storage path is needed.

        Returns:
            Path pointing to the stored ontology version.
        """

        return self._version_dir(ontology_id, version)

    def delete_version(self, ontology_id: str, version: str) -> int:
        """Delete a stored ontology version returning reclaimed bytes.

        Args:
            ontology_id: Identifier whose stored version should be removed.
            version: Version string targeted for deletion.

        Returns:
            Number of bytes reclaimed by removing the version directory.

        Raises:
            OSError: Propagated if filesystem deletion fails.
        """

        path = self._version_dir(ontology_id, version)
        if not path.exists():
            return 0
        reclaimed = _directory_size(path)
        shutil.rmtree(path)
        return reclaimed

    def set_latest_version(self, ontology_id: str, version: str) -> None:
        """Update symlink and marker file indicating the latest version.

        Args:
            ontology_id: Identifier whose latest marker should be updated.
            version: Version string to record as the latest processed build.

        Returns:
            None.
        """

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
    """Hybrid storage backend that mirrors artifacts to an fsspec location.

    Attributes:
        fs: ``fsspec`` filesystem instance used for remote operations.
        base_path: Root path within the remote filesystem where artefacts live.

    Examples:
        >>> backend = FsspecStorageBackend("memory://ontologies")  # doctest: +SKIP
        >>> backend.available_ontologies()  # doctest: +SKIP
        []
    """

    def __init__(self, url: str) -> None:
        """Create a hybrid storage backend backed by an fsspec URL.

        Args:
            url: Remote ``fsspec`` URL (for example ``s3://bucket/prefix``).

        Raises:
            ConfigError: If :mod:`fsspec` is not installed or the URL is invalid.

        Returns:
            None
        """

        if fsspec is None:  # pragma: no cover - exercised when dependency missing
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
        """Return versions aggregated from local cache and remote storage.

        Args:
            ontology_id: Identifier whose version catalogue is required.

        Returns:
            Sorted list combining local and remote version identifiers.
        """

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
        """Return ontology identifiers available locally or remotely.

        Args:
            None

        Returns:
            Sorted set union of local and remote ontology identifiers.
        """

        local_ids = super().available_ontologies()
        try:
            entries = self.fs.ls(str(self.base_path), detail=False)
        except FileNotFoundError:
            entries = []
        remote_ids = [
            PurePosixPath(entry).name for entry in entries if entry and not entry.endswith(".tmp")
        ]
        return sorted({*local_ids, *remote_ids})

    def ensure_local_version(self, ontology_id: str, version: str) -> Path:
        """Mirror a remote ontology version into the local cache when absent.

        Args:
            ontology_id: Identifier whose version should exist locally.
            version: Version string to ensure within the local cache.

        Returns:
            Path to the local directory containing the requested version.
        """

        base = super().ensure_local_version(ontology_id, version)
        manifest_path = base / "manifest.json"
        if manifest_path.exists():
            return base

        remote_dir = self._remote_version_path(ontology_id, version)
        if not self.fs.exists(str(remote_dir)):
            return base

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
        """Upload the finalized local directory to the remote store.

        Args:
            ontology_id: Identifier of the ontology that has completed processing.
            version: Version string associated with the finalised ontology.
            local_dir: Directory containing the validated ontology payload.

        Returns:
            None.
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
        """Delete both local and remote copies of a stored version.

        Args:
            ontology_id: Identifier whose stored version should be deleted.
            version: Version string targeted for deletion.

        Returns:
            Total bytes reclaimed across local and remote storage.

        Raises:
            OSError: Propagated if remote deletion fails irrecoverably.
        """

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
    """Instantiate the storage backend based on environment configuration.

    Args:
        None

    Returns:
        Storage backend instance selected according to ``ONTOFETCH_STORAGE_URL``.
    """

    storage_url = os.getenv("ONTOFETCH_STORAGE_URL")
    if storage_url:
        return FsspecStorageBackend(storage_url)
    return LocalStorageBackend(LOCAL_ONTOLOGY_DIR)


STORAGE: StorageBackend = get_storage_backend()
