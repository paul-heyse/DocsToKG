"""Storage backend facade for the ontology downloader."""

from __future__ import annotations

from . import ontology_download as _core

StorageBackend = _core.StorageBackend
LocalStorageBackend = _core.LocalStorageBackend
FsspecStorageBackend = _core.FsspecStorageBackend

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "FsspecStorageBackend",
    "STORAGE",
    "get_storage_backend",
]


def get_storage_backend() -> StorageBackend:
    """Return the active storage backend and sync the core facade."""

    backend = _core.get_storage_backend()
    _core.STORAGE = backend
    return backend


STORAGE = get_storage_backend()
