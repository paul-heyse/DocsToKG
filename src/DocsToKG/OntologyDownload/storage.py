# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.storage",
#   "purpose": "Implements DocsToKG.OntologyDownload.storage behaviors and helpers",
#   "sections": [
#     {
#       "id": "get-storage-backend",
#       "name": "get_storage_backend",
#       "anchor": "function-get-storage-backend",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Storage backend facade for the ontology downloader."""

from __future__ import annotations

from . import ontology_download as _core
# --- Globals ---

StorageBackend = _core.StorageBackend
LocalStorageBackend = _core.LocalStorageBackend
FsspecStorageBackend = _core.FsspecStorageBackend

# --- Globals ---

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "FsspecStorageBackend",
    "STORAGE",
    "get_storage_backend",
]
# --- Public Functions ---


def get_storage_backend() -> StorageBackend:
    """Return the active storage backend and sync the core facade.

    Args:
        None

    Returns:
        StorageBackend: Instance implementing storage operations for ontology artifacts.
    """

    backend = _core.get_storage_backend()
    _core.STORAGE = backend
    return backend


STORAGE = get_storage_backend()
