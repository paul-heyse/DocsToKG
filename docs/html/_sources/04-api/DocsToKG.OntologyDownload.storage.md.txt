# 1. Module: storage

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.storage``.

Storage backends for ontology artifacts.

This module centralizes access to ontology storage locations, supporting both
the default local filesystem layout (managed via :mod:`pystow`) and optional
remote backends powered by :mod:`fsspec`. Callers interact with the storage
backend abstractly, allowing the downloader to mirror versioned ontology
artifacts to remote object stores while maintaining a local working copy for
streaming and validation.

## 1. Functions

### `_safe_identifiers(ontology_id, version)`

Return sanitized identifiers suitable for filesystem usage.

### `get_storage_backend()`

Instantiate the appropriate storage backend based on environment settings.

Args:
None

Returns:
StorageBackend implementation bound to either local or remote storage.

### `prepare_version(self, ontology_id, version)`

Return a local directory prepared for the given ontology/version.

Args:
ontology_id: Stable ontology identifier requested by the caller.
version: Canonical version string resolved for the ontology.

Returns:
Path pointing to a base directory containing all working sub-folders.

### `ensure_local_version(self, ontology_id, version)`

Ensure the specified version exists locally, syncing from remote if needed.

Args:
ontology_id: Ontology identifier whose artefacts are required.
version: Version string that must be present locally.

Returns:
Path to the local directory where the ontology version resides.

### `available_versions(self, ontology_id)`

Return sorted list of available versions from storage.

Args:
ontology_id: Ontology identifier to inspect.

Returns:
Sorted list of version strings available to the backend.

### `finalize_version(self, ontology_id, version, local_dir)`

Persist local version directory to remote storage when applicable.

Args:
ontology_id: Ontology identifier that completed processing.
version: Version string corresponding to the processed ontology.
local_dir: Local directory tree ready for publication.

Returns:
None

### `_version_dir(self, ontology_id, version)`

*No documentation available.*

### `prepare_version(self, ontology_id, version)`

Create a local working directory structure for an ontology version.

Args:
ontology_id: Identifier of the ontology being processed.
version: Canonical version string for the ontology.

Returns:
Path to the prepared base directory containing ``original``,
``normalized``, and ``validation`` subdirectories.

### `ensure_local_version(self, ontology_id, version)`

Ensure a local directory exists for the requested ontology version.

Args:
ontology_id: Identifier of the ontology being processed.
version: Version string that should exist locally.

Returns:
Path to the local directory for the ontology version.

### `available_versions(self, ontology_id)`

List versions already present on disk for an ontology.

Args:
ontology_id: Identifier of the ontology whose versions are requested.

Returns:
Sorted list of version strings discovered in the local store.

### `finalize_version(self, ontology_id, version, local_dir)`

Finalize a local version directory (no-op for purely local storage).

Args:
ontology_id: Identifier of the ontology that finished processing.
version: Version string associated with the processed ontology.
local_dir: Path to the local directory ready for consumption.

Returns:
None

### `_remote_version_path(self, ontology_id, version)`

*No documentation available.*

### `available_versions(self, ontology_id)`

Return unique versions aggregated from local cache and remote storage.

Args:
ontology_id: Identifier of the ontology whose versions are requested.

Returns:
Sorted list of versions available either locally or remotely.

### `ensure_local_version(self, ontology_id, version)`

Mirror a remote ontology version into the local cache if necessary.

Args:
ontology_id: Identifier of the ontology being requested.
version: Version string to ensure locally.

Returns:
Path to the local directory containing the requested version.

### `finalize_version(self, ontology_id, version, local_dir)`

Upload the finalized local version directory to the remote store.

Args:
ontology_id: Identifier of the ontology ready for publication.
version: Version string for the finalized ontology artifacts.
local_dir: Local directory tree containing normalized outputs.

Returns:
None

## 2. Classes

### `StorageBackend`

Protocol describing storage backend operations.

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

### `LocalStorageBackend`

Storage backend that keeps ontology artifacts on the local filesystem.

Attributes:
root: Root directory under which ontology artifacts are stored. Each
ontology/version pair receives a dedicated subdirectory.

Examples:
>>> backend = LocalStorageBackend(Path("/tmp/ontologies"))
>>> workspace = backend.prepare_version("hp", "2024-01-01")
>>> (workspace / "original").exists()
True

### `FsspecStorageBackend`

Storage backend that mirrors ontology artifacts to a remote location via fsspec.

Attributes:
fs: fsspec filesystem instance used for remote interactions.
base_path: Root path within the remote filesystem where artefacts live.

Examples:
>>> backend = FsspecStorageBackend("memory://ontologies")
Traceback (most recent call last):
...
ConfigError: fsspec required for remote storage. Install it via 'pip install fsspec'.
