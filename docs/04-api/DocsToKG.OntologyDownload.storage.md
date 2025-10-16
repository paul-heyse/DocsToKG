# 1. Module: storage

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.storage``.

## 1. Overview

Storage backend implementations for the ontology downloader.

## 2. Functions

### `_safe_identifiers(ontology_id, version)`

Return identifiers sanitised for filesystem usage.

### `_directory_size(path)`

Return the total size in bytes for files rooted at *path*.

### `get_storage_backend()`

Instantiate the storage backend based on environment configuration.

### `prepare_version(self, ontology_id, version)`

Return a workspace for the given ontology/version combination.

### `ensure_local_version(self, ontology_id, version)`

Ensure a local directory exists for the requested ontology version.

### `available_versions(self, ontology_id)`

Return sorted version identifiers available for *ontology_id*.

### `available_ontologies(self)`

Return sorted ontology identifiers known to the backend.

### `finalize_version(self, ontology_id, version, local_dir)`

Persist processed artefacts for *ontology_id*/*version*.

### `version_path(self, ontology_id, version)`

Return the canonical path for *ontology_id*/*version*.

### `delete_version(self, ontology_id, version)`

Delete stored data for *ontology_id*/*version*, returning bytes reclaimed.

### `set_latest_version(self, ontology_id, version)`

Record the latest processed version for *ontology_id*.

### `_version_dir(self, ontology_id, version)`

*No documentation available.*

### `prepare_version(self, ontology_id, version)`

Create a workspace for ``ontology_id``/``version`` with required subdirs.

### `ensure_local_version(self, ontology_id, version)`

Ensure the version directory exists and return its path.

### `available_versions(self, ontology_id)`

List version identifiers currently stored for ``ontology_id``.

### `available_ontologies(self)`

List ontology identifiers known to the local backend.

### `finalize_version(self, ontology_id, version, local_dir)`

Hook for subclasses; local backend writes occur in-place already.

### `version_path(self, ontology_id, version)`

Return the filesystem path holding ``ontology_id``/``version`` data.

### `delete_version(self, ontology_id, version)`

Remove stored data for ``ontology_id``/``version`` and return bytes reclaimed.

### `set_latest_version(self, ontology_id, version)`

Update symbolic links/markers to highlight the latest processed version.

### `_remote_version_path(self, ontology_id, version)`

Return the remote storage path for ``ontology_id``/``version``.

### `available_versions(self, ontology_id)`

Combine local and remote version identifiers for ``ontology_id``.

### `available_ontologies(self)`

Return the union of ontology ids present locally and in remote storage.

### `finalize_version(self, ontology_id, version, local_dir)`

Mirror processed artefacts to remote storage after local completion.

### `delete_version(self, ontology_id, version)`

Remove local and remote artefacts for ``ontology_id``/``version``.

## 3. Classes

### `StorageBackend`

Protocol describing storage operations required by the pipeline.

### `LocalStorageBackend`

Storage backend that keeps artefacts on the local filesystem.

### `FsspecStorageBackend`

Hybrid backend that mirrors artefacts to an fsspec location.
