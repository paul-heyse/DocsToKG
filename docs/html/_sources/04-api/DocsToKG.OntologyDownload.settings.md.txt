# 1. Module: settings

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.settings``.

## 1. Overview

Configuration, optional dependency, and storage utilities for ontology downloads.

## 2. Functions

### `ensure_python_version()`

Ensure the interpreter meets the minimum supported Python version.

### `_coerce_sequence(value)`

*No documentation available.*

### `parse_rate_limit_to_rps(limit_str)`

Convert a rate limit expression into requests-per-second.

### `get_env_overrides()`

Return environment-derived overrides as stringified key/value pairs.

### `_apply_env_overrides(defaults)`

Mutate ``defaults`` in-place using values from :class:`EnvironmentOverrides`.

### `build_resolved_config(raw_config)`

Materialise a :class:`ResolvedConfig` from a raw mapping loaded from disk.

### `_validate_schema(raw, config)`

*No documentation available.*

### `load_raw_yaml(config_path)`

Read a YAML configuration file and return its top-level mapping.

### `load_config(config_path)`

Load, validate, and resolve configuration suitable for execution.

### `validate_config(config_path)`

Load a configuration solely for validation feedback.

### `_create_stub_module(name, attrs)`

*No documentation available.*

### `_create_stub_bnode(value)`

*No documentation available.*

### `_create_stub_literal(value)`

*No documentation available.*

### `_create_stub_uri(value)`

*No documentation available.*

### `_import_module(name)`

*No documentation available.*

### `_create_pystow_stub(root)`

*No documentation available.*

### `_create_rdflib_stub()`

*No documentation available.*

### `_create_pronto_stub()`

*No documentation available.*

### `_create_owlready_stub()`

*No documentation available.*

### `get_pystow()`

Return the ``pystow`` module, supplying a stub when unavailable.

### `get_rdflib()`

Return the ``rdflib`` module, supplying a stub when unavailable.

### `get_pronto()`

Return the ``pronto`` module, supplying a stub when unavailable.

### `get_owlready2()`

Return the ``owlready2`` module, supplying a stub when unavailable.

### `_safe_identifiers(ontology_id, version)`

Return identifiers sanitised for filesystem usage.

### `_directory_size(path)`

Return the total size in bytes for files rooted at *path*.

### `get_storage_backend()`

Instantiate the storage backend based on environment configuration.

### `validate_level(cls, value)`

Normalise logging levels and ensure they match the supported set.

### `validate_rate_limits(cls, value)`

Ensure per-resolver rate limits follow the supported syntax.

### `validate_allowed_ports(cls, value)`

Normalise and validate the optional port allowlist from configuration.

Args:
value: Collection of port numbers supplied in the settings payload.

Returns:
Sanitised list of ports preserving input order when provided, otherwise ``None``.

Raises:
ValueError: If any port is not an integer or falls outside the TCP range 1-65535.

### `rate_limit_per_second(self)`

Return the configured per-host rate limit in requests per second.

### `parse_service_rate_limit(self, service)`

Return service-specific rate limits expressed as requests per second.

### `allowed_port_set(self)`

Return the union of default ports and user-configured allowances.

### `normalized_allowed_hosts(self)`

Split allowed host list into exact domains, wildcard suffixes, and per-host port allowances.

### `polite_http_headers(self)`

Compute polite HTTP headers for outbound resolver requests.

### `validate_prefer_source(cls, value)`

Ensure preferred resolvers belong to the supported resolver set.

### `from_defaults(cls)`

Construct a resolved configuration populated with default values only.

### `__getitem__(self, key)`

*No documentation available.*

### `bind(self, prefix, namespace)`

Register a namespace binding in the lightweight stub manager.

### `namespaces(self)`

Yield currently registered ``(prefix, namespace)`` pairs.

### `parse(self, source, format)`

Parse a Turtle-like text file into an in-memory triple list.

### `serialize(self, destination, format)`

Serialise parsed triples to the supplied destination.

### `add(self, triple)`

Append a triple to the stub graph, mirroring rdflib behaviour.

### `bind(self, prefix, namespace)`

Register a namespace binding within the stub graph.

### `namespaces(self)`

Yield namespace bindings previously registered via :meth:`bind`.

### `__len__(self)`

*No documentation available.*

### `__iter__(self)`

*No documentation available.*

### `join()`

Mimic :func:`pystow.join` by joining segments onto the stub root.

### `get_ontology(iri)`

Return a stub ontology instance for the provided IRI.

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

### `mirror_cas_artifact(self, algorithm, digest, source)`

Mirror ``source`` into a content-addressable cache and return its path.

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

### `mirror_cas_artifact(self, algorithm, digest, source)`

Copy ``source`` into the content-addressable cache.

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

### `mirror_cas_artifact(self, algorithm, digest, source)`

Mirror CAS artefact locally and to remote storage.

### `terms(self)`

Return a deterministic collection of ontology term identifiers.

### `dump(self, destination, format)`

Write minimal ontology contents to ``destination`` for tests.

### `load(self)`

Provide fluent API parity with owlready2 ontologies.

### `classes(self)`

Return example ontology classes for tests and fallbacks.

## 3. Classes

### `LoggingConfiguration`

Logging-related configuration for ontology downloads.

### `ValidationConfig`

Settings controlling ontology validation throughput and limits.

### `DownloadConfiguration`

HTTP download, retry, and politeness settings for resolvers.

### `DefaultsConfig`

Composite configuration applied when no per-spec overrides exist.

### `ResolvedConfig`

Materialised configuration combining defaults and fetch specifications.

### `EnvironmentOverrides`

Pydantic settings model exposing environment-derived overrides.

### `_StubNamespace`

*No documentation available.*

### `_StubNamespaceManager`

*No documentation available.*

### `_StubGraph`

*No documentation available.*

### `StorageBackend`

Protocol describing storage operations required by the pipeline.

### `LocalStorageBackend`

Storage backend that keeps artefacts on the local filesystem.

### `FsspecStorageBackend`

Hybrid backend that mirrors artefacts to an fsspec location.

### `_StubOntology`

*No documentation available.*

### `_StubOntology`

*No documentation available.*
