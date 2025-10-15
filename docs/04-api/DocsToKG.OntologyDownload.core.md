# Module: core

Ontology Download Orchestration

This module coordinates resolver planning, document downloading, validation,
and manifest generation for ontology artifacts. It serves as the main entry
point for fetching ontologies from configured sources and producing provenance
metadata that downstream knowledge graph construction can rely upon.

## Functions

### `_read_manifest(manifest_path)`

*No documentation available.*

### `_validate_manifest(manifest)`

*No documentation available.*

### `_write_manifest(manifest_path, manifest)`

*No documentation available.*

### `_build_destination(spec, plan, config)`

*No documentation available.*

### `_ensure_license_allowed(plan, config, spec)`

*No documentation available.*

### `fetch_one(spec)`

Fetch and validate a single ontology described by *spec*.

Args:
spec: Fetch specification outlining resolver and target formats.
config: Optional resolved configuration; defaults to library values.
correlation_id: Identifier that groups log entries for observability.
logger: Optional logger to reuse existing logging infrastructure.
force: When True, ignore existing manifests and re-download artifacts.

Returns:
FetchResult capturing download metadata and produced artifacts.

Raises:
ResolverError: If the resolver cannot produce a viable fetch plan.
OntologyDownloadError: If download or extraction steps fail.
ConfigurationError: If manifest validation or license checks fail.

### `fetch_all(specs)`

Fetch a sequence of ontologies sequentially.

Args:
specs: Iterable of fetch specifications to process.
config: Optional resolved configuration shared across downloads.
logger: Logger used to emit progress and error events.
force: When True, skip manifest reuse and download everything again.

Returns:
List of FetchResult entries corresponding to completed downloads.

Raises:
OntologyDownloadError: Propagated when downloads fail and the pipeline
is configured to stop on error.

### `to_json(self)`

Serialize the manifest to a stable, human-readable JSON string.

Args:
None

Returns:
JSON document encoding the manifest metadata.

### `plan(self, spec, config, logger)`

Return a FetchPlan describing how to obtain the ontology.

Args:
spec: Ontology fetch specification under consideration.
config: Fully resolved configuration containing defaults.
logger: Logger adapter scoped to the current fetch request.

Returns:
Concrete plan containing download URL, headers, and metadata.

### `join(self)`

Build a path relative to the fallback pystow root directory.

Args:
*segments: Path segments appended to the root directory.

Returns:
Path object pointing to the requested cache location.

## Classes

### `OntologyDownloadError`

Base exception for ontology download failures.

Args:
message: Description of the failure encountered.

Examples:
>>> raise OntologyDownloadError("unexpected error")
Traceback (most recent call last):
...
OntologyDownloadError: unexpected error

### `ResolverError`

Raised when resolver planning fails.

Args:
message: Description of the resolver failure.

Examples:
>>> raise ResolverError("resolver unavailable")
Traceback (most recent call last):
...
ResolverError: resolver unavailable

### `ValidationError`

Raised when validation encounters unrecoverable issues.

Args:
message: Human-readable description of the validation failure.

Examples:
>>> raise ValidationError("robot validator crashed")
Traceback (most recent call last):
...
ValidationError: robot validator crashed

### `ConfigurationError`

Raised when configuration or manifest validation fails.

Args:
message: Details about the configuration inconsistency.

Examples:
>>> raise ConfigurationError("manifest missing sha256")
Traceback (most recent call last):
...
ConfigurationError: manifest missing sha256

### `FetchSpec`

Specification describing a single ontology download.

Attributes:
id: Stable identifier for the ontology to fetch.
resolver: Name of the resolver strategy used to locate resources.
extras: Resolver-specific configuration overrides.
target_formats: Normalized ontology formats that should be produced.

Examples:
>>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
>>> spec.resolver
'obo'

### `FetchResult`

Outcome of a single ontology fetch operation.

Attributes:
spec: Fetch specification that initiated the download.
local_path: Path to the downloaded ontology document.
status: Final download status (e.g., `success`, `skipped`).
sha256: SHA-256 digest of the downloaded file.
manifest_path: Path to the generated manifest JSON file.
artifacts: Ancillary files produced during extraction or validation.

Examples:
>>> from pathlib import Path
>>> spec = FetchSpec(id="CHEBI", resolver="obo", extras={}, target_formats=("owl",))
>>> result = FetchResult(
...     spec=spec,
...     local_path=Path("CHEBI.owl"),
...     status="success",
...     sha256="deadbeef",
...     manifest_path=Path("manifest.json"),
...     artifacts=(),
... )
>>> result.status
'success'

### `Manifest`

Provenance information for a downloaded ontology artifact.

Attributes:
id: Ontology identifier recorded in the manifest.
resolver: Resolver used to retrieve the ontology.
url: Final URL from which the ontology was fetched.
filename: Local filename of the downloaded artifact.
version: Resolver-reported ontology version, if available.
license: License identifier associated with the ontology.
status: Result status reported by the downloader.
sha256: Hash of the downloaded artifact for integrity checking.
etag: HTTP ETag returned by the upstream server, when provided.
last_modified: Upstream last-modified timestamp, if supplied.
downloaded_at: UTC timestamp of the completed download.
target_formats: Desired conversion targets for normalization.
validation: Mapping of validator names to their results.
artifacts: Additional file paths generated during processing.

Examples:
>>> manifest = Manifest(
...     id="CHEBI",
...     resolver="obo",
...     url="https://example.org/chebi.owl",
...     filename="chebi.owl",
...     version=None,
...     license="CC-BY",
...     status="success",
...     sha256="deadbeef",
...     etag=None,
...     last_modified=None,
...     downloaded_at="2024-01-01T00:00:00Z",
...     target_formats=("owl",),
...     validation={},
...     artifacts=(),
... )
>>> manifest.resolver
'obo'

### `Resolver`

Protocol describing resolver planning behaviour.

Attributes:
None

Examples:
>>> import logging
>>> spec = FetchSpec(id="CHEBI", resolver="dummy", extras={}, target_formats=("owl",))
>>> class DummyResolver:
...     def plan(self, spec, config, logger):
...         return FetchPlan(
...             url="https://example.org/chebi.owl",
...             headers={},
...             filename_hint="chebi.owl",
...             version="v1",
...             license="CC-BY",
...             media_type="application/rdf+xml",
...         )
...
>>> plan = DummyResolver().plan(spec, ResolvedConfig.from_defaults(), logging.getLogger("test"))
>>> plan.url
'https://example.org/chebi.owl'

### `_PystowFallback`

Minimal pystow replacement used when the dependency is absent.

Attributes:
_root: Base directory used to emulate pystow's storage root.

Examples:
>>> fallback = _PystowFallback()
>>> isinstance(fallback.join("ontology"), Path)
True
