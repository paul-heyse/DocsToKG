# 1. Module: core

The orchestrator previously exported as ``DocsToKG.OntologyDownload.core`` is folded
into ``DocsToKG.OntologyDownload.ontology_download``.

This module plans resolver candidates, enforces license allowlists, performs
fallback-aware downloads, orchestrates streaming normalization, and writes
schema-validated manifests with deterministic fingerprints. It aligns with the
refactored ontology download specification by recording resolver attempt
chains, honoring CLI concurrency overrides, and supporting batch operations for
planning and pull commands.

## 1. Functions

### `get_manifest_schema()`

Return a deep copy of the manifest JSON Schema definition.

Args:
None

Returns:
Dictionary describing the manifest JSON Schema.

### `validate_manifest_dict(payload)`

Validate manifest payload against the JSON Schema definition.

Args:
payload: Manifest dictionary loaded from JSON.
source: Optional filesystem path for contextual error reporting.

Returns:
None

Raises:
ConfigurationError: If validation fails.

### `_coerce_datetime(value)`

Return timezone-aware datetime parsed from HTTP or ISO timestamp.

### `_normalize_timestamp(value)`

Return canonical ISO8601 string for HTTP timestamp headers.

### `_populate_plan_metadata(planned, config, adapter)`

Augment planned fetch with HTTP metadata when available.

### `_read_manifest(manifest_path)`

Return previously recorded manifest data if a valid JSON file exists.

Args:
manifest_path: Filesystem path where the manifest is stored.

Returns:
Parsed manifest dictionary when available and valid, otherwise ``None``.

### `_validate_manifest(manifest)`

Check that a manifest instance satisfies structural and type requirements.

Args:
manifest: Manifest produced after a download completes.

Raises:
ConfigurationError: If required fields are missing or contain invalid types.

### `_parse_last_modified(value)`

Return a timezone-aware datetime parsed from HTTP date headers.

### `_fetch_last_modified(plan, config, logger)`

Probe the upstream plan URL for a Last-Modified header.

### `_write_manifest(manifest_path, manifest)`

Persist a validated manifest to disk as JSON.

Args:
manifest_path: Destination path for the manifest file.
manifest: Manifest describing the downloaded ontology artifact.

### `_build_destination(spec, plan, config)`

Determine the output directory and filename for a download.

Args:
spec: Fetch specification identifying the ontology.
plan: Resolver plan containing URL metadata and optional hints.
config: Resolved configuration with storage layout parameters.

Returns:
Tuple containing the target file path, resolved version, and base directory.

### `_ensure_license_allowed(plan, config, spec)`

Confirm the ontology license is present in the configured allow list.

Args:
plan: Resolver plan returned for the ontology.
config: Resolved configuration containing accepted licenses.
spec: Fetch specification for contextual error reporting.

Raises:
ConfigurationError: If the plan's license is not permitted.

### `_resolver_candidates(spec, config)`

*No documentation available.*

### `_resolve_plan_with_fallback(spec, config, adapter)`

*No documentation available.*

### `fetch_one(spec)`

Fetch, validate, and persist a single ontology described by *spec*.

Args:
spec: Fetch specification outlining resolver selection and target formats.
config: Optional resolved configuration supplying defaults and limits.
correlation_id: Identifier used to correlate structured log entries.
logger: Logger instance reused for download and validation telemetry.
force: When True, ignore cached manifests and re-download artefacts.

Returns:
FetchResult capturing download status, SHA-256 hashes, and manifest path.

Raises:
ResolverError: If all resolvers fail to produce a viable FetchPlan.
OntologyDownloadError: If download, extraction, or validation fails.
ConfigurationError: If licence checks or manifest validation fail.

### `plan_one(spec)`

Return a resolver plan for a single ontology without performing downloads.

Args:
spec: Fetch specification describing the ontology to plan.
config: Optional resolved configuration providing defaults and limits.
correlation_id: Correlation identifier reused for logging context.
logger: Logger instance used to emit resolver telemetry.

Returns:
PlannedFetch containing the normalized spec, resolver name, and plan.

Raises:
ResolverError: If all resolvers fail to produce a plan for ``spec``.
ConfigurationError: If licence checks reject the planned ontology.

### `plan_all(specs)`

Return resolver plans for a collection of ontologies.

Args:
specs: Iterable of fetch specifications to resolve.
config: Optional resolved configuration reused across plans.
logger: Logger instance used for annotation-aware logging.
since: Optional cutoff date; plans older than this timestamp are filtered out.

Returns:
List of PlannedFetch entries describing each ontology plan.

Raises:
ResolverError: Propagated when fallback planning fails for any spec.
ConfigurationError: When licence enforcement rejects a planned ontology.

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

### `to_dict(self)`

Return a JSON-serializable dictionary for the manifest.

Args:
None

Returns:
Dictionary representing the manifest payload.

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

### `_add(name)`

*No documentation available.*

## 2. Classes

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
schema_version: Manifest schema version identifier.
id: Ontology identifier recorded in the manifest.
resolver: Resolver used to retrieve the ontology.
url: Final URL from which the ontology was fetched.
filename: Local filename of the downloaded artifact.
version: Resolver-reported ontology version, if available.
license: License identifier associated with the ontology.
status: Result status reported by the downloader.
sha256: Hash of the downloaded artifact for integrity checking.
normalized_sha256: Hash of the canonical normalized TTL output.
fingerprint: Composite fingerprint combining key provenance values.
etag: HTTP ETag returned by the upstream server, when provided.
last_modified: Upstream last-modified timestamp, if supplied.
downloaded_at: UTC timestamp of the completed download.
target_formats: Desired conversion targets for normalization.
validation: Mapping of validator names to their results.
artifacts: Additional file paths generated during processing.
resolver_attempts: Ordered record of resolver attempts during download.

Examples:
>>> manifest = Manifest(
...     schema_version="1.0",
...     id="CHEBI",
...     resolver="obo",
...     url="https://example.org/chebi.owl",
...     filename="chebi.owl",
...     version=None,
...     license="CC-BY",
...     status="success",
...     sha256="deadbeef",
...     normalized_sha256=None,
...     fingerprint=None,
...     etag=None,
...     last_modified=None,
...     downloaded_at="2024-01-01T00:00:00Z",
...     target_formats=("owl",),
...     validation={},
...     artifacts=(),
...     resolver_attempts=(),
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

### `ResolverCandidate`

Resolver plan captured for download-time fallback.

Attributes:
resolver: Name of the resolver that produced the plan.
plan: Concrete :class:`FetchPlan` describing how to fetch the ontology.

Examples:
>>> candidate = ResolverCandidate(
...     resolver="obo",
...     plan=FetchPlan(
...         url="https://example.org/hp.owl",
...         headers={},
...         filename_hint=None,
...         version="2024-01-01",
...         license="CC-BY",
...         media_type="application/rdf+xml",
...         service="obo",
...     ),
... )
>>> candidate.resolver
'obo'

### `PlannedFetch`

Plan describing how an ontology would be fetched without side effects.

Attributes:
spec: Original fetch specification provided by the caller.
resolver: Name of the resolver selected to satisfy the plan.
plan: Concrete :class:`FetchPlan` generated by the resolver.
candidates: Ordered list of resolver candidates available for fallback.

Examples:
>>> fetch_plan = PlannedFetch(
...     spec=FetchSpec(id="hp", resolver="obo", extras={}, target_formats=("owl",)),
...     resolver="obo",
...     plan=FetchPlan(
...         url="https://example.org/hp.owl",
...         headers={},
...         filename_hint="hp.owl",
...         version="2024-01-01",
...         license="CC-BY-4.0",
...         media_type="application/rdf+xml",
...     ),
...     candidates=(
...         ResolverCandidate(
...             resolver="obo",
...             plan=FetchPlan(
...                 url="https://example.org/hp.owl",
...                 headers={},
...                 filename_hint="hp.owl",
...                 version="2024-01-01",
...                 license="CC-BY-4.0",
...                 media_type="application/rdf+xml",
...             ),
...         ),
...     ),
... )
>>> fetch_plan.resolver
'obo'
