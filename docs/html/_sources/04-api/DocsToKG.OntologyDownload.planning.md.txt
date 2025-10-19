# 1. Module: planning

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.planning``.

## 1. Overview

Download planning and orchestration helpers for ontology fetching.

## 2. Functions

### `_log_with_extra(logger, level, message, extra)`

Log ``message`` with structured ``extra`` supporting LoggerAdapters.

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

### `_make_fetch_spec(raw_spec, defaults)`

Create a fetch specification from raw configuration and defaults.

### `merge_defaults(raw_spec, defaults)`

Merge user-provided specification with defaults to create a fetch spec.

### `_cancel_pending_futures(futures)`

Cancel any futures that are still pending execution.

### `_executor_is_shutting_down(executor)`

Return ``True`` when *executor* has begun shutdown processing.

### `_shutdown_executor_nowait(executor)`

Request non-blocking shutdown with cancellation for *executor*.

### `parse_http_datetime(value)`

Parse HTTP ``Last-Modified`` style timestamps into UTC datetimes.

Args:
value: Timestamp string from HTTP headers such as ``Last-Modified``.

Returns:
Optional[datetime]: Normalized UTC datetime when parsing succeeds.

Raises:
None: Parsing failures are converted into a ``None`` return value.

### `parse_iso_datetime(value)`

Parse ISO-8601 timestamps into timezone-aware UTC datetimes.

Args:
value: ISO-8601 formatted timestamp string.

Returns:
Optional[datetime]: Normalized UTC datetime when parsing succeeds.

Raises:
None: Invalid values return ``None`` instead of raising.

### `parse_version_timestamp(value)`

Parse version strings or manifest timestamps into UTC datetimes.

Args:
value: Version identifier or timestamp string to normalize.

Returns:
Optional[datetime]: Parsed UTC datetime when the input matches supported formats.

Raises:
None: All parsing failures result in ``None``.

### `infer_version_timestamp(value)`

Infer a timestamp from resolver version identifiers.

Args:
value: Resolver version string containing date-like fragments.

Returns:
Optional[datetime]: Parsed UTC datetime when the value contains recoverable dates.

Raises:
None: Returns ``None`` instead of raising on unparseable inputs.

### `_coerce_datetime(value)`

Return timezone-aware datetime parsed from HTTP or ISO timestamp.

### `_normalize_timestamp(value)`

Return canonical ISO8601 string for HTTP timestamp headers.

### `_canonical_media_type(value)`

Return a normalized MIME type without parameters.

### `_select_validators(media_type)`

Return validator names appropriate for ``media_type``.

### `planner_http_probe()`

Issue a polite planner probe using shared networking primitives.

Call graph: :func:`plan_one`/ :func:`plan_all` → :func:`_populate_plan_metadata` /
:func:`_fetch_last_modified` → :func:`planner_http_probe` → :func:`SESSION_POOL.lease`.

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

### `_atomic_write_text(path, content)`

Atomically replace ``path`` with ``content`` to avoid partial writes.

### `_atomic_write_json(path, payload)`

Serialize ``payload`` to JSON and atomically persist it to ``path``.

### `_write_manifest(manifest_path, manifest)`

Persist a validated manifest to disk as JSON.

Args:
manifest_path: Destination path for the manifest file.
manifest: Manifest describing the downloaded ontology artifact.

### `_append_index_entry(ontology_dir, entry)`

Append or update the ontology-level ``index.json`` with ``entry`` safely.

### `_ontology_index_lock(ontology_dir)`

Serialize ontology index mutations across versions for the same ontology.

### `_mirror_to_cas_if_enabled()`

Mirror ``destination`` into the content-addressable cache when enabled.

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
spec: Ontology fetch specification describing sources and formats.
config: Optional resolved configuration overriding global defaults.
correlation_id: Correlation identifier for structured logging.
logger: Optional logger to reuse instead of configuring a new one.
force: When ``True``, bypass local cache checks and redownload artifacts.
cancellation_token: Optional token for cooperative cancellation.

Returns:
FetchResult: Structured result containing manifest metadata and resolver attempts.

Raises:
ResolverError: If all resolver candidates fail to retrieve the ontology.

### `plan_one(spec)`

Return a resolver plan for a single ontology without performing downloads.

Args:
spec: Fetch specification describing the ontology to plan.
config: Optional resolved configuration providing defaults and limits.
correlation_id: Correlation identifier reused for logging context.
logger: Logger instance used to emit resolver telemetry.
cancellation_token: Optional token for cooperative cancellation.

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
total: Optional total number of specifications, used for progress metadata when
the iterable cannot be sized cheaply.
cancellation_token_group: Optional group of cancellation tokens for cooperative cancellation.

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
total: Optional total number of specifications for progress metadata when
the iterable cannot be cheaply materialised.
cancellation_token_group: Optional group of cancellation tokens for cooperative cancellation.

Returns:
List of FetchResult entries corresponding to completed downloads.

Raises:
OntologyDownloadError: Propagated when downloads fail and the pipeline
is configured to stop on error.

### `_safe_lock_component(value)`

Return a filesystem-safe token for lock filenames.

### `_version_lock(ontology_id, version)`

Acquire an inter-process lock for a specific ontology version.

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

### `_add_candidate(candidate)`

*No documentation available.*

### `_add(name)`

*No documentation available.*

### `_submit(spec, index)`

*No documentation available.*

### `_submit(spec, index)`

*No documentation available.*

### `_issue(current_method)`

*No documentation available.*

### `_execute_candidate()`

*No documentation available.*

### `_perform_once()`

*No documentation available.*

### `_on_retry(attempt, exc, delay)`

*No documentation available.*

## 3. Classes

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

### `BatchPlanningError`

Raised when ontology planning aborts after a failure.

### `BatchFetchError`

Raised when ontology downloads abort after a failure.

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
content_type: MIME type reported by upstream servers when available.
content_length: Content-Length reported by upstream servers when available.
source_media_type_label: Friendly label describing the source media type.
streaming_content_sha256: Streaming canonical content hash when available.
streaming_prefix_sha256: Hash of Turtle prefix header when available.
downloaded_at: UTC timestamp of the completed download.
target_formats: Desired conversion targets for normalization.
validation: Mapping of validator names to their results.
artifacts: Additional file paths generated during processing.
resolver_attempts: Ordered record of resolver attempts during download.
expected_checksum: Optional checksum metadata enforced for the download.

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
...     content_type=None,
...     content_length=None,
...     source_media_type_label=None,
...     downloaded_at="2024-01-01T00:00:00Z",
...     target_formats=("owl",),
...     validation={},
...     artifacts=(),
...     resolver_attempts=(),
... )
>>> manifest.resolver
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

### `PlannerProbeResult`

Normalized response metadata produced by planner HTTP probes.
