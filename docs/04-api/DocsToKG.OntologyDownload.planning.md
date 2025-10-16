# 1. Module: planning

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.planning``.

## 1. Overview

Download planning and orchestration helpers for ontology fetching.

## 2. Functions

### `_normalize_algorithm(algorithm)`

*No documentation available.*

### `_normalize_checksum(algorithm, value)`

*No documentation available.*

### `_checksum_from_extras(extras)`

*No documentation available.*

### `_checksum_url_from_extras(extras)`

*No documentation available.*

### `_extract_checksum_from_text(text)`

*No documentation available.*

### `_fetch_checksum_from_url()`

*No documentation available.*

### `_resolve_expected_checksum()`

Determine the expected checksum string passed to the downloader.

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

### `_populate_plan_metadata(planned, config, adapter)`

Augment planned fetch with HTTP metadata when available.

### `_migrate_manifest_inplace(payload)`

Upgrade manifests created with older schema versions in place.

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

### `_append_index_entry(ontology_dir, entry)`

Append or update the ontology-level ``index.json`` with ``entry``.

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

### `_safe_lock_component(value)`

Return a filesystem-safe token for lock filenames.

### `_version_lock(ontology_id, version)`

Acquire an inter-process lock for a specific ontology version.

### `normalize_license_to_spdx(value)`

Normalize common license strings to canonical SPDX identifiers.

Resolver metadata frequently reports informal variants such as ``CC BY 4.0``;
converting to SPDX ensures allowlist comparisons remain consistent.

Args:
value: Raw license string returned by a resolver (may be ``None``).

Returns:
Canonical SPDX identifier when a mapping is known, otherwise the
cleaned original value or ``None`` when the input is empty.

### `_parse_checksum_extra(value)`

Normalize checksum extras to ``(algorithm, value)`` tuples.

### `_parse_checksum_url_extra(value)`

Normalize checksum URL extras to ``(url, algorithm)`` tuples.

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

### `_add_candidate(candidate)`

*No documentation available.*

### `_add(name)`

*No documentation available.*

### `_execute_with_retry(self, func)`

Run a callable with retry semantics tailored for resolver APIs.

Args:
func: Callable performing the API request.
config: Resolved configuration containing retry and timeout settings.
logger: Logger adapter used to record retry attempts.
name: Human-friendly resolver name used in log messages.

Returns:
Result returned by the supplied callable.

Raises:
ResolverError: When retry limits are exceeded or HTTP errors occur.
UserConfigError: When upstream services reject credentials.

### `_extract_correlation_id(self, logger)`

Return the correlation id from a logger adapter when available.

Args:
logger: Logger or adapter potentially carrying an ``extra`` dictionary.

Returns:
Correlation identifier string when present, otherwise ``None``.

### `_build_polite_headers(self, config, logger)`

Create polite headers derived from configuration and logger context.

Args:
config: Resolved configuration providing HTTP header defaults.
logger: Logger adapter whose correlation id is propagated to headers.

Returns:
Dictionary of polite header values ready to attach to HTTP sessions.

### `_apply_headers_to_session(session, headers)`

Apply polite headers to a client session when supported.

Args:
session: HTTP client or session object whose ``headers`` may be updated.
headers: Mapping of header names to polite values to merge.

Returns:
None

### `_build_plan(self)`

Construct a ``FetchPlan`` from resolver components.

Args:
url: Canonical download URL for the ontology.
headers: HTTP headers required when issuing the download.
filename_hint: Suggested filename derived from resolver metadata.
version: Version string reported by the resolver.
license: License identifier reported by the resolver.
media_type: MIME type associated with the ontology.
service: Logical service identifier used for rate limiting.

Returns:
FetchPlan capturing resolver metadata with a security-validated URL.

### `plan(self, spec, config, logger)`

Resolve download URLs using Bioregistry-provided endpoints.

Args:
spec: Fetch specification describing the ontology to download.
config: Global configuration with retry and timeout settings.
logger: Logger adapter used to emit planning telemetry.

Returns:
FetchPlan pointing to the preferred download URL.

Raises:
ResolverError: If no download URL can be derived.
UserConfigError: When required Bioregistry helpers are unavailable.

### `plan(self, spec, config, logger)`

Discover download locations via the OLS API.

Args:
spec: Fetch specification containing ontology identifiers and extras.
config: Resolved configuration that provides retry policies.
logger: Logger adapter used for planner progress messages.

Returns:
FetchPlan describing the download URL, headers, and metadata.

Raises:
UserConfigError: When the API rejects credentials.
ResolverError: When no download URLs can be resolved.

### `_load_api_key(self)`

Load the BioPortal API key from disk when available.

Args:
self: Resolver instance requesting the API key.

Returns:
Optional[str]: API key string stripped of whitespace, or ``None`` when missing.

### `plan(self, spec, config, logger)`

Resolve BioPortal download URLs and authorization headers.

Args:
spec: Fetch specification with optional API extras like acronyms.
config: Resolved configuration that governs HTTP retry behaviour.
logger: Logger adapter for structured telemetry.

Returns:
FetchPlan containing the resolved download URL and headers.

Raises:
UserConfigError: If authentication fails.
ResolverError: If no download link is available.

### `_fetch_metadata(self, uri, timeout)`

*No documentation available.*

### `_iter_dicts(payload)`

*No documentation available.*

### `plan(self, spec, config, logger)`

Discover download metadata from the LOV API.

Args:
spec: Fetch specification providing ontology identifier and extras.
config: Resolved configuration supplying timeout and header defaults.
logger: Logger adapter used to emit planning telemetry.

Returns:
FetchPlan describing the resolved download URL and metadata.

Raises:
UserConfigError: If required resolver metadata is missing.
ResolverError: If the LOV API does not provide a download URL.

### `plan(self, spec, config, logger)`

Return a fetch plan for explicitly provided SKOS URLs.

Args:
spec: Fetch specification containing the `extras.url` field.
config: Resolved configuration (unused, included for API symmetry).
logger: Logger adapter used to report resolved URL information.

Returns:
FetchPlan with the provided URL and appropriate media type.

Raises:
UserConfigError: If the specification omits the required URL.

### `plan(self, spec, config, logger)`

Return a fetch plan using the direct URL provided in ``spec.extras``.

Args:
spec: Fetch specification containing the upstream download details.
config: Resolved configuration (unused, provided for interface parity).
logger: Logger adapter used to record telemetry.

Returns:
FetchPlan referencing the explicit URL.

Raises:
UserConfigError: If the specification omits the required URL or provides invalid extras.

### `plan(self, spec, config, logger)`

Return a fetch plan for XBRL ZIP archives provided via extras.

Args:
spec: Fetch specification containing the upstream download URL.
config: Resolved configuration (unused, included for API compatibility).
logger: Logger adapter for structured observability.

Returns:
FetchPlan referencing the specified ZIP archive.

Raises:
UserConfigError: If the specification omits the required URL.

### `plan(self, spec, config, logger)`

Return a fetch plan pointing to Ontobee-managed PURLs.

Args:
spec: Fetch specification describing the ontology identifier and preferred formats.
config: Resolved configuration (unused beyond interface compatibility).
logger: Logger adapter for structured telemetry.

Returns:
FetchPlan pointing to an Ontobee-hosted download URL.

Raises:
UserConfigError: If the ontology identifier is invalid.

### `_execute_candidate()`

*No documentation available.*

### `_retryable(exc)`

*No documentation available.*

### `_on_retry(attempt, exc, sleep_time)`

*No documentation available.*

### `_invoke()`

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

### `FetchPlan`

Concrete plan output from a resolver.

Attributes:
url: Final URL from which to download the ontology document.
headers: HTTP headers required by the upstream service.
filename_hint: Optional filename recommended by the resolver.
version: Version identifier derived from resolver metadata.
license: License reported for the ontology.
media_type: MIME type of the artifact when known.
service: Logical service identifier used for rate limiting.
checksum: Optional checksum value supplied by resolver.
checksum_algorithm: Hash algorithm associated with ``checksum``.
checksum_url: URL where a checksum file can be retrieved when provided.

Examples:
>>> plan = FetchPlan(
...     url="https://example.org/ontology.owl",
...     headers={"Accept": "application/rdf+xml"},
...     filename_hint="ontology.owl",
...     version="2024-01-01",
...     license="CC-BY",
...     media_type="application/rdf+xml",
...     service="ols",
... )
>>> plan.service
'ols'

### `BaseResolver`

Shared helpers for resolver implementations.

Provides polite header construction, retry orchestration, and metadata
normalization utilities shared across concrete resolver classes.

Attributes:
None

Examples:
>>> class DemoResolver(BaseResolver):
...     def plan(self, spec, config, logger):
...         return self._build_plan(url="https://example.org/demo.owl")
...
>>> demo = DemoResolver()
>>> isinstance(demo._build_plan(url="https://example.org").url, str)
True

### `OBOResolver`

Resolve ontologies hosted on the OBO Library using Bioregistry helpers.

Attributes:
None

Examples:
>>> resolver = OBOResolver()
>>> callable(getattr(resolver, "plan"))
True

### `OLSResolver`

Resolve ontologies from the Ontology Lookup Service (OLS4).

Attributes:
client: OLS client instance used to perform API calls.
credentials_path: Path where the API token is expected.

Examples:
>>> resolver = OLSResolver()
>>> resolver.credentials_path.name.endswith(".txt")
True

### `BioPortalResolver`

Resolve ontologies using the BioPortal (OntoPortal) API.

Attributes:
client: BioPortal client used to query ontology metadata.
api_key_path: Path on disk containing the API key.

Examples:
>>> resolver = BioPortalResolver()
>>> resolver.api_key_path.suffix
'.txt'

### `LOVResolver`

Resolve vocabularies from Linked Open Vocabularies (LOV).

Queries the LOV API, normalises metadata fields, and returns Turtle
download plans enriched with service identifiers for rate limiting.

Attributes:
API_ROOT: Base URL for the LOV API endpoints.
session: Requests session used to execute API calls.

Examples:
>>> resolver = LOVResolver()
>>> isinstance(resolver.session, requests.Session)
True

### `SKOSResolver`

Resolver for direct SKOS/RDF URLs.

Attributes:
None

Examples:
>>> resolver = SKOSResolver()
>>> callable(getattr(resolver, "plan"))
True

### `DirectResolver`

Resolver that consumes explicit URLs supplied via ``spec.extras``.

### `XBRLResolver`

Resolver for XBRL taxonomy packages.

Attributes:
None

Examples:
>>> resolver = XBRLResolver()
>>> callable(getattr(resolver, "plan"))
True

### `OntobeeResolver`

Resolver that constructs Ontobee-backed PURLs for OBO ontologies.

Provides a lightweight fallback resolver that constructs deterministic
PURLs for OBO prefixes when primary resolvers fail.

Attributes:
_FORMAT_MAP: Mapping of preferred formats to extensions and media types.

Examples:
>>> resolver = OntobeeResolver()
>>> resolver._FORMAT_MAP['owl'][0]
'owl'
