# 1. Module: resolvers

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.resolvers``.

## 1. Overview

Ontology resolver implementations.

This module defines the strategies that translate planner specifications into
actionable fetch plans. Each resolver applies polite headers, unified retry
logic, SPDX-normalized licensing, and service-specific rate limits while
participating in the automatic fallback chains described in the ontology
download refactor. New resolvers can be registered through the ``RESOLVERS`` map.

## 2. Functions

### `normalize_license_to_spdx(value)`

Normalize common license strings to canonical SPDX identifiers.

Resolver metadata frequently reports informal variants such as ``CC BY 4.0``;
converting to SPDX ensures allowlist comparisons remain consistent.

Args:
value: Raw license string returned by a resolver (may be ``None``).

Returns:
Canonical SPDX identifier when a mapping is known, otherwise the
cleaned original value or ``None`` when the input is empty.

### `_get_service_bucket(service, config)`

Return a token bucket for resolver API requests respecting rate limits.

### `_load_resolver_plugins(logger)`

Discover resolver plugins registered via Python entry points.

### `_load_resolver_plugins(logger)`

Discover resolver plugins registered via Python entry points.

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
ConfigError: When retry limits are exceeded or HTTP errors occur.

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
FetchPlan capturing resolver metadata.

### `plan(self, spec, config, logger)`

Resolve download URLs using Bioregistry-provided endpoints.

Args:
spec: Fetch specification describing the ontology to download.
config: Global configuration with retry and timeout settings.
logger: Logger adapter used to emit planning telemetry.

Returns:
FetchPlan pointing to the preferred download URL.

Raises:
ConfigError: If no download URL can be derived.

### `plan(self, spec, config, logger)`

Discover download locations via the OLS API.

Args:
spec: Fetch specification containing ontology identifiers and extras.
config: Resolved configuration that provides retry policies.
logger: Logger adapter used for planner progress messages.

Returns:
FetchPlan describing the download URL, headers, and metadata.

Raises:
ConfigError: When the API rejects credentials or yields no URLs.

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
ConfigError: If authentication fails or no download link is available.

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
ConfigError: If required metadata is missing or the LOV API fails.

### `plan(self, spec, config, logger)`

Return a fetch plan for explicitly provided SKOS URLs.

Args:
spec: Fetch specification containing the `extras.url` field.
config: Resolved configuration (unused, included for API symmetry).
logger: Logger adapter used to report resolved URL information.

Returns:
FetchPlan with the provided URL and appropriate media type.

Raises:
ConfigError: If the specification omits the required URL.

### `plan(self, spec, config, logger)`

Return a fetch plan for XBRL ZIP archives provided via extras.

Args:
spec: Fetch specification containing the upstream download URL.
config: Resolved configuration (unused, included for API compatibility).
logger: Logger adapter for structured observability.

Returns:
FetchPlan referencing the specified ZIP archive.

Raises:
ConfigError: If the specification omits the required URL.

### `plan(self, spec, config, logger)`

Return a fetch plan pointing to Ontobee-managed PURLs.

Args:
spec: Fetch specification describing the ontology identifier and preferred formats.
config: Resolved configuration (unused beyond interface compatibility).
logger: Logger adapter for structured telemetry.

Returns:
FetchPlan pointing to an Ontobee-hosted download URL.

Raises:
ConfigError: If the ontology identifier is invalid.

### `_retryable(exc)`

*No documentation available.*

### `_on_retry(attempt, exc, sleep_time)`

*No documentation available.*

### `_invoke()`

*No documentation available.*

## 3. Classes

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
