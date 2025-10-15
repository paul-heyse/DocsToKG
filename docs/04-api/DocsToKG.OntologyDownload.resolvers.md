# 1. Module: resolvers

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.resolvers``.

Ontology Resolver Implementations

This module defines the resolver strategies that convert download
specifications into actionable fetch plans. Each resolver encapsulates the
API integration, retry logic, and metadata extraction necessary to interact
with external services such as the OBO Library, OLS, BioPortal, and SKOS/XBRL
endpoints.

Key Features:
- Shared retry/backoff helpers for consistent API resilience
- Resolver-specific metadata extraction (version, license, media type)
- Support for additional services through the pluggable ``RESOLVERS`` map

Usage:
    from DocsToKG.OntologyDownload.resolvers import RESOLVERS

    resolver = RESOLVERS["obo"]
    plan = resolver.plan(spec, config, logger)

## 1. Functions

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
None

Returns:
API key string stripped of whitespace, or ``None`` when missing.

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

## 2. Classes

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
