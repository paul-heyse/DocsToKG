# Module: resolvers

Ontology Resolver Implementations

This module defines resolver strategies that translate fetch specifications
into concrete download plans. Resolvers integrate with services such as the
OBO Library, OLS, and BioPortal to identify canonical document URLs for
downloading ontology content.

## Functions

### `_execute_with_retry(self, func)`

*No documentation available.*

### `_build_plan(self)`

*No documentation available.*

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

*No documentation available.*

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

### `join(self)`

*No documentation available.*

## Classes

### `FetchPlan`

Concrete plan output from a resolver.

Attributes:
url: Final URL from which to download the ontology document.
headers: HTTP headers required by the upstream service.
filename_hint: Optional filename recommended by the resolver.
version: Version identifier derived from resolver metadata.
license: License reported for the ontology.
media_type: MIME type of the artifact when known.

### `BaseResolver`

Shared helpers for resolver implementations.

### `OBOResolver`

Resolve ontologies hosted on the OBO Library using Bioregistry helpers.

### `OLSResolver`

Resolve ontologies from the Ontology Lookup Service (OLS4).

### `BioPortalResolver`

Resolve ontologies using the BioPortal (OntoPortal) API.

### `SKOSResolver`

Resolver for direct SKOS/RDF URLs.

### `XBRLResolver`

Resolver for XBRL taxonomy packages.

### `_PystowFallback`

Minimal pystow replacement used when optional dependency is absent.
