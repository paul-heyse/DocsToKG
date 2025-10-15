# Module: crossref

Resolver that queries the Crossref metadata API to surface publisher-hosted PDFs.

## Functions

### `_fetch_crossref_data(doi, mailto, timeout, headers_key)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI available for lookup.

Args:
config: Resolver configuration providing Crossref connectivity details.
artifact: Work artifact containing bibliographic metadata.

Returns:
Boolean indicating whether a Crossref query should be attempted.

### `iter_urls(self, session, config, artifact)`

Yield URLs discovered via the Crossref API for a given artifact.

Args:
session: HTTP session capable of issuing outbound requests.
config: Resolver configuration including timeouts and polite headers.
artifact: Work artifact carrying DOI metadata.

Returns:
Iterable of resolver results describing discovered URLs or errors.

## Classes

### `CrossrefResolver`

Resolve candidate URLs from the Crossref metadata API.

Attributes:
name: Resolver identifier advertised to the orchestrating pipeline.

Examples:
>>> resolver = CrossrefResolver()
>>> resolver.name
'crossref'
