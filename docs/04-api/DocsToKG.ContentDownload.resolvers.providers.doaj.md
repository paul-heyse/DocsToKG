# Module: doaj

DOAJ (Directory of Open Access Journals) resolver.

## Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for DOAJ lookup.

Args:
config: Resolver configuration containing DOAJ API credentials.
artifact: Work artifact possibly holding a DOI identifier.

Returns:
Boolean indicating whether DOAJ resolution should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered via DOAJ article metadata.

Args:
session: HTTP session capable of performing DOAJ API requests.
config: Resolver configuration specifying headers and API key.
artifact: Work artifact representing the item being resolved.

Returns:
Iterable of resolver results for candidate Open Access URLs.

## Classes

### `DoajResolver`

Resolve Open Access links using the DOAJ API.

Attributes:
name: Resolver identifier surfaced to the orchestration pipeline.

Examples:
>>> resolver = DoajResolver()
>>> resolver.name
'doaj'
