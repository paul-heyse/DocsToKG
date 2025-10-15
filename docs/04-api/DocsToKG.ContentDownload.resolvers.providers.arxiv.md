# Module: arxiv

Resolver that transforms arXiv identifiers into direct PDF download URLs.

## Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has an arXiv identifier.

Args:
config: Resolver configuration providing arXiv availability toggles.
artifact: Work artifact potentially containing an arXiv identifier.

Returns:
Boolean indicating whether arXiv resolution should be attempted.

### `iter_urls(self, session, config, artifact)`

Yield candidate arXiv download URLs.

Args:
session: HTTP session available to perform network requests.
config: Resolver configuration describing timeouts and headers.
artifact: Work artifact with arXiv metadata for resolution.

Returns:
Iterable of resolver results containing download URLs or metadata events.

## Classes

### `ArxivResolver`

Resolve arXiv preprints using arXiv identifier lookups.

Attributes:
name: Resolver identifier announced to the pipeline.

Examples:
>>> resolver = ArxivResolver()
>>> resolver.name
'arxiv'
