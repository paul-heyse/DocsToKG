# Module: osf

Resolver targeting the Open Science Framework API for preprint downloads.

## Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for OSF lookup.

Args:
config: Resolver configuration providing OSF request details.
artifact: Work artifact potentially containing a DOI value.

Returns:
Boolean indicating whether the resolver should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate download URLs from the OSF API.

Args:
session: HTTP session available for API requests.
config: Resolver configuration including polite headers and timeouts.
artifact: Work artifact describing the record under consideration.

Returns:
Iterable of resolver results containing candidate download URLs.

## Classes

### `OsfResolver`

Resolve artefacts hosted on the Open Science Framework.

Attributes:
name: Resolver identifier shared with the pipeline.

Examples:
>>> resolver = OsfResolver()
>>> resolver.name
'osf'
