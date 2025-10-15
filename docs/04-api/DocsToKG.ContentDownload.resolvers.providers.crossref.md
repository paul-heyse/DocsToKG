# 1. Module: crossref

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.crossref``.

Crossref Resolver Provider

This module integrates with the Crossref metadata API to surface direct and
publisher-hosted PDF links for scholarly works. It uses polite rate limiting,
centralised retry handling, and caching strategies to comply with Crossref
service guidelines.

Key Features:
- Cached metadata retrieval with header normalisation for polite access.
- Robust error handling for HTTP, timeout, and JSON parsing failures.
- Deduplication of returned URLs to minimise redundant download attempts.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.crossref import CrossrefResolver

    resolver = CrossrefResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `_fetch_crossref_data(doi, mailto, timeout, headers_key)`

Retrieve Crossref metadata for ``doi`` with polite header caching.

Args:
doi: Normalised DOI string to request metadata for.
mailto: Contact email used for Crossref's polite rate limiting headers.
timeout: Request timeout in seconds.
headers_key: Hashable representation of polite headers for cache lookups.

Returns:
Decoded JSON payload returned by the Crossref API.

Raises:
requests.HTTPError: If the Crossref API responds with a non-success status.

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

## 2. Classes

### `CrossrefResolver`

Resolve candidate URLs from the Crossref metadata API.

Attributes:
name: Resolver identifier advertised to the orchestrating pipeline.

Examples:
>>> resolver = CrossrefResolver()
>>> resolver.name
'crossref'
