# 1. Module: unpaywall

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.unpaywall``.

Resolver that integrates with the Unpaywall API to locate open access PDFs.

## 1. Functions

### `_headers_cache_key(headers)`

Create a hashable cache key for polite header dictionaries.

### `_fetch_unpaywall_data(doi, email, timeout, headers_key)`

Fetch Unpaywall metadata for ``doi`` using polite caching.

### `is_enabled(self, config, artifact)`

Return ``True`` when Unpaywall is configured and the work has a DOI.

Args:
config: Resolver configuration containing Unpaywall credentials.
artifact: Work artifact potentially containing a DOI identifier.

Returns:
Boolean indicating whether the Unpaywall resolver should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate PDF URLs discovered via the Unpaywall API.

Args:
session: HTTP session used to issue API requests.
config: Resolver configuration defining headers, timeouts, and email.
artifact: Work artifact describing the scholarly record to resolve.

Returns:
Iterable of resolver results with download URLs or status events.

## 2. Classes

### `UnpaywallResolver`

Resolve PDFs via the Unpaywall API.

Attributes:
name: Resolver identifier announced to the pipeline.

Examples:
>>> resolver = UnpaywallResolver()
>>> resolver.name
'unpaywall'
