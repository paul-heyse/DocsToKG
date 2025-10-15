# 1. Module: unpaywall

This reference describes the ``UnpaywallResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.resolvers.providers``.

Unpaywall Resolver Provider

This module interfaces with the Unpaywall API to discover open-access PDFs for
scholarly works that expose DOI metadata.

Key Features:
- Configurable polite headers and email registration for API compliance.
- Fallback path that leverages memoised requests when no session is supplied.
- Deduplication of candidate URLs across best and alternate OA locations.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import UnpaywallResolver

    resolver = UnpaywallResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `_fetch_unpaywall_data(doi, email, timeout, headers_key)`

Fetch Unpaywall metadata for ``doi`` using polite caching.

Args:
doi: DOI identifier to query against the Unpaywall API.
email: Registered contact email required by the Unpaywall terms.
timeout: Request timeout in seconds.
headers_key: Hashable representation of polite headers for cache lookups.

Returns:
Parsed JSON payload describing open-access locations for the DOI.

Raises:
requests.HTTPError: If the Unpaywall API returns a non-success status code.

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
