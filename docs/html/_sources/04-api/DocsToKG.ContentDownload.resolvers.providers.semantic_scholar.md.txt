# 1. Module: semantic_scholar

This reference describes the ``SemanticScholarResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.resolvers.providers``.

Semantic Scholar Resolver Provider

This module integrates with the Semantic Scholar Graph API to locate open
access PDFs associated with DOI-indexed papers.

Key Features:
- Memoised API lookups to respect rate limits and improve performance.
- Optional API key support via standard ``x-api-key`` headers.
- Structured error emission covering HTTP and JSON decoding failures.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import SemanticScholarResolver

    resolver = SemanticScholarResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `_fetch_semantic_scholar_data(doi, api_key, timeout, headers_key)`

Fetch Semantic Scholar Graph API metadata for ``doi`` with caching.

Args:
doi: Normalised DOI string to query.
api_key: Optional Semantic Scholar API key to include in the request.
timeout: Request timeout in seconds.
headers_key: Hashable representation of polite headers for cache lookups.

Returns:
Decoded JSON payload returned by the Semantic Scholar Graph API.

Raises:
requests.HTTPError: If the API responds with a non-success status code.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI suitable for lookup.

Args:
config: Resolver configuration containing Semantic Scholar settings.
artifact: Work artifact potentially carrying a DOI.

Returns:
Boolean indicating whether the resolver can operate on the artifact.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered via Semantic Scholar.

Args:
session: HTTP session for outbound API communication.
config: Resolver configuration providing headers and timeouts.
artifact: Work artifact describing the scholarly work to resolve.

Returns:
Iterable of resolver results containing download URLs or metadata events.

## 2. Classes

### `SemanticScholarResolver`

Resolve PDFs via the Semantic Scholar Graph API.

Attributes:
name: Resolver identifier exposed to the orchestration layer.

Examples:
>>> resolver = SemanticScholarResolver()
>>> resolver.name
'semantic_scholar'
