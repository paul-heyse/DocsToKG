# 1. Module: semantic_scholar

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.semantic_scholar``.

Resolver that queries the Semantic Scholar Graph API for open access PDFs.

## 1. Functions

### `_fetch_semantic_scholar_data(doi, api_key, timeout, headers_key)`

Fetch Semantic Scholar Graph API metadata for ``doi`` with caching.

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
