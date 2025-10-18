# 1. Module: semantic_scholar

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.semantic_scholar``.

## 1. Overview

Resolver implementation for the Semantic Scholar Graph API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is available for Semantic Scholar queries.

Args:
config: Resolver configuration containing API credentials.
artifact: Work record being processed.

Returns:
bool: Whether the resolver should attempt to fetch metadata.

### `iter_urls(self, session, config, artifact)`

Yield Semantic Scholar hosted PDFs linked to ``artifact``.

Args:
session: Requests session for outbound HTTP calls.
config: Resolver configuration with API keys and limits.
artifact: Work metadata containing DOI information.

Yields:
ResolverResult: Candidate download URLs or diagnostic events.

## 3. Classes

### `SemanticScholarResolver`

Resolve PDFs via the Semantic Scholar Graph API.
