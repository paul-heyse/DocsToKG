# 1. Module: core

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.core``.

## 1. Overview

Resolver implementation for the CORE API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is present and the CORE API is configured.

Args:
config: Resolver configuration containing API credentials.
artifact: Work record under consideration.

Returns:
bool: ``True`` if the resolver can operate for this work.

### `iter_urls(self, session, config, artifact)`

Yield download URLs discovered via the CORE search API.

Args:
session: Requests session to execute HTTP calls.
config: Resolver configuration with credentials and limits.
artifact: Work record providing identifiers.

Yields:
ResolverResult: Candidate download URLs or diagnostic events.

## 3. Classes

### `CoreResolver`

Resolve PDFs using the CORE API.
