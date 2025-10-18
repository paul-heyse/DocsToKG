# 1. Module: crossref

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.crossref``.

## 1. Overview

Resolver implementation for the Crossref metadata API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the work exposes a DOI Crossref can query.

Args:
config: Resolver configuration (unused, required for signature).
artifact: Work record under consideration.

Returns:
bool: Whether the resolver should attempt this work.

### `iter_urls(self, session, config, artifact)`

Yield PDF URLs referenced by Crossref metadata for ``artifact``.

Args:
session: Requests session for outbound HTTP calls.
config: Resolver configuration controlling behaviour.
artifact: Work record containing DOI and other metadata.

Yields:
ResolverResult: Candidate download URLs or skip events.

### `_score(candidate)`

*No documentation available.*

## 3. Classes

### `CrossrefResolver`

Resolve candidate URLs from the Crossref metadata API.
