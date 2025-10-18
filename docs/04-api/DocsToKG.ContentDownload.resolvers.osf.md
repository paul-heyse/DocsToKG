# 1. Module: osf

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.osf``.

## 1. Overview

Resolver implementation for the Open Science Framework API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is available for OSF lookups.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record describing the document.

Returns:
bool: Whether the resolver should attempt the work.

### `iter_urls(self, session, config, artifact)`

Yield OSF download URLs corresponding to ``artifact``.

Args:
session: Requests session for HTTP operations.
config: Resolver configuration managing limits.
artifact: Work metadata providing DOI information.

Yields:
ResolverResult: Candidate download URLs or skip events.

## 3. Classes

### `OsfResolver`

Resolve artefacts hosted on the Open Science Framework.
