# 1. Module: doaj

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.doaj``.

## 1. Overview

Resolver implementation for the DOAJ API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is present for DOAJ lookups.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record being processed.

Returns:
bool: Whether DOAJ should attempt resolution.

### `iter_urls(self, session, config, artifact)`

Yield PDF links surfaced by the DOAJ search API.

Args:
session: Requests session for outbound HTTP calls.
config: Resolver configuration containing optional API key.
artifact: Work record supplying DOI metadata.

Yields:
ResolverResult: Candidate download URLs or skip events.

## 3. Classes

### `DoajResolver`

Resolve Open Access links using the DOAJ API.
