# 1. Module: hal

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.hal``.

## 1. Overview

Resolver implementation for the HAL open archive API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the work includes a DOI for HAL search.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record being evaluated.

Returns:
bool: Whether the resolver is applicable to the work.

### `iter_urls(self, session, config, artifact)`

Yield HAL download URLs referencing ``artifact``.

Args:
session: Requests session to execute HTTP calls.
config: Resolver configuration with request limits.
artifact: Work metadata that supplies the DOI.

Yields:
ResolverResult: Candidate PDF URLs or diagnostic events.

## 3. Classes

### `HalResolver`

Resolve publications from the HAL open archive.
