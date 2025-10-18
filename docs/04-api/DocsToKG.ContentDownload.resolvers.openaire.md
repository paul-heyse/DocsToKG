# 1. Module: openaire

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.openaire``.

## 1. Overview

Resolver implementation for the OpenAIRE API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the work includes a DOI for OpenAIRE queries.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record under evaluation.

Returns:
bool: Whether this resolver should attempt the work.

### `iter_urls(self, session, config, artifact)`

Yield OpenAIRE URLs that point to downloadable PDFs.

Args:
session: Requests session for issuing HTTP requests.
config: Resolver configuration providing timeouts and headers.
artifact: Work metadata containing the DOI lookup.

Yields:
ResolverResult: Candidate download URLs or diagnostic events.

## 3. Classes

### `OpenAireResolver`

Resolve URLs using the OpenAIRE API.
