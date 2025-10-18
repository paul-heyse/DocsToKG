# 1. Module: unpaywall

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.unpaywall``.

## 1. Overview

Resolver implementation for the Unpaywall API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when unpaywall credentials and a DOI are provided.

Args:
config: Resolver configuration containing the Unpaywall contact email.
artifact: Work record offering DOI metadata.

Returns:
bool: Whether the resolver should attempt to fetch data.

### `iter_urls(self, session, config, artifact)`

Yield Unpaywall-sourced PDF URLs for ``artifact``.

Args:
session: Requests session for outbound HTTP calls.
config: Resolver configuration with API parameters.
artifact: Work metadata used to build the lookup.

Yields:
ResolverResult: Candidate download URLs or diagnostic events.

## 3. Classes

### `UnpaywallResolver`

Resolve PDFs via the Unpaywall API.
