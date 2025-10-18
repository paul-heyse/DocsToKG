# 1. Module: zenodo

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.zenodo``.

## 1. Overview

Resolver implementation for the Zenodo REST API.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when a DOI is present to drive Zenodo lookups.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record describing the document.

Returns:
bool: Whether the resolver should attempt the work.

### `iter_urls(self, session, config, artifact)`

Yield Zenodo hosted PDFs for the supplied work.

Args:
session: Requests session for issuing HTTP calls.
config: Resolver configuration providing retry policies.
artifact: Work metadata containing DOI information.

Yields:
ResolverResult: Candidate download URLs or diagnostic events.

## 3. Classes

### `ZenodoResolver`

Resolve Zenodo records into downloadable open-access PDF URLs.
