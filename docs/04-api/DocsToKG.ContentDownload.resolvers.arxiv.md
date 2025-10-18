# 1. Module: arxiv

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.arxiv``.

## 1. Overview

Resolver implementation for arXiv preprints.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the work exposes an arXiv identifier.

Args:
config: Resolver configuration (unused but part of the contract).
artifact: Work record under consideration.

Returns:
bool: Whether this resolver should attempt to resolve the work.

### `iter_urls(self, session, config, artifact)`

Yield the canonical arXiv PDF URL for ``artifact`` when available.

Args:
session: Requests session to use for follow-up HTTP calls.
config: Resolver configuration supplied by the pipeline.
artifact: Work record containing source identifiers.

Yields:
ResolverResult: Candidate PDF URL or skip event.

## 3. Classes

### `ArxivResolver`

Resolve arXiv preprints using arXiv identifier lookups.
