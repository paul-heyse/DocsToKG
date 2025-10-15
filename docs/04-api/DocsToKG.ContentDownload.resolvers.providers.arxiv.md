# 1. Module: arxiv

This reference describes the ``ArxivResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.resolvers.providers``.

arXiv Resolver Provider

This module implements a lightweight resolver that converts arXiv identifiers
into direct PDF download URLs. It participates in the modular content download
pipeline when arXiv metadata is available for a work.

Key Features:
- Normalises arXiv identifiers sourced from OpenAlex metadata.
- Emits deterministic PDF URLs suitable for direct download.
- Records skip events when metadata is incomplete.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import ArxivResolver

    resolver = ArxivResolver()
    list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has an arXiv identifier.

Args:
config: Resolver configuration providing arXiv availability toggles.
artifact: Work artifact potentially containing an arXiv identifier.

Returns:
Boolean indicating whether arXiv resolution should be attempted.

### `iter_urls(self, session, config, artifact)`

Yield candidate arXiv download URLs.

Args:
session: HTTP session available to perform network requests.
config: Resolver configuration describing timeouts and headers.
artifact: Work artifact with arXiv metadata for resolution.

Returns:
Iterable of resolver results containing download URLs or metadata events.

## 2. Classes

### `ArxivResolver`

Resolve arXiv preprints using arXiv identifier lookups.

Attributes:
name: Resolver identifier announced to the pipeline.

Examples:
>>> resolver = ArxivResolver()
>>> resolver.name
'arxiv'
