# 1. Module: doaj

This reference describes the ``DoajResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.pipeline.providers``.

DOAJ Resolver Provider

This module integrates with the Directory of Open Access Journals (DOAJ) API
to discover publisher-hosted open-access PDFs associated with scholarly works.

Key Features:
- Optional authenticated access via DOAJ API keys.
- Deduplicated URL emission and structured error metadata.
- Integration with the shared retry helper for polite request handling.

Usage:
    from DocsToKG.ContentDownload.pipeline.providers import DoajResolver

    resolver = DoajResolver()
    urls = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for DOAJ lookup.

Args:
config: Resolver configuration containing DOAJ API credentials.
artifact: Work artifact possibly holding a DOI identifier.

Returns:
Boolean indicating whether DOAJ resolution should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered via DOAJ article metadata.

Args:
session: HTTP session capable of performing DOAJ API requests.
config: Resolver configuration specifying headers and API key.
artifact: Work artifact representing the item being resolved.

Returns:
Iterable of resolver results for candidate Open Access URLs.

## 2. Classes

### `DoajResolver`

Resolve Open Access links using the DOAJ API.

Attributes:
name: Resolver identifier surfaced to the orchestration pipeline.

Examples:
>>> resolver = DoajResolver()
>>> resolver.name
'doaj'
