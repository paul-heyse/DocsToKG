# 1. Module: hal

This reference describes the ``HalResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.pipeline.providers``.

HAL Resolver Provider

This module integrates with the HAL (Hyper Articles en Ligne) open archive API
to discover repository-hosted PDFs for works indexed in OpenAlex.

Key Features:
- Query construction against HAL search endpoints using DOI filters.
- Extraction of both main and auxiliary file URLs with PDF filtering.
- Error handling and logging for HTTP and JSON parsing issues.

Usage:
    from DocsToKG.ContentDownload.pipeline.providers import HalResolver

    resolver = HalResolver()
    urls = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for HAL lookup.

Args:
config: Resolver configuration providing HAL request settings.
artifact: Work artifact potentially containing a DOI identifier.

Returns:
Boolean indicating whether HAL resolution is applicable.

### `iter_urls(self, session, config, artifact)`

Yield candidate HAL download URLs.

Args:
session: HTTP session used for outbound HAL API requests.
config: Resolver configuration specifying headers and timeouts.
artifact: Work artifact representing the item to resolve.

Returns:
Iterable of resolver results describing resolved URLs.

## 2. Classes

### `HalResolver`

Resolve publications from the HAL open archive.

Attributes:
name: Resolver identifier communicated to the pipeline orchestration.

Examples:
>>> resolver = HalResolver()
>>> resolver.name
'hal'
