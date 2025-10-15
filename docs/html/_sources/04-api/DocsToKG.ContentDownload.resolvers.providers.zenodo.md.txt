# 1. Module: zenodo

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.zenodo``.

Zenodo Resolver Provider

This module queries the Zenodo API to locate repository-hosted PDFs associated
with DOI-indexed research objects.

Key Features:
- Support for sorting and limiting Zenodo API search results.
- Extensive logging for malformed responses and missing file metadata.
- Deduplication-free iteration over PDF file entries.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers.zenodo import ZenodoResolver

    resolver = ZenodoResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return True when the artifact publishes a DOI for Zenodo lookup.

Args:
config: Resolver configuration (unused but part of the protocol signature).
artifact: Work metadata potentially referencing Zenodo.

Returns:
bool: ``True`` when a DOI is available, otherwise ``False``.

### `iter_urls(self, session, config, artifact)`

Query the Zenodo API by DOI and yield PDF file URLs.

Args:
session: Requests session used for making Zenodo API calls.
config: Resolver configuration providing polite headers and timeouts.
artifact: Work metadata containing the DOI search key.

Returns:
Iterable[ResolverResult]: Iterator yielding resolver results for accessible PDFs.

Notes:
All HTTP calls honour per-resolver timeouts by delegating to
:meth:`ResolverConfig.get_timeout`.

## 2. Classes

### `ZenodoResolver`

Resolve Zenodo records into downloadable open-access PDF URLs.

Attributes:
name: Resolver identifier registered with the content download pipeline.

Examples:
>>> resolver = ZenodoResolver()
>>> resolver.name
'zenodo'
