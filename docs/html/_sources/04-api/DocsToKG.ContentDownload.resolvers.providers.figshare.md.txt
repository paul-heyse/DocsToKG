# 1. Module: figshare

This reference describes the ``FigshareResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.pipeline.providers``.

Figshare Resolver Provider

This module integrates with the Figshare API to locate repository-hosted PDFs
associated with DOI-indexed research outputs.

Key Features:
- POST-based Figshare search requests with polite headers.
- Iteration over article file metadata to extract PDF download URLs.
- Extensive logging for malformed payloads and API errors.

Usage:
    from DocsToKG.ContentDownload.pipeline.providers import FigshareResolver

    resolver = FigshareResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return True when the artifact exposes a DOI for Figshare lookup.

Args:
config: Resolver configuration (unused but part of the protocol signature).
artifact: Work metadata that may reference a Figshare DOI.

Returns:
bool: ``True`` when a DOI is present, otherwise ``False``.

### `iter_urls(self, session, config, artifact)`

Search the Figshare API by DOI and yield PDF file URLs.

Args:
session: Requests session used for API calls (supports retry injection).
config: Resolver configuration providing polite headers and timeouts.
artifact: Work metadata containing the DOI search key.

Returns:
Iterable[ResolverResult]: Iterator yielding resolver results for each candidate URL.

Notes:
Requests honour resolver-specific timeouts using
:meth:`ResolverConfig.get_timeout` and reuse
:func:`request_with_retries` for resilient execution.

## 2. Classes

### `FigshareResolver`

Resolve Figshare repository metadata into download URLs.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> resolver = FigshareResolver()
>>> resolver.name
'figshare'
