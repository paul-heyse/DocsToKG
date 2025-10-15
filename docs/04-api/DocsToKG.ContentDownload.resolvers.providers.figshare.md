# 1. Module: figshare

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.figshare``.

Figshare repository resolver for DOI-indexed research outputs.

## 1. Functions

### `is_enabled(self, config, artifact)`

Return True when the artifact exposes a DOI for Figshare lookup.

Args:
config: Resolver configuration (unused but part of the protocol signature).
artifact: Work metadata that may reference a Figshare DOI.

Returns:
``True`` when a DOI is present, otherwise ``False``.

### `iter_urls(self, session, config, artifact)`

Search the Figshare API by DOI and yield PDF file URLs.

Args:
session: Requests session used for API calls (supports retry injection).
config: Resolver configuration providing polite headers and timeouts.
artifact: Work metadata containing the DOI search key.

Returns:
Iterator yielding :class:`ResolverResult` objects for each candidate URL.

Raises:
None

## 2. Classes

### `FigshareResolver`

Resolve Figshare repository metadata into download URLs.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> resolver = FigshareResolver()
>>> resolver.name
'figshare'
