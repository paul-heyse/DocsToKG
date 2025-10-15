# 1. Module: zenodo

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.zenodo``.

Zenodo repository resolver for DOI-indexed research outputs.

## 1. Functions

### `is_enabled(self, config, artifact)`

Return True when the artifact publishes a DOI for Zenodo lookup.

Args:
config: Resolver configuration (unused but part of the protocol signature).
artifact: Work metadata potentially referencing Zenodo.

Returns:
``True`` when a DOI is available, otherwise ``False``.

### `iter_urls(self, session, config, artifact)`

Query the Zenodo API by DOI and yield PDF file URLs.

Args:
session: Requests session used for making Zenodo API calls.
config: Resolver configuration providing polite headers and timeouts.
artifact: Work metadata containing the DOI search key.

Returns:
Iterator yielding :class:`ResolverResult` objects for each accessible PDF.

Raises:
None

## 2. Classes

### `ZenodoResolver`

Resolve Zenodo records into downloadable open-access PDF URLs.

Attributes:
name: Resolver identifier registered with the content download pipeline.

Examples:
>>> resolver = ZenodoResolver()
>>> resolver.name
'zenodo'
