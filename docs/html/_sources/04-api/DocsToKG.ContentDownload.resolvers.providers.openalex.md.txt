# 1. Module: openalex

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.providers.openalex``.

OpenAlex direct URL resolver (position 0 in pipeline).

## 1. Functions

### `is_enabled(self, config, artifact)`

Return True when the OpenAlex artifact exposes accessible PDF URLs.

Args:
config: Resolver configuration (unused but required for interface parity).
artifact: OpenAlex metadata object containing URL fields.

Returns:
``True`` when at least one candidate PDF URL is available, otherwise ``False``.

### `iter_urls(self, session, config, artifact)`

Yield resolver results for each accessible PDF URL in the artifact.

Args:
session: Requests session forwarded for interface compatibility.
config: Resolver configuration (unused but accepted for uniform signature).
artifact: OpenAlex metadata object providing URL candidates.

Returns:
Iterator producing :class:`ResolverResult` objects for each unique URL.

Raises:
None

## 2. Classes

### `OpenAlexResolver`

Resolve OpenAlex work metadata into candidate download URLs.

Attributes:
name: Resolver identifier advertised to the pipeline.

Examples:
>>> resolver = OpenAlexResolver()
>>> resolver.name
'openalex'
