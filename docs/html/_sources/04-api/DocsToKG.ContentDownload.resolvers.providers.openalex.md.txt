# 1. Module: openalex

This reference describes the ``OpenAlexResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.resolvers.providers``.

OpenAlex Resolver Provider

This module yields candidate download URLs directly from OpenAlex metadata,
serving as the first resolver in the pipeline for low-latency successes.

Key Features:
- Deduplication of OpenAlex-provided PDF and open-access URLs.
- Skip events when no URLs are advertised within the OpenAlex record.
- Compatibility shim that ignores unused parameters required by the resolver protocol.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import OpenAlexResolver

    resolver = OpenAlexResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return True when the OpenAlex artifact exposes accessible PDF URLs.

Args:
config: Resolver configuration (unused but required for interface parity).
artifact: OpenAlex metadata object containing URL fields.

Returns:
bool: ``True`` when at least one candidate PDF URL is available.

### `iter_urls(self, session, config, artifact)`

Yield resolver results for each accessible PDF URL in the artifact.

Args:
session: Requests session forwarded for interface compatibility.
config: Resolver configuration (unused but accepted for uniform signature).
artifact: OpenAlex metadata object providing URL candidates.

Returns:
Iterable[ResolverResult]: Iterator producing resolver results for each unique URL.

## 2. Classes

### `OpenAlexResolver`

Resolve OpenAlex work metadata into candidate download URLs.

Attributes:
name: Resolver identifier advertised to the pipeline.

Examples:
>>> resolver = OpenAlexResolver()
>>> resolver.name
'openalex'
