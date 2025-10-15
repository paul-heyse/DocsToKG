# 1. Module: openaire

This reference describes the ``OpenAireResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.resolvers.providers``.

OpenAIRE Resolver Provider

This module integrates with the OpenAIRE research infrastructure to discover
open-access artefacts linked to DOI-indexed works.

Key Features:
- Recursive traversal of OpenAIRE JSON payloads to locate URL candidates.
- Resilient handling of cases where responses arrive as either JSON or text.
- Deduplication of candidate URLs before yielding resolver results.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import OpenAireResolver

    resolver = OpenAireResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `_collect_candidate_urls(node, results)`

Recursively collect HTTP(S) URLs from nested OpenAIRE response payloads.

Args:
node: Arbitrary node from the OpenAIRE response payload.
results: Mutable list to append discovered URL strings into.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI.

Args:
config: Resolver configuration controlling OpenAIRE behaviour.
artifact: Work artifact containing metadata such as DOI.

Returns:
Boolean indicating whether an OpenAIRE lookup should be attempted.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered via OpenAIRE search.

Args:
session: HTTP session available for outbound requests.
config: Resolver configuration with polite headers and timeouts.
artifact: Work artifact describing the item being resolved.

Returns:
Iterable of resolver results representing discovered URLs.

## 2. Classes

### `OpenAireResolver`

Resolve URLs using the OpenAIRE API.

Attributes:
name: Resolver identifier used when interacting with the pipeline.

Examples:
>>> resolver = OpenAireResolver()
>>> resolver.name
'openaire'
