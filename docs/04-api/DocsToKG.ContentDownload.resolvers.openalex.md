# 1. Module: openalex

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.openalex``.

## 1. Overview

Resolver implementation that surfaces OpenAlex-provided URLs.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when OpenAlex metadata includes candidate URLs.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record being evaluated.

Returns:
bool: Whether the resolver should attempt the work.

### `iter_urls(self, session, config, artifact)`

Yield URLs surfaced directly from OpenAlex metadata.

Args:
session: Requests session (unused; signature parity).
config: Resolver configuration controlling policy headers.
artifact: Work metadata containing PDF candidates.

Yields:
ResolverResult: Candidate download URLs or skip events.

## 3. Classes

### `OpenAlexResolver`

Resolve OpenAlex work metadata into candidate download URLs.
