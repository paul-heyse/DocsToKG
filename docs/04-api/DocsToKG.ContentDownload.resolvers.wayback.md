# 1. Module: wayback

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.wayback``.

## 1. Overview

Resolver that queries the Internet Archive Wayback Machine with CDX-first discovery algorithm.

## 2. Functions

### `request_with_retries()`

Proxy to :func:`resolvers.base.request_with_retries` for patchability.

### `is_enabled(self, config, artifact)`

Return ``True`` when prior resolver attempts have failed.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record containing failed PDF URLs.

Returns:
bool: Whether the Wayback resolver should run.

### `iter_urls(self, client, config, artifact)`

Query the Wayback Machine for archived versions of failed URLs using CDX-first discovery.

Args:
client: HTTPX client for HTTP calls with caching and rate limiting.
config: Resolver configuration providing timeouts, headers, and Wayback-specific options.
artifact: Work metadata listing failed PDF URLs to retry.

Yields:
ResolverResult: Archived download URLs or diagnostic events.

## 3. Classes

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine with CDX-first discovery algorithm. Supports both direct PDF snapshot recovery and HTML parsing for PDF link discovery.
