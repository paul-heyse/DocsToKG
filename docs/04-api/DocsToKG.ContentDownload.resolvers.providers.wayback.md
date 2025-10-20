# 1. Module: wayback

This reference describes the ``WaybackResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.pipeline.providers``.

Wayback Machine Resolver Provider

This module queries the Internet Archive Wayback Machine using CDX-first discovery
to retrieve archived snapshots of PDF URLs that previously failed during resolver execution.

Key Features:

- CDX-first discovery algorithm for comprehensive snapshot search
- Availability API fast-path for quick snapshot detection
- HTML parsing fallback to extract PDF links from archived landing pages
- PDF verification with HEAD requests and signature checking
- Structured telemetry for monitoring resolver effectiveness

Usage:
    from DocsToKG.ContentDownload.pipeline.providers import WaybackResolver

    resolver = WaybackResolver()
    results = list(resolver.iter_urls(client, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when previous resolvers have recorded failed URLs.

Args:
config: Resolver configuration governing Wayback usage.
artifact: Work artifact containing details of failed downloads.

Returns:
Boolean indicating whether the Wayback resolver should run.

### `iter_urls(self, client, config, artifact)`

Yield archived URLs from the Internet Archive when available using CDX-first discovery.

Args:
client: HTTPX client with caching and rate limiting for Wayback API calls.
config: Resolver configuration exposing headers, timeouts, and Wayback-specific options.
artifact: Work artifact describing failed PDF URLs to recover.

Returns:
Iterable of resolver results referencing archived snapshots.

## 2. Classes

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine with CDX-first discovery algorithm.

Attributes:
name: Resolver identifier surfaced to the pipeline.

Examples:
>>> resolver = WaybackResolver()
>>> resolver.name
'wayback'
