# 1. Module: wayback

This reference describes the ``WaybackResolver`` class provided by the consolidated module ``DocsToKG.ContentDownload.resolvers.providers``.

Wayback Machine Resolver Provider

This module queries the Internet Archive Wayback Machine to retrieve archived
snapshots of PDF URLs that previously failed during resolver execution.

Key Features:
- Recovery of archived snapshots for failed HTTP URLs.
- Structured error reporting for network and HTTP issues.
- Graceful handling when no snapshots are available or when failures persist.

Usage:
    from DocsToKG.ContentDownload.resolvers.providers import WaybackResolver

    resolver = WaybackResolver()
    results = list(resolver.iter_urls(session, config, artifact))

## 1. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when previous resolvers have recorded failed URLs.

Args:
config: Resolver configuration governing Wayback usage.
artifact: Work artifact containing details of failed downloads.

Returns:
Boolean indicating whether the Wayback resolver should run.

### `iter_urls(self, session, config, artifact)`

Yield archived URLs from the Internet Archive when available.

Args:
session: HTTP session used to contact the Wayback API.
config: Resolver configuration exposing headers and timeouts.
artifact: Work artifact describing failed PDF URLs to recover.

Returns:
Iterable of resolver results referencing archived snapshots.

## 2. Classes

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine.

Attributes:
name: Resolver identifier surfaced to the pipeline.

Examples:
>>> resolver = WaybackResolver()
>>> resolver.name
'wayback'
