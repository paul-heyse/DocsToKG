# 1. Module: wayback

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.wayback``.

## 1. Overview

Resolver that queries the Internet Archive Wayback Machine.

## 2. Functions

### `is_enabled(self, config, artifact)`

Return ``True`` when prior resolver attempts have failed.

Args:
config: Resolver configuration (unused for enablement).
artifact: Work record containing failed PDF URLs.

Returns:
bool: Whether the Wayback resolver should run.

### `iter_urls(self, session, config, artifact)`

Query the Wayback Machine for archived versions of failed URLs.

Args:
session: Requests session for HTTP calls.
config: Resolver configuration providing timeouts and headers.
artifact: Work metadata listing failed PDF URLs to retry.

Yields:
ResolverResult: Archived download URLs or diagnostic events.

## 3. Classes

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine.
