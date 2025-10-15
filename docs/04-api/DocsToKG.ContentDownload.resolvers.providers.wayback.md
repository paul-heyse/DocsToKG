# Module: wayback

Internet Archive Wayback Machine fallback resolver.

## Functions

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

## Classes

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine.

Attributes:
name: Resolver identifier surfaced to the pipeline.

Examples:
>>> resolver = WaybackResolver()
>>> resolver.name
'wayback'
