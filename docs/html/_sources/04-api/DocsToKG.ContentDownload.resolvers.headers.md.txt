# 1. Module: headers

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.headers``.

Resolver header utilities shared across provider implementations.

This module exposes helpers that normalise HTTP header dictionaries into
hashable cache keys. Centralising the logic avoids subtle drift between
provider modules and eliminates cross-imports of private helpers.

## 1. Functions

### `headers_cache_key(headers)`

Return a deterministic cache key for HTTP header dictionaries.

Args:
headers: Mapping of header names to values.

Returns:
Tuple of lowercase header names paired with their original values,
sorted alphabetically for stable hashing.
