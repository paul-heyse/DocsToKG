# Module: __init__

Resolver pipeline and provider implementations.

This module maintains backward compatibility by re-exporting all public APIs.
New code should import from submodules (pipeline, types, providers) directly.

## Functions

### `clear_resolver_caches()`

Clear resolver-level LRU caches.

Args:
None

Returns:
None
