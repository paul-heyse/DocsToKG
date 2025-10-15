# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.__init__``.

Resolver pipeline and provider implementations.

This module maintains backward compatibility by re-exporting legacy entry points
so existing integrations and tests can continue to monkeypatch ``requests`` or
``time`` on the resolver namespace.

## 1. Functions

### `clear_resolver_caches()`

Clear resolver-level LRU caches.

Args:
None

Returns:
None
