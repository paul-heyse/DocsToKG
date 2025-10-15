# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.__init__``.

Resolver pipeline and provider implementations.

This module maintains backward compatibility by re-exporting legacy entry points
so existing integrations and tests can continue to monkeypatch ``requests`` or
``time`` on the resolver namespace.

.. note::
   Importing from this facade now emits ``DeprecationWarning``. Prefer importing
   from explicit submodules such as
   ``DocsToKG.ContentDownload.resolvers.pipeline`` or
   ``DocsToKG.ContentDownload.resolvers.types``.

## 1. Functions

### `clear_resolver_caches()`

Clear resolver-level LRU caches.

Args:
None

Returns:
None
