# 1. Module: __init__

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.__init__``.

Resolver pipeline and provider implementations.

This module maintains backward compatibility by re-exporting legacy entry points
so existing integrations and tests can continue to monkeypatch ``requests`` or
``time`` on the resolver namespace. Accessing ``time`` or ``requests`` now emits
a :class:`DeprecationWarning`; import the standard modules directly when
updating custom tooling.

The module also exposes ``ResolverPipeline``/``ResolverConfig`` convenience
aliases for callers that have not yet migrated to the modular
``pipeline``/``types`` submodules.

## 1. Functions

### `clear_resolver_caches()`

Clear resolver-level LRU caches.

Args:
None

Returns:
None

### `__getattr__(name)`

Return compatibility shims for deprecated attributes (``time`` and
``requests``) while emitting :class:`DeprecationWarning` to encourage direct
imports.
