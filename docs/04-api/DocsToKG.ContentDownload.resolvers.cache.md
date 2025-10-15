# 1. Module: cache

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.cache``.

Resolver Cache Utilities

This helper module exposes functions that clear resolver-level LRU caches.
It exists so callers can invalidate cached HTTP lookups without importing the
deprecated :mod:`DocsToKG.ContentDownload.resolvers` facade, which emits
deprecation warnings when accessed directly.

Usage:
    from DocsToKG.ContentDownload.resolvers.cache import clear_resolver_caches

    clear_resolver_caches()

## 1. Functions

### `clear_resolver_caches()`

Clear resolver-level HTTP caches to force fresh lookups.

This utility resets the internal LRU caches used by the Unpaywall,
Crossref, and Semantic Scholar resolvers. It should be called before
executing resolver pipelines when deterministic behaviour across runs is
required (for example, in unit tests or benchmarking scenarios).
