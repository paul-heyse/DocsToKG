# 1. Module: schema

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.schema``.

> **Deprecated:** ``DocsToKG.HybridSearch.schema`` is a compatibility shim that re-exports
> OpenSearch schema utilities from ``DocsToKG.HybridSearch.storage``. The shim emits a
> :class:`DeprecationWarning` on import to guide migrations.

## 1. Re-exported symbols

| Symbol | New home |
| ------ | -------- |
| ``OpenSearchIndexTemplate`` | ``DocsToKG.HybridSearch.storage`` |
| ``OpenSearchSchemaManager`` | ``DocsToKG.HybridSearch.storage`` |

The underlying schema management API is unchanged. Update imports to reference the
``storage`` module to ensure compatibility once the shim is removed.
