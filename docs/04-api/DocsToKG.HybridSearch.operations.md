# 1. Module: operations

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.operations``.

> **Deprecated:** ``DocsToKG.HybridSearch.operations`` is a compatibility shim retained for
> one release cycle. Import helpers directly from their consolidated modules:
>
> - ``DocsToKG.HybridSearch.service.build_stats_snapshot``
> - ``DocsToKG.HybridSearch.service.verify_pagination``
> - ``DocsToKG.HybridSearch.service.should_rebuild_index``
> - ``DocsToKG.HybridSearch.vectorstore.serialize_state``
> - ``DocsToKG.HybridSearch.vectorstore.restore_state``

The shim re-exports these callables and emits a ``DeprecationWarning`` at import time to
help downstream users migrate to the new module structure introduced in the
``consolidate-hybridsearch-modules`` change.

## 1. Re-exported symbols

| Symbol | New home |
| ------ | -------- |
| ``build_stats_snapshot`` | ``DocsToKG.HybridSearch.service`` |
| ``verify_pagination`` | ``DocsToKG.HybridSearch.service`` |
| ``should_rebuild_index`` | ``DocsToKG.HybridSearch.service`` |
| ``serialize_state`` | ``DocsToKG.HybridSearch.vectorstore`` |
| ``restore_state`` | ``DocsToKG.HybridSearch.vectorstore`` |
| ``PaginationCheckResult`` | ``DocsToKG.HybridSearch.service`` |

All functionality remains unchanged; only import paths differ. New development should
target the consolidated modules directly.
