# 1. Module: results

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.results``.

> **Deprecated:** ``DocsToKG.HybridSearch.results`` is a compatibility shim preserved for
> one release cycle. Import result shaping utilities from
> ``DocsToKG.HybridSearch.ranking`` instead. The shim re-exports
> :class:`ResultShaper` and emits a :class:`DeprecationWarning` at import time to
> guide migrations to the consolidated ranking module.

## 1. Re-exported symbols

| Symbol | New home |
| ------ | -------- |
| ``ResultShaper`` | ``DocsToKG.HybridSearch.ranking`` |

No behaviour has changed—only the import location. Update imports in
applications and notebooks to reference the ranking module directly.
