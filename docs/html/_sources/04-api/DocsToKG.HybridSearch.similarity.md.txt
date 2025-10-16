# 1. Module: similarity

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.similarity``.

> **Deprecated:** ``DocsToKG.HybridSearch.similarity`` is a compatibility shim that now
> forwards to ``DocsToKG.HybridSearch.vectorstore``. The module re-exports the GPU
> similarity helpers listed below and emits a :class:`DeprecationWarning` when imported.

## 1. Re-exported symbols

| Symbol | New home |
| ------ | -------- |
| ``normalize_rows`` | ``DocsToKG.HybridSearch.vectorstore`` |
| ``cosine_against_corpus_gpu`` | ``DocsToKG.HybridSearch.vectorstore`` |
| ``pairwise_inner_products`` | ``DocsToKG.HybridSearch.vectorstore`` |
| ``max_inner_product`` | ``DocsToKG.HybridSearch.vectorstore`` |

All function signatures remain unchanged. Update imports to reference the
``vectorstore`` module directly to avoid future breakage when the shim is
removed.
