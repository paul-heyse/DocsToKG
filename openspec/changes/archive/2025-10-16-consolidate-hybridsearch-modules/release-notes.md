## HybridSearch module consolidation

- Ranking, result shaping, similarity kernels, and pagination/stats helpers now live alongside
  their primary modules (`ranking`, `vectorstore`, `service`).
- Deprecated modules (`operations`, `results`, `similarity`, `retrieval`, `schema`, `tools`) emit
  `DeprecationWarning` and will be removed in v0.6.0.
- The HybridSearch validation CLI is available through
  `python -m DocsToKG.HybridSearch.validation`; CI workflows have been updated accordingly.
- See `docs/hybrid_search_module_migration.md` for import path translations and upgrade guidance.
