# 1. Hybrid Search Module Migration Guide

The `consolidate-hybridsearch-modules` change collapses thin adapter modules into their
functional homes while preserving backwards compatibility through explicit shims. This
guide summarises the deprecated paths, the new module locations, and the removal timeline
so downstream projects can update ahead of the DocsToKG v0.6.0 release.

## 1. Deprecated import paths

| Deprecated path | Replacement | Notes |
|-----------------|-------------|-------|
| `DocsToKG.HybridSearch.operations` | `DocsToKG.HybridSearch.service` (pagination & stats)<br>`DocsToKG.HybridSearch.vectorstore` (state serialisation) | Emits `DeprecationWarning` on import.
| `DocsToKG.HybridSearch.results` | `DocsToKG.HybridSearch.ranking` | Result shaping now lives alongside fusion/MMR helpers.
| `DocsToKG.HybridSearch.similarity` | `DocsToKG.HybridSearch.vectorstore` | GPU similarity helpers are co-located with FAISS resource owners.
| `DocsToKG.HybridSearch.retrieval` | `DocsToKG.HybridSearch.service` | Service orchestrator is the canonical entry point.
| `DocsToKG.HybridSearch.schema` | `DocsToKG.HybridSearch.storage` | Index templates share the storage module.
| `DocsToKG.HybridSearch.tools.*` | `python -m DocsToKG.HybridSearch.validation` | Validation CLI consolidates all modes.

## 2. Module dependency graph

```mermaid
graph LR
    types --> config
    types --> storage
    types --> vectorstore
    config --> ingest
    config --> service
    config --> vectorstore
    config --> validation
    features --> ingest
    features --> service
    observability --> ingest
    observability --> service
    storage --> ingest
    storage --> service
    vectorstore --> ingest
    vectorstore --> service
    vectorstore --> validation
    ranking --> service
    ingest --> service
    service --> validation
```

## 3. Removal timeline

- **DocsToKG v0.5.x** – Shims remain available and emit warnings.
- **DocsToKG v0.6.0** – Deprecated modules will be removed. Update imports before upgrading.

## 4. Recommended actions

1. Replace legacy imports with the new module locations shown above.
2. Update CI workflows to invoke `python -m DocsToKG.HybridSearch.validation` instead of
   `HybridSearch/tools/*.py`.
3. Adjust documentation, notebooks, and automation scripts to reference the consolidated
   modules so warning logs remain clean.

Reach out on the DocsToKG discussion board if additional migration assistance is required.
