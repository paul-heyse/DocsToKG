# Consolidate HybridSearch Modules – Analysis Notes

## 1. Module dependencies

The table below summarises direct intra-package imports (ignoring standard library and
third-party dependencies) across `src/DocsToKG/HybridSearch`.

| Module | Depends on |
|--------|------------|
| `config` | — |
| `features` | `config`, `types` |
| `ingest` | `config`, `observability`, `storage`, `types`, `vectorstore` |
| `observability` | `types` |
| `ranking` | `config`, `features`, `storage`, `types`, `vectorstore` |
| `service` | `config`, `features`, `observability`, `ranking`, `storage`, `types`, `vectorstore` |
| `storage` | `config`, `types` |
| `types` | — |
| `validation` | `config`, `features`, `ingest`, `observability`, `service`, `storage`, `types`, `vectorstore` |
| `vectorstore` | `config`, `types` |

These relationships maintain the desired DAG: foundational modules (`types`, `config`,
`storage`, `vectorstore`) sit at the bottom, while orchestration (`service`, `validation`)
build on top.

## 2. Package exports

The consolidated `__init__.py` exposes the following symbols:

- Service orchestration: `HybridSearchService`, `HybridSearchAPI`,
  `HybridSearchValidator`, `PaginationCheckResult`, `verify_pagination`,
  `build_stats_snapshot`, `should_rebuild_index`.
- Configuration and utilities: `HybridSearchConfig`, `HybridSearchConfigManager`,
  `FeatureGenerator`, `ChunkIngestionPipeline`, `Observability`.
- Storage/state: `OpenSearchIndexTemplate`, `OpenSearchSchemaManager`,
  `FaissIndexManager`, `serialize_state`, `restore_state`.
- Ranking: `ReciprocalRankFusion`, `apply_mmr_diversification`, `ResultShaper`.
- Types: `ChunkPayload`, `DocumentInput`, `HybridSearchRequest`,
  `HybridSearchResponse`, `HybridSearchResult`, `vector_uuid_to_faiss_int`.

## 3. Tests touching deprecated modules

`tests/hybrid_search/test_suite.py` is the canonical suite. All direct imports now target
the consolidated modules. Explicit warning checks cover the shims for
`operations`, `results`, `similarity`, `retrieval`, and `schema` to ensure users receive
migration guidance.

## 4. CI references

No GitHub workflow referenced the deleted `HybridSearch/tools/*` scripts. The main
pipeline (`.github/workflows/ci.yml`) will now invoke the unified validation CLI (see
implementation updates).

## 5. Deprecation tracking

See `openspec/changes/consolidate-hybridsearch-modules/deprecation-tracking.md` for the
removal plan, migration messaging, and release milestones.
