# Consolidate HybridSearch Module Structure

## Why

The current HybridSearch implementation spans 12 modules plus a Tools directory containing redundant CLI scripts. This architectural sprawl creates several critical issues that undermine maintainability and developer productivity:

1. **Cross-module dependency fragmentation**: Multiple thin adapter modules (`retrieval.py`, `results.py`, `similarity.py`, `operations.py`) exist solely to bridge between core functional modules, creating unnecessary import chains and obscuring the actual data flow through the hybrid search pipeline.

2. **GPU resource management opacity**: GPU device and resource handles are passed through multiple module boundaries without a clear ownership model, making it difficult to reason about resource lifecycle and increasing the risk of resource leaks or incorrect device placement.

3. **Cognitive overhead for contributors**: New contributors must navigate excessive module boundaries to understand even simple operations like "how does a search request flow through the system" or "where are FAISS similarity operations performed."

4. **Testing and CI maintenance burden**: The Tools directory duplicates validation entry points that could be unified, while the module proliferation requires extensive mocking and setup boilerplate in tests.

The consolidation addresses these issues by collapsing functionally cohesive responsibilities into fewer, more self-contained modules that respect clear architectural boundaries.

## What Changes

This change consolidates the HybridSearch module structure from 12 modules down to 9 core modules, eliminating adapter modules and clarifying ownership boundaries:

1. **Eliminate `retrieval.py`**: Merge orchestration logic into `service.py` where `HybridSearchService` and `HybridSearchAPI` already reside, creating a single entry point for in-process and HTTP-style search execution.

2. **Merge `results.py` into `ranking.py`**: Collocate `ResultShaper` (responsible for deduplication, per-document quotas, and highlight generation) with `ReciprocalRankFusion` and `apply_mmr_diversification`, unifying all ranking and result shaping concerns that share GPU device/resource parameters.

3. **Integrate `similarity.py` into `vectorstore.py`**: Move GPU-based cosine similarity helpers (`cosine_against_corpus_gpu`, `pairwise_inner_products`, `max_inner_product`, `normalize_rows`) into the FAISS index manager module, centralizing all GPU computation near the FAISS resource lifecycle.

4. **Distribute `operations.py` functions**:
   - Service-level operations (`verify_pagination`, `build_stats_snapshot`, `should_rebuild_index`) move to `service.py`
   - FAISS state management (`serialize_state`, `restore_state`) moves to `vectorstore.py` as public functions

5. **Merge `schema.py` with `storage.py`**: Unify OpenSearch index template management (`OpenSearchIndexTemplate`, `OpenSearchSchemaManager`) with the `OpenSearchSimulator` to create a single lexical storage boundary.

6. **Retire `Tools/` directory**: Remove redundant CLI scripts (`run_hybrid_tests.py`, `run_real_vector_ci.py`) and consolidate their functionality into `validation.py` as module entry points accessible via `python -m DocsToKG.HybridSearch.validation`.

7. **Maintain backward compatibility**: Preserve deprecated module files as re-export shims with `DeprecationWarning` messages for one release cycle, ensuring existing import statements continue to function while guiding users to the new locations.

The consolidation SHALL NOT alter any public API signatures, data structures, or behavioral semantics of the HybridSearch system. GPU resource management patterns, FAISS index lifecycle, and search result ordering SHALL remain identical.

## Impact

**Affected specs:**

- `hybrid-search` (new specification being created as part of this change)

**Affected code:**

- Core modules: `src/DocsToKG/HybridSearch/*.py` (all modules undergo import path updates)
- Public interface: `src/DocsToKG/HybridSearch/__init__.py` (re-exports updated to maintain backward compatibility)
- Tests: `tests/hybrid_search/*.py` (import statements updated to reflect new module structure)
- CI configuration: Any scripts in `.github/workflows/` or CI tooling that invoke `HybridSearch/tools/*` scripts
- Documentation: `docs/hybrid_search*.md` files referencing module organization

**Breaking changes:**
None for external consumers during the deprecation period. After the deprecation period (one release cycle), imports from deprecated module paths will fail with `ImportError`.

**Migration path:**
For users importing from deprecated modules:

- `from DocsToKG.HybridSearch.retrieval import ...` → `from DocsToKG.HybridSearch.service import ...`
- `from DocsToKG.HybridSearch.results import ResultShaper` → `from DocsToKG.HybridSearch.ranking import ResultShaper`
- `from DocsToKG.HybridSearch.similarity import ...` → `from DocsToKG.HybridSearch.vectorstore import ...`
- `from DocsToKG.HybridSearch.operations import verify_pagination, ...` → `from DocsToKG.HybridSearch.service import ...`
- `from DocsToKG.HybridSearch.operations import serialize_state, ...` → `from DocsToKG.HybridSearch.vectorstore import ...`
- `from DocsToKG.HybridSearch.schema import ...` → `from DocsToKG.HybridSearch.storage import ...`

CI scripts SHALL update to invoke `python -m DocsToKG.HybridSearch.validation [args]` instead of executing Tools directory scripts directly.
