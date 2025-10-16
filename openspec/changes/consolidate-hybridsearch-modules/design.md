# Consolidate HybridSearch Module Structure - Design Document

## Context

The HybridSearch subsystem provides GPU-accelerated hybrid retrieval combining BM25 lexical search, SPLADE sparse embeddings, and dense vector similarity search using FAISS. The current implementation spans 12 modules organized as follows:

**Current structure:**

- `__init__.py`: Public API exports
- `config.py`: Configuration dataclasses and manager (4 config types + thread-safe manager)
- `features.py`: Query featurization (tokenization, BM25 weights, SPLADE weights, dense embeddings)
- `ingest.py`: Dual-write pipeline coordinating FAISS and OpenSearch upserts with ID resolution
- `observability.py`: Metrics collection, tracing spans, and structured logging
- `operations.py`: Operational utilities (pagination verification, stats snapshots, state serialization)
- `ranking.py`: Reciprocal Rank Fusion (RRF) and Maximal Marginal Relevance (MMR) diversification
- `results.py`: Result shaping (deduplication, per-document quotas, highlight generation)
- `retrieval.py`: Thin orchestration shim delegating to `service.py`
- `service.py`: Main orchestrator (`HybridSearchService` executing BM25/SPLADE/dense channels + HTTP adapter)
- `similarity.py`: GPU-accelerated cosine similarity and inner product kernels
- `storage.py`: OpenSearch simulator for lexical storage + index templates
- `types.py`: Request/response models, payloads, and exception types
- `validation.py`: Comprehensive validation harness with scale testing capabilities
- `vectorstore.py`: FAISS index manager with GPU promotion, remove-ids probing, and lifecycle management
- `Tools/`: CLI scripts duplicating validation entry points

**Architectural challenges:**

1. **Thin adapter modules create unnecessary indirection**: `retrieval.py`, `results.py`, `similarity.py`, and partially `operations.py` exist primarily to route calls between other modules without adding substantial domain logic. This obscures the actual control flow and data dependencies.

2. **GPU resource ownership ambiguity**: GPU device handles and `faiss.StandardGpuResources` instances are threaded through multiple module boundaries (`service → ranking → shaper`, `service → vectorstore → similarity`), creating opportunities for incorrect device placement or resource lifecycle mismanagement.

3. **Functional cohesion violations**: Result shaping (`results.py`) is semantically part of the ranking pipeline since both operate on fused scores and use identical GPU parameters for similarity-based deduplication. Splitting them forces unnecessary parameter propagation.

4. **Operational function dispersion**: `operations.py` mixes service-level concerns (pagination, stats) with FAISS state management (serialization), forcing it to import from both `service` and `vectorstore` and creating bidirectional coupling.

5. **CLI entry point duplication**: The `Tools/` directory contains scripts that replicate functionality already present in `validation.py`, requiring maintenance of parallel code paths and divergent argument parsing logic.

## Goals

**Primary goals:**

1. **Reduce module count to ≤9** while maintaining clear separation of concerns, enabling developers to locate functionality within two levels of navigation (package → module → class/function).

2. **Centralize GPU resource management** to a single authoritative source (`vectorstore.py` owning FAISS resources) with explicit device/resource parameter passing to consumers, eliminating hidden resource state.

3. **Unify functionally cohesive responsibilities** by collocating ranking and result shaping (both operate on fused scores with GPU deduplication) and consolidating all FAISS-related GPU math (similarity kernels) near the index lifecycle.

4. **Eliminate adapter modules** by moving their functionality into the modules they primarily serve, reducing import chain depth and making data flow explicit.

5. **Preserve backward compatibility** for one release cycle using deprecation shims, allowing external code to migrate gradually without immediate breakage.

6. **Maintain GPU-first execution model** where all FAISS operations and similarity computations remain on GPU without CPU fallback paths introduced by the refactoring.

**Non-goals:**

1. **Not changing public API signatures**: All exported functions, classes, and methods retain identical signatures including parameter names, types, and ordering.

2. **Not altering behavioral semantics**: Search result ordering, scoring algorithms, deduplication thresholds, and GPU compute correctness must remain bit-exact.

3. **Not introducing new dependencies**: The consolidation uses only existing module relationships in different configurations; no new external packages required.

4. **Not optimizing performance**: While performance must not regress, this change does not attempt to improve latency, throughput, or memory usage beyond what naturally emerges from reduced import overhead.

5. **Not refactoring internal algorithms**: RRF scoring, MMR diversification, BM25 weighting, and FAISS index construction logic remain unchanged; only their module locations change.

## Decisions

### Decision 1: Merge `results.py` into `ranking.py`

**Rationale:**
Both `ResultShaper` (in `results.py`) and `apply_mmr_diversification` (in `ranking.py`) accept identical `device` and `resources` parameters for GPU-based similarity operations. They operate sequentially in the search pipeline (MMR diversification → result shaping) and share conceptual ownership of "producing the final ranked result list." Collocating them reduces parameter threading and makes the ranking-to-output transformation explicit.

**Implementation approach:**

- Move `ResultShaper` class verbatim to `ranking.py`, placing it after `apply_mmr_diversification` to maintain pipeline order clarity.
- Update `service.py` import from `from .results import ResultShaper` to `from .ranking import ResultShaper`.
- Convert `results.py` to a shim: `from .ranking import ResultShaper  # noqa: F401` with `warnings.warn("...", DeprecationWarning)`.
- Verify tests in `test_ranking.py` or equivalent cover result shaping behavior.

**Alternatives considered:**

- **Keep separate:** Would preserve narrower module scope but perpetuate the parameter threading issue and obscure the functional pipeline.
- **Merge both into `service.py`:** Rejected because ranking/shaping logic should remain independent of orchestration concerns; service coordinates channels but shouldn't contain scoring algorithms.

### Decision 2: Integrate `similarity.py` into `vectorstore.py`

**Rationale:**
The GPU similarity kernels (`cosine_against_corpus_gpu`, `pairwise_inner_products`, `max_inner_product`, `normalize_rows`) are tightly coupled to FAISS resource management. They require `faiss.StandardGpuResources` instances that only `FaissIndexManager` creates, and their correctness depends on proper GPU device selection. Housing them in `vectorstore.py` makes GPU resource lifecycle explicit: the module that creates GPU resources also provides the compute functions using those resources.

**Implementation approach:**

- Move all four similarity functions to `vectorstore.py`, placing them after `FaissIndexManager` class definition in a "GPU-accelerated similarity operations" section.
- Update imports in `ranking.py` (uses `pairwise_inner_products` for MMR) and `validation.py` (uses all functions for metrics).
- Ensure functions remain module-level (not methods of `FaissIndexManager`) to avoid circular dependencies if other code needs standalone similarity without instantiating the full index.
- Convert `similarity.py` to re-export shim with deprecation warning.

**Alternatives considered:**

- **Create dedicated `gpu_ops.py` module:** Rejected because it would introduce a new module (increasing count) and still require coupling to FAISS resources.
- **Make similarity functions methods of `FaissIndexManager`:** Rejected because validation and ranking code would need index instances just to compute similarity, creating awkward lifecycle dependencies.

### Decision 3: Distribute `operations.py` functions by responsibility

**Rationale:**
`operations.py` currently mixes two orthogonal concerns: service-level operations that orchestrate across multiple storage backends (pagination verification, stats snapshots, rebuild heuristics) and FAISS-specific state management (serialization/restoration). The former belong conceptually in `service.py` since they coordinate service behavior, while the latter belong in `vectorstore.py` since they manipulate FAISS internal state.

**Implementation approach:**

- **Service-level ops to `service.py`:**
  - `verify_pagination(service, request)` – uses `HybridSearchService` to execute queries
  - `build_stats_snapshot(faiss, opensearch, registry)` – aggregates stats from multiple backends
  - `should_rebuild_index(registry, deleted, threshold)` – heuristic for service lifecycle decisions
- **State management to `vectorstore.py`:**
  - `serialize_state(faiss_index, registry)` – captures FAISS bytes and registered IDs
  - `restore_state(faiss_index, payload)` – rebuilds FAISS from serialized bytes
- Expose both sets of functions publicly from their respective modules for backward compatibility.
- `operations.py` becomes a shim with conditional re-exports: `from .service import verify_pagination, ...` and `from .vectorstore import serialize_state, ...`.

**Alternatives considered:**

- **Keep `operations.py` unified:** Rejected because it forces the module to import from both `service` and `vectorstore`, creating bidirectional coupling and obscuring ownership.
- **Move everything to `service.py`:** Rejected because FAISS state serialization is a vectorstore concern, not a service orchestration concern; mixing would violate separation of concerns.

### Decision 4: Merge `schema.py` into `storage.py` (if separate)

**Rationale:**
OpenSearch index template management (`OpenSearchIndexTemplate`, `OpenSearchSchemaManager`) is intrinsically linked to the storage layer. Templates define the schema that `OpenSearchSimulator` uses for document indexing and search operations. Collocating them in `storage.py` creates a single lexical boundary for "everything related to OpenSearch," reducing the need to navigate between schema definition and schema usage.

**Implementation approach:**

- Verify `schema.py` exists as a separate file; some implementations may already have schema integrated into `storage.py`.
- If separate, move `OpenSearchIndexTemplate` dataclass and `OpenSearchSchemaManager` class to `storage.py` under an "Index template management" comment header.
- Update imports in `ingest.py`, `service.py`, or tests from `from .schema import ...` to `from .storage import ...`.
- Convert `schema.py` to deprecation shim if it existed.

**Alternatives considered:**

- **Keep schema separate:** Would maintain narrower module scope but requires developers to understand the schema-storage relationship across two files, increasing cognitive load for OpenSearch modifications.

### Decision 5: Eliminate `retrieval.py` by consolidating into `service.py`

**Rationale:**
`retrieval.py` currently acts as a thin shim that delegates all actual work to `HybridSearchService` in `service.py`. The orchestration logic, channel execution, fusion, and HTTP adapter already reside in `service.py`, making the separate retrieval module redundant. Consolidating eliminates an import hop without losing any functionality.

**Implementation approach:**

- Review `retrieval.py` to confirm it contains no unique logic not already in `service.py`.
- If any functions exist, move them to `service.py` under appropriate method or helper function.
- Update all imports from `from .retrieval import ...` to `from .service import ...`.
- Convert `retrieval.py` to minimal shim: `from .service import HybridSearchService, HybridSearchAPI, ...  # noqa: F401` with deprecation warning.

**Alternatives considered:**

- **Keep retrieval as the sole public entry point:** Rejected because it creates unnecessary layering; `service.py` is already structured as the orchestrator and should be the canonical entry point.

### Decision 6: Retire `Tools/` directory and consolidate CLI entry points

**Rationale:**
The `Tools/` directory contains scripts (`run_hybrid_tests.py`, `run_real_vector_ci.py`) that duplicate functionality already present in `validation.py`. Maintaining parallel entry points creates divergence risk (argument parsing differences, missing features, inconsistent output formats) and confuses users about which script to invoke. Consolidating into `validation.py:main()` with mode flags or subcommands provides a single, well-tested entry point.

**Implementation approach:**

- Review functionality in `run_hybrid_tests.py`: likely invokes basic validation suite.
  - Ensure `validation.py:main()` with `--mode basic` provides equivalent functionality.
- Review `run_real_vector_ci.py`: likely performs CI-specific checks (e.g., GPU smoke tests, real vector fixtures).
  - Add `--mode ci` or equivalent flag to `validation.py:main()` to cover these scenarios.
- Update CI workflow files to invoke `python -m DocsToKG.HybridSearch.validation [args]` instead of `python HybridSearch/tools/run_*.py [args]`.
- Delete `Tools/__init__.py`, `Tools/run_hybrid_tests.py`, `Tools/run_real_vector_ci.py`.
- Remove `Tools/` directory entirely.

**Alternatives considered:**

- **Keep Tools as legacy compatibility layer:** Rejected because it perpetuates maintenance burden; better to have a deprecation cycle and migrate all callers.
- **Move tools to separate `cli/` directory:** Rejected because `validation.py` already serves as the CLI entry point; creating another directory increases module count.

### Decision 7: Use deprecation shims for backward compatibility

**Rationale:**
External code may import from deprecated module paths. Immediate removal would break existing integrations without warning. Python's module aliasing capabilities allow us to preserve import paths while guiding users toward new locations through `DeprecationWarning` messages.

**Implementation approach:**

- For each deprecated module file, replace contents with:

  ```python
  from .<new_module> import <symbols>  # noqa: F401
  import warnings as _warnings
  _warnings.warn(
      "Module DocsToKG.HybridSearch.<old_module> is deprecated; "
      "import from DocsToKG.HybridSearch.<new_module> instead. "
      "This shim will be removed in version X.Y.Z.",
      DeprecationWarning,
      stacklevel=2
  )
  ```

- Update `__init__.py` to use `sys.modules` aliasing for complete module path compatibility:

  ```python
  import sys
  sys.modules[__name__ + ".similarity"] = sys.modules[__name__ + ".vectorstore"]
  sys.modules[__name__ + ".results"] = sys.modules[__name__ + ".ranking"]
  # etc.
  ```

- Document deprecation timeline (one release cycle) in `CHANGELOG.md` and migration guide.

**Alternatives considered:**

- **Hard cutover without deprecation period:** Rejected because it breaks existing integrations and violates semantic versioning principles (backward incompatible change requires major version bump).
- **Permanent shims:** Rejected because it perpetuates technical debt; better to guide users to new paths and remove shims after transition period.

## Architecture

### Post-consolidation module structure

**Final 9 modules:**

```
HybridSearch/
├── __init__.py          # Public API exports + deprecation module aliasing
├── config.py            # {Chunking, Dense, Fusion, Retrieval} configs + manager
├── features.py          # Tokenization + query featurization (BM25/SPLADE/dense)
├── ingest.py            # ChunkIngestionPipeline (dual writes + ID resolution)
├── observability.py     # MetricsCollector + TraceRecorder + Observability facade
├── ranking.py           # RRF + MMR + ResultShaper (unified ranking pipeline)
├── service.py           # HybridSearchService + API + service-level ops
├── storage.py           # OpenSearchSimulator + templates + filters
├── types.py             # Request/response models + ChunkPayload + errors
├── validation.py        # HybridSearchValidator + CLI entry point
└── vectorstore.py       # FaissIndexManager + GPU similarity ops + state mgmt
```

**Dependency graph (unidirectional):**

```
types ← (all modules)
config ← service, vectorstore, ranking, ingest
vectorstore ← ingest, service, ranking, validation
storage ← ingest, service, ranking
ranking ← service
features ← service
observability ← ingest, service
ingest ← (tests/validation only, not service)
service ← (validation, HTTP layer; no imports into other modules)
validation ← (CLI/test runner only)
```

### GPU resource ownership model

**Single source of truth:**

- `FaissIndexManager` in `vectorstore.py` creates and owns `faiss.StandardGpuResources` instance.
- All GPU similarity functions in `vectorstore.py` accept `resources` as explicit parameter.
- `service.py` passes `faiss_manager.gpu_resources` and `faiss_manager.device` to ranking/shaping functions.

**No hidden state:**

- No module-level GPU resource caching or singletons.
- Every function accepting `resources` parameter documents GPU requirement in docstring.
- Tests explicitly instantiate GPU resources or use CPU-only code paths.

### Critical invariants

1. **Import direction:** No circular imports; `types` and `config` are leaves, `service` is root (only imported by CLI/tests).

2. **GPU compute locality:** All FAISS operations and similarity kernels in `vectorstore.py`; no GPU code scattered across modules.

3. **Backward compatibility:** Deprecated import paths work with warnings until explicit removal release.

4. **Behavioral equivalence:** Search results, scores, and GPU device placement identical before and after consolidation (verified via pytest suite and validation harness).

5. **Test coverage:** No coverage regression; tests updated to use new import paths but exercise identical code paths.

## Risks and Trade-offs

### Risk 1: Import chain disruption during transition

**Description:** Tests and external code may fail during development if imports updated non-atomically.

**Mitigation:**

- Implement deprecation shims immediately when moving functions, before updating any test imports.
- Use feature branch workflow with comprehensive CI run before merging.
- Update imports in small, isolated commits (one module consolidation per commit).

### Risk 2: Performance regression from module reorganization

**Description:** Python import caching or module initialization order changes could theoretically impact startup time or first-query latency.

**Mitigation:**

- Run performance benchmarks comparing pre-consolidation and post-consolidation timing.
- Use profiling tools (cProfile, py-spy) to identify any unexpected overhead.
- If regression detected, investigate and resolve before merging (highly unlikely given Python's import caching).

### Risk 3: GPU resource lifecycle bugs introduced by parameter threading changes

**Description:** Consolidating GPU similarity functions into `vectorstore.py` changes their import paths, potentially causing incorrect device placement if callers don't update parameter passing.

**Mitigation:**

- Comprehensive GPU test suite already validates device placement for ranking/MMR/shaping.
- Static type checking (mypy) catches missing or incorrect `resources` parameters.
- Validation harness with `--mode scale` exercises GPU codepaths under realistic load.

### Risk 4: Breaking external integrations that import from deprecated paths

**Description:** Users may rely on specific module paths in production code; deprecation cycle helps but doesn't eliminate breakage risk.

**Mitigation:**

- Deprecation warnings loudly signal migration need.
- Provide detailed migration guide with old-to-new path mappings.
- Support deprecated paths for full release cycle (e.g., v0.5.x keeps shims, v0.6.0 removes them).
- Announce changes prominently in release notes and communication channels.

### Trade-off 1: Larger module files vs. more modules

**Choice:** Consolidation creates larger files (`ranking.py` gains `ResultShaper`, `vectorstore.py` gains similarity functions).

**Rationale:** Modern editors with symbol navigation (LSP, IDE indexing) make large files manageable. The cognitive benefit of functional cohesion (all ranking logic in one place) outweighs the navigation cost of scrolling through a longer file.

### Trade-off 2: Deprecation maintenance burden vs. immediate breaking change

**Choice:** Maintain deprecation shims for one release cycle.

**Rationale:** While shims require testing and documentation, the alternative (immediate breakage) violates semantic versioning and harms user trust. The temporary maintenance cost is acceptable for smooth migration.

## Migration Plan

### Phase 1: Implementation (current change)

1. Execute tasks 1-11 in `tasks.md` to consolidate modules, add shims, and update tests.
2. Validate via CI pipeline (CPU and GPU test suites) and validation harness.
3. Merge to main branch as feature-flagged or under development branch until fully validated.

### Phase 2: Deprecation period (next release, e.g., v0.5.0)

1. Ship consolidated modules with deprecation shims active.
2. Announce deprecation in release notes, CHANGELOG, and documentation.
3. Provide migration guide with import path mappings.
4. Monitor user feedback and address any unforeseen compatibility issues.

### Phase 3: Removal (subsequent release, e.g., v0.6.0)

1. Delete deprecation shim files (`retrieval.py`, `results.py`, `similarity.py`, `operations.py`, `schema.py` if existed).
2. Remove `sys.modules` aliasing from `__init__.py`.
3. Update documentation to remove references to deprecated paths.
4. Announce removal in release notes as breaking change (major or minor version bump per semver).

## Open Questions

**None at this time.** All architectural decisions finalized based on analysis of current codebase structure and import patterns. Implementation can proceed directly from this design.
