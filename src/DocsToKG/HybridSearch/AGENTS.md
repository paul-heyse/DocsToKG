# Agents Guide - HybridSearch

Last updated: 2025-10-18

## Mission and Scope
- Mission: Deliver low-latency hybrid retrieval (lexical + dense) with deterministic fusion, scalable ingestion, and GPU-aware storage for DocsToKG.
- Scope boundary: In-scope—chunk ingestion, feature generation, FAISS/OpenSearch orchestration, fusion logic, API/service layer, observability. Out-of-scope—downstream answer generation, long-term document storage policy, embedding model training.

## High-Level Architecture & Data Flow
```mermaid
flowchart LR
  A[DocumentInput / ChunkPayload] --> B[ChunkIngestionPipeline]
  B --> C[ChunkRegistry + LexicalIndex]
  B --> D[ManagedFaissAdapter (FAISS)]
  subgraph Query Path
    Q[HybridSearchAPI] --> S[HybridSearchService]
    S --> C
    S --> D
    S --> F[ReciprocalRankFusion + MMR]
    F --> R[HybridSearchResponse]
  end
  D -.-> G[(Serialized snapshots)]:::cache
  classDef cache stroke-dasharray: 3 3;
```
- Components: ingestion pipeline (`pipeline.py`), dense store (`store.py`), namespace router (`router.py`), service/API (`service.py`), configuration manager (`config.py`), feature generators (`features.py`).
- Primary data edges: document artifacts → chunk features → lexical + dense indexes → fused results.
- One failure path to consider: GPU FAISS allocation failure during ingest triggers CPU fallback and increases latency—monitor `AdapterStats` and ensure snapshot restore works.

## Hot Paths & Data Shapes
- Hot paths:
  - `HybridSearchService.search()` orchestrates sparse/dense retrieval and fusion.
  - `ManagedFaissAdapter.search_many()` / `cosine_topk_blockwise()` for dense similarity.
  - `ChunkIngestionPipeline.ingest()` for bulk document onboarding.
  - `ResultShaper.shape()` for token/byte budget trimming.
- Typical payload sizes: embeddings 768–1536 dims (float32) (TODO verify actual dims per config); chunk payloads ~1–4 KB JSON; ingestion batches often 100–10k chunks.
- Key schemas/models: `ChunkPayload`, `ChunkFeatures`, `HybridSearchRequest/Response`, `DenseIndexConfig`, `FusionConfig`; ensure docstrings reflect required fields.

## Performance Objectives & Baselines
- Targets: TODO establish P50 < 120 ms, P95 < 250 ms for single-namespace search at top_k=20; ingestion throughput TODO (e.g., 5k chunks/min on GPU).
- Known baseline: TODO capture latest CI benchmark from `tests/hybrid_search/test_suite.py::test_hybrid_scale_suite`.
- Measurement recipes:
  ```bash
  direnv exec . pytest tests/hybrid_search/test_suite.py::test_hybrid_retrieval_end_to_end -q
  direnv exec . python -m cProfile -m DocsToKG.HybridSearch.service --profile-search '{"query":"hybrid search"}'
  ```

## Profiling & Optimization Playbook
- Quick profile:
  ```bash
  direnv exec . python -m cProfile -m DocsToKG.HybridSearch.service --profile-search '{"query":"example"}'
  direnv exec . pyinstrument -r html -o profile.html python -m DocsToKG.HybridSearch.service --profile-search '{"query":"example"}'
  direnv exec . pytest tests/hybrid_search/test_suite.py::test_hybrid_scale_suite --maxfail=1 -q
  ```
- Tactics:
  - Batch sparse + dense searches (reduce per-document dispatch over thread pool).
  - Use blockwise cosine (`cosine_topk_blockwise`) instead of naive loops; tune block size.
  - Normalize once (`normalize_rows`) and reuse across fusion stages.
  - Avoid repeated JSON encode/decode in hot loops (reuse `ChunkPayload` objects).
  - Cache query features when reranking similar queries; respect invalidation rules.
  - For GPU indices, pre-reserve memory (`DenseIndexConfig.expected_ntotal`) and use pinned buffers.

## Complexity & Scalability Guidance
- Retrieval complexity: O(k log k) for fusion after O(n) sparse/dense fetch (n = hits per channel), ensure `k` stays small (FusionConfig.max_chunks_per_doc).
- Ingestion complexity: O(N) over chunks with dedupe thresholds; memory proportional to chunk batch size.
- Memory growth: FAISS indices scale with `ntotal * dim * bytes`; monitor GPU VRAM and use IVFPQ when ntotal > ~1e6 (config).
- Large-N strategies: sharded namespaces via `FaissRouter(per_namespace=True)`; streaming ingestion with chunk batching, snapshot/restore to migrate indices.

## I/O, Caching & Concurrency
- I/O patterns: CPU/GPU FAISS operations, optional OpenSearch simulator for lexical; serialized snapshots persisted via `serialize_state`.
- Cache keys & invalidation: namespace-keyed FAISS router snapshots; chunk registry maps vector_id → payload and must stay in sync with dense store (remove/add pair).
- Concurrency: `HybridSearchService` uses `ThreadPoolExecutor` for channels; `ManagedFaissAdapter` guarded by locks; `FaissRouter` ensures thread-safe namespace creation. Avoid manual threading—use provided executors.

## Invariants to Preserve (change with caution)
- Stable mapping between vector UUIDs and FAISS ids (`_vector_uuid_to_faiss_int`).
- Fusion determinism: same inputs/config → same ordering (RRf/MMR results).
- Pagination guarantees: `verify_pagination` must reject unsupported page sizes/filters.
- Chunk registry consistency: lexical index and dense store must receive identical add/remove sequences.
- Observability budgets: `ResultShaper` enforces token/byte limits—never bypass without updating budget checks/tests.

## Preferred Refactor Surfaces
- Extend retrieval by implementing `DenseVectorStore` or `LexicalIndex` protocols in new modules and wiring via `HybridSearchConfigManager`.
- Improve metrics in `pipeline.Observability` or adapters in `store.ManagedFaissAdapter`.
- Add fusion heuristics inside `service.ReciprocalRankFusion` / `apply_mmr_diversification`.
- Avoid first-touch changes inside low-level FAISS wrappers unless you have extensive test coverage.

## Code Documentation Requirements
- Maintain NAVMAP headers in `service.py`, `pipeline.py`, `store.py`, `types.py`, etc.—update when adding sections.
- Public methods/classes require docstrings describing data contracts (`ChunkPayload`, `HybridSearchRequest`).
- Provide usage snippets for new API surfaces (similar to examples in `__init__.py` docstring).
- Follow `MODULE_ORGANIZATION_GUIDE.md`, `CODE_ANNOTATION_STANDARDS.md`, and `STYLE_GUIDE.md`; tests should assert docstring invariants when feasible.

## Test Matrix & Quality Gates
```bash
direnv exec . ruff check src/DocsToKG/HybridSearch tests/hybrid_search
direnv exec . mypy src/DocsToKG/HybridSearch
direnv exec . pytest tests/hybrid_search/test_suite.py -q
direnv exec . pytest tests/hybrid_search/test_suite.py::test_hybrid_scale_suite -q  # optional perf smoke
```
- TODO add GPU-targeted test markers for IVFPQ scenarios; ensure CPU fallback covered.
- Maintain fixtures under `tests/hybrid_search/fixtures/` (TODO create if missing) for deterministic regression.

## Failure Modes & Debug Hints
| Symptom | Likely cause | Quick checks |
|---|---|---|
| Search latency spikes | Dense store evicted to CPU or nprobe too high | Inspect `AdapterStats`; check `FaissRouter.stats()` for `evicted=True`; tune `DenseIndexConfig.nprobe`. |
| Missing highlights | Lexical index out of date after ingest | Verify `ChunkRegistry` vs lexical bulk_upsert; rerun ingestion pipeline for namespace. |
| GPU OOM during ingest | IVFPQ config mis-sized or replication enabled on large datasets | Lower `expected_ntotal`, disable replication, switch to CPU persist mode. |
| Pagination duplicates | `verify_pagination` disabled or cursor misuse | Ensure page_size <= config limit; run `test_hybrid_retrieval_end_to_end`. |

## Canonical Commands
```bash
# Ingest toy dataset and run search (example harness)
direnv exec . python examples/hybrid_search/ingest_and_search.py  # TODO script path

# Run full hybrid search test suite
direnv exec . pytest tests/hybrid_search/test_suite.py -q

# Snapshot and restore FAISS index (example)
direnv exec . python - <<'PY'
from DocsToKG.HybridSearch.store import serialize_state, restore_state, ManagedFaissAdapter
# TODO: fill in adapter initialization and snapshot usage
PY
```

## Indexing Hints
- Read first: `service.py` (search orchestration), `store.py` (FAISS adapter), `pipeline.py` (ingestion + metrics), `config.py` (knobs), `types.py` (data contracts).
- High-signal tests: `tests/hybrid_search/test_suite.py::test_hybrid_retrieval_end_to_end`, `test_operations_snapshot_and_restore_roundtrip`, `test_gpu_ivfpq_build_and_search`.
- Key schemas/contracts: ensure alignment with `HybridSearchRequest`, `HybridSearchResponse`, and config dataclasses.

## Ownership & Documentation Links
- Owners/reviewers: TODO_OWNERS (check root `CODEOWNERS` for `src/DocsToKG/HybridSearch/`).
- Additional docs: `src/DocsToKG/HybridSearch/README.md` (TODO if absent), architecture notes in `docs/` (TODO link).

## Changelog and Update Procedure
- Update this guide when adding new retrieval channels, changing fusion logic, or adjusting performance targets.
- Keep TODO placeholders in sync with actual metrics/config; bump `Last updated` after substantive edits.
