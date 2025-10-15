## 1. FAISS Index Construction
- [x] 1.1 Replace manual CPU→GPU build logic in `dense.py` with factory/GPU one-liners and retain `IndexIDMap2` wrapping.
- [x] 1.2 Apply IVF parameters (`nprobe`, future PQ knobs) through `GpuParameterSpace`/`ParameterSpace` immediately after index creation and log the effective values.

## 2. ID Lifecycle & Delete Path
- [x] 2.1 Swap `_remove_ids` to use `faiss.IDSelectorBatch`, preserve CPU fallback, and clean up bookkeeping in one place.
- [x] 2.2 Introduce a FAISS-ID→vector-ID bridge inside `ChunkRegistry`, refactor `FaissIndexManager` to drop `_id_lookup`, and adjust serialization/restore accordingly.

## 3. GPU-Assisted Similarity
- [x] 3.1 Update MMR diversification to batch compute cosine similarities via `faiss.pairwise_distance_gpu`, with NumPy fallback for CPU-only runs.
- [x] 3.2 Rework `ResultShaper` duplicate detection to batch compare embeddings using the same GPU helper.

## 4. Sparse Search Simplification
- [x] 4.1 Introduce `_search_sparse` in `storage.py` and refactor BM25/SPLADE search methods to delegate scoring into lambdas.
- [x] 4.2 Ensure pagination/tests still assert identical behaviour after refactor.

## 5. Tests & Docs
- [x] 5.1 Extend/adjust unit tests covering FAISS add/remove, serialization, GPU batching fallbacks, and MMR/duplicate behaviour to reflect the new paths and optional dataset-driven dense queries.
- [x] 5.2 Update developer documentation/README references to point to the relocated hybrid CLI tools under `src/DocsToKG/HybridSearch/tools/` and note the centralised FAISS tuning.
