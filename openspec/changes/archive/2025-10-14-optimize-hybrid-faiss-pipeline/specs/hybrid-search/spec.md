## MODIFIED Requirements

### Requirement: FAISS Index Provisioning
The system SHALL provision GPU FAISS indexes for 2560-d embeddings using cosine similarity via L2 normalization and Inner Product metrics, supporting GpuIndexFlatIP for exact search and GpuIndexIVFFlat or GpuIndexIVFPQ for ANN with train-before-add semantics. Index construction SHALL leverage FAISS factory strings or GPU-native constructors to minimize boilerplate, enforce `IndexIDMap2` wrapping for explicit IDs, and immediately apply IVF parameters (`nprobe`, PQ knobs) through FAISS’ parameter space helpers so configuration changes take effect without per-query overrides.

#### Scenario: Initialize FAISS index with factory helpers
- **GIVEN** StandardGpuResources and configuration indicating Flat or IVF mode
- **WHEN** the retrieval service starts
- **THEN** the FAISS index is built via the factory/GPU helper APIs in one pass, wrapped in `IndexIDMap2`, and IVF-specific parameters (e.g., `nprobe`) are set through FAISS’ parameter space utilities before serving queries
- **AND** construction logs report the effective index type and tuning parameters for observability

### Requirement: Batch Upsert and Delete Semantics
Upsert workflows SHALL batch vectors (e.g., 50k–100k) to control GPU memory pressure, normalize embeddings with `faiss.normalize_L2`, remove existing IDs before re-adding, and ensure deletions call both OpenSearch delete and `remove_ids` in FAISS. Delete operations SHALL rely on FAISS’ batch ID selector APIs with CPU fallback when GPU indexes raise `remove_ids not implemented`, and internal ID bookkeeping SHALL be delegated to the chunk registry so serialized index payloads do not duplicate state.

#### Scenario: Remove vectors with streamlined selector
- **GIVEN** a FAISS index running on GPU that raises `remove_ids not implemented`
- **WHEN** the ingestion pipeline re-ingests a batch that requires deleting existing vector IDs
- **THEN** the manager issues a batch ID selector, transparently downgrades to CPU on unsupported GPU paths, and re-promotes the index
- **AND** the chunk registry remains the single source of mapping truth for vector IDs

### Requirement: Result Shaping and Diversification
Result shaping SHALL collapse duplicate doc_ids, enforce chunk-per-doc budgets, provide per-result diagnostics, and dedupe candidates using cosine thresholds. When diversification (MMR) is enabled, cosine similarity computations SHALL use FAISS’ GPU pairwise distance helpers when available (with CPU fallbacks) to efficiently score redundancy across the candidate set.

#### Scenario: Diversify candidates with GPU cosine batching
- **GIVEN** a diversified hybrid search request with sufficient GPU resources
- **WHEN** MMR executes over the fused candidate list
- **THEN** cosine similarities are computed in bulk via FAISS pairwise distance helpers, producing the same ranking quality faster than per-result NumPy loops
- **AND** the pipeline falls back to NumPy when GPU helpers are unavailable

### Requirement: Hybrid Retrieval Storage
The system SHALL persist chunk-level content in OpenSearch for sparse retrieval and in GPU FAISS indexes for dense retrieval, using a shared `vector_id` (UUID) across both systems via IndexIDMap2. Implementation SHALL avoid redundant identity maps by sourcing FAISS int→vector mappings from the chunk registry and shall expose sparse search helpers that reuse filter/sort/paginate logic for BM25 and SPLADE to reduce duplication.

#### Scenario: Execute sparse search through shared helper
- **GIVEN** a BM25 or SPLADE query with namespace filters
- **WHEN** the simulator executes the sparse search
- **THEN** it passes a scoring lambda into a shared sparse search helper that handles filtering, score aggregation, and pagination identically for both BM25 and SPLADE
- **AND** pagination semantics remain unchanged from the previous implementation
