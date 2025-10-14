## ADDED Requirements

### Requirement: Hybrid Retrieval Storage
The system SHALL persist chunk-level content in OpenSearch for sparse retrieval and in GPU FAISS indexes for dense retrieval, using a shared `vector_id` (UUID) across both systems via IndexIDMap2.

#### Scenario: Ingest chunk into dual indexes
- **GIVEN** a chunk payload that includes doc metadata, SPLADEv3 weights, and a normalized Qwen3-4B embedding
- **WHEN** the upsert pipeline runs
- **THEN** the chunk is stored in the OpenSearch index with its metadata and sparse features
- **AND** the same chunk vector is inserted into the FAISS index under its `vector_id`

### Requirement: Chunk Schema and Index Configuration
OpenSearch mappings SHALL include fields for `doc_id`, `chunk_id`, `namespace`, `text`, `splade` rank_features, `vector_id`, and filterable metadata such as `author`, `tags`, `published_at`, and ACL fields. Chunking SHALL default to 400–800 token windows with 100–200 token overlaps, configurable per corpus.

#### Scenario: Bootstrap OpenSearch mappings
- **GIVEN** a new namespace onboarding
- **WHEN** the schema bootstrap routine executes
- **THEN** the OpenSearch index is created with the required keyword, text, rank_features, and numeric/date fields
- **AND** chunking parameters are recorded so ingest jobs emit overlapping windows within the specified bounds

### Requirement: Chunk Processing and Feature Generation Pipeline
The ingestion service SHALL accept raw documents, apply the configured chunker, and emit chunk payloads that include deterministic UUIDs, normalized metadata, SPLADE token/weight maps, and Qwen3-4B dense embeddings generated through approved model helpers. Failed chunk transformations SHALL surface retryable vs terminal errors distinctly and shall never emit partial chunks for a document.

#### Scenario: Transform document into enriched chunks
- **GIVEN** a document with raw text and metadata
- **WHEN** the pipeline processes the document
- **THEN** it produces chunk records whose UUIDs remain stable across re-processing, with SPLADE/BM25 features derived from the shared tokenizer, normalized Qwen embeddings, and sanitized metadata aligned to the OpenSearch schema
- **AND** any failure during SPLADE or dense embedding generation marks the entire document batch for retry while logging trace context for dead-letter handling

### Requirement: Configuration and Parameter Management
The system SHALL expose configuration files or service-level configuration for choosing FAISS index type (Flat, IVF-Flat, IVFPQ), setting IVF parameters (`nlist`, `nprobe`, PQ `M`, `nbits`), chunk window sizes, oversampling ratios, fusion `k0`, MMR λ, cosine dedupe thresholds, and namespace-to-index mappings. Configuration changes SHALL be reloadable without redeploying the service.

#### Scenario: Adjust retrieval parameters without outage
- **GIVEN** an operator updates the retrieval configuration to raise `nprobe` and adjust MMR λ
- **WHEN** the configuration reload endpoint or watcher triggers
- **THEN** the running service applies the new parameters to subsequent searches without downtime and persists the effective configuration for audit

### Requirement: FAISS Index Provisioning
The system SHALL provision GPU FAISS indexes for 2560-d embeddings using cosine similarity via L2 normalization and Inner Product metrics, supporting GpuIndexFlatIP for exact search and GpuIndexIVFFlat or GpuIndexIVFPQ for ANN with train-before-add semantics.

#### Scenario: Initialize FAISS index
- **GIVEN** StandardGpuResources and configuration indicating Flat or IVF mode
- **WHEN** the retrieval service starts
- **THEN** the FAISS index is created on the target GPU, trained if IVF/PQ is selected, and wrapped in IndexIDMap2 to enforce explicit IDs

### Requirement: Batch Upsert and Delete Semantics
Upsert workflows SHALL batch vectors (e.g., 50k–100k) to control GPU memory pressure, normalize embeddings with `faiss.normalize_L2`, remove existing IDs before re-adding, and ensure deletions call both OpenSearch delete and `remove_ids` in FAISS.

#### Scenario: Re-ingest updated chunk batch
- **GIVEN** an incoming batch that updates existing chunk UUIDs
- **WHEN** the batch upsert runs
- **THEN** the pipeline removes prior FAISS entries for those IDs, re-adds normalized vectors, and upserts OpenSearch documents in the same transaction scope
- **AND** ingest metrics record successes/failures while transactional guards prevent FAISS/OpenSearch divergence by retrying failed legs and flagging the batch for manual inspection if reconciliation fails

### Requirement: Multi-Channel Query Execution
For each search request, the system SHALL: (1) issue BM25 queries to OpenSearch with filter predicates and PIT + search_after pagination, (2) issue SPLADE neural sparse queries using token→weight rank_features, and (3) compute a normalized dense embedding, perform FAISS `search(k')` with oversampling for filters, and join dense hits back to metadata via `vector_id`.

#### Scenario: Execute hybrid search request
- **GIVEN** a user query with namespace and tag filters
- **WHEN** the retrieval service processes the request
- **THEN** BM25, SPLADE, and dense searches run in parallel or pipelined fashion, each respecting the filters (dense via post-filter on joined metadata)
- **AND** the dense search oversamples results before post-filtering to retain at least `k` matches per namespace constraint

### Requirement: Query Service Contract
The retrieval service SHALL expose a synchronous API (`POST /v1/hybrid-search`) that accepts the query text, optional namespace/metadata filters, pagination cursor, flags for diversification, and desired result count. The response SHALL include fused results with per-channel diagnostics (scores, ranks) and cursors for subsequent pages.

#### Scenario: Invoke hybrid search endpoint
- **GIVEN** a client sends a JSON body containing `query`, `namespace`, `filters`, `page_size`, and `cursor`
- **WHEN** the service completes BM25/SPLADE/dense searches and fusion
- **THEN** it returns HTTP 200 with a payload that lists fused results (with doc/chunk IDs, highlight spans, scores by channel, and citations), alongside a `next_cursor` when more results exist and timing metadata for observability

### Requirement: Hybrid Fusion and Diversification
The system SHALL combine BM25, SPLADE, and dense candidate lists using Reciprocal Rank Fusion with configurable `k0` (default 60) and support optional score normalization. When diversification is enabled, it SHALL apply MMR with λ∈[0.5,0.8] over the top-M fused set before emitting the top-K results.

#### Scenario: Fuse retrieval channels with MMR
- **GIVEN** candidate lists from BM25, SPLADE, and dense searches
- **WHEN** the caller enables diversification
- **THEN** RRF produces fused scores
- **AND** MMR prunes the fused list so the final top-K balances relevance and cosine-based novelty within configured λ bounds

### Requirement: Result Shaping for RAG Responses
Result shaping SHALL collapse hits by `doc_id` (configurable chunks per doc), dedupe near-duplicates by dense cosine ≥0.98 or shared fingerprints, request OpenSearch highlights for BM25/SPLADE matches, provide fallback snippets for dense-only hits, and assemble context blocks with citation metadata and token budgets.

#### Scenario: Assemble RAG context package
- **GIVEN** fused hybrid hits that include multiple chunks of the same document
- **WHEN** result shaping runs
- **THEN** the response collapses duplicate doc_ids to the configured limit, removes high-similarity duplicates, and returns highlight-rich context blocks with citations and remaining token budget
- **AND** the service annotates each context block with channel contributions, snippet provenance offsets, and token counts so downstream RAG components can budget prompts

### Requirement: Namespace Isolation and Filtering
Namespaces SHALL be represented as keyword fields in OpenSearch documents and carried through to FAISS results via `vector_id` joins. The system SHALL support both single shared FAISS indexes with downstream filtering and per-namespace FAISS indexes when isolation is required.

#### Scenario: Apply namespace filter
- **GIVEN** indexed chunks across two namespaces
- **WHEN** a query specifies `namespace = e2e_test_a`
- **THEN** the final result set only includes chunks from that namespace, regardless of whether FAISS is shared or per-namespace

### Requirement: Operations, Stats, and Persistence
The system SHALL expose stats for FAISS (`ntotal`, training state, throughput) and OpenSearch (`_stats`, ingest/search latency), support pagination cursors (PIT + search_after, fused slicing), and provide backup/restore flows (OpenSearch snapshots, FAISS GPU→CPU serialization). Delete churn SHALL trigger guidance for periodic FAISS rebuilds.

#### Scenario: Snapshot and restore retrieval state
- **GIVEN** a scheduled maintenance window
- **WHEN** the operator triggers OpenSearch snapshot and FAISS serialization
- **THEN** both stores are captured to durable storage
- **AND** restoration into a clean environment reproduces identical `vector_id` alignments and comparable scores within float tolerances
- **AND** operational dashboards display current FAISS ntotal, delete ratio, last training timestamp, OpenSearch ingestion/search latency percentiles, and alert when thresholds defined in configuration are exceeded

### Requirement: Observability and Logging
The ingestion and retrieval components SHALL emit structured logs with trace IDs, expose Prometheus-compatible metrics (ingest throughput, FAISS latency, OpenSearch latency, fusion timings, error counts), and provide per-request tracing that correlates sparse and dense sub-requests.

#### Scenario: Trace hybrid search latency regression
- **GIVEN** observability tooling detects elevated latency
- **WHEN** an engineer inspects traces for `POST /v1/hybrid-search`
- **THEN** they can see spans for BM25, SPLADE, FAISS, fusion, and result shaping with individual durations and error tags, enabling pinpointing of the degraded subsystem

### Requirement: Validation and Calibration Suite
An automated validation harness SHALL cover ingest integrity (field presence, dimension checks), dense self-hit accuracy (≥0.95 @1 for IVF or 1.00 for Flat), sparse relevance sanity (≥90% self-match @10), namespace filtering, pagination stability, fusion efficacy, highlight packaging, and calibration sweeps for `nprobe`/PQ parameters.

#### Scenario: Run end-to-end validation on sample corpus
- **GIVEN** the provided JSONL dataset of sparse and dense features
- **WHEN** the validation command executes
- **THEN** all ingest, search, fusion, pagination, and backup checks pass with thresholds defined above, and calibration results are recorded for operational tuning
- **AND** the harness writes a human-readable report and machine JSON artifact indicating pass/fail per check, stored under `reports/validation/<timestamp>/`
