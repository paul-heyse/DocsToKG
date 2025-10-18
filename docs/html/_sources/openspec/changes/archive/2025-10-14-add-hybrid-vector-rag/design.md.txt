## Context
DocsToKG is transitioning from ad-hoc document-level vector storage to a production retrieval stack that blends sparse (BM25 + SPLADEv3) and dense (Qwen3-4B, 2560-d) signals. The repository now includes a GPU-enabled FAISS wheel that can run Flat, IVF-Flat, and IVFPQ indexes. We must define schemas, ingestion, search orchestration, and operational guarantees so that future RAG tooling can depend on stable hybrid retrieval behavior.

## Goals / Non-Goals
- Goals:
  - Persist chunk-level content and metadata in OpenSearch with both BM25 text and SPLADE rank features
  - Maintain synchronized dense embeddings in GPU FAISS indexes with explicit vector IDs
  - Serve hybrid retrieval (BM25, SPLADE, dense) with Reciprocal Rank Fusion and optional MMR diversification
  - Provide result shaping that returns deduped, highlighted, and namespace-scoped RAG context blocks
  - Supply lifecycle tooling (upsert, delete, stats, calibration, backup/restore, validation harness)
- Non-Goals:
  - Replacing OpenSearch with alternative sparse engines
  - Implementing cross-encoder re-ranking (left for future iteration)
  - Delivering hard predicate filtering inside FAISS IVF (post-filtering suffices for this change)

## Decisions
- Decision: **Chunk the corpus into 400–800 token windows with 100–200 token overlaps**, storing doc_id, chunk_id, namespace, and metadata fields in OpenSearch to support filters, ACLs, dedupe, and highlighting.
  Alternatives: sentence-level chunks (too fine-grained, higher overhead) or document-level blobs (insufficient recall control). We keep chunk granularity adjustable but start with the recommended window sizes.
- Decision: **Use OpenSearch rank_features to persist SPLADEv3 token→weight maps** alongside BM25 text, ensuring we can issue neural sparse queries without external services.
  Alternatives: storing sparse vectors in a side-car service would complicate query orchestration.
- Decision: **Normalize dense vectors (float32) and store them in FAISS GPU indexes wrapped by IndexIDMap2 using vector_id == chunk UUID.**
  Alternatives: rely on FAISS auto IDs (breaks joins) or maintain CPU indexes (misses GPU acceleration). UUID bridge keeps joins deterministic.
- Decision: **Support both GpuIndexFlatIP and GpuIndexIVFFlat/GpuIndexIVFPQ**, enabling exact search for small corpora and ANN for large ones.
  Alternatives: only Flat (memory-heavy) or only IVF (higher setup cost). We keep both and gate via configuration.
- Decision: **Apply Reciprocal Rank Fusion (k0≈60) across BM25, SPLADE, and dense candidate lists, then optionally run MMR diversification with λ∈[0.5,0.8] on the fused top-M.**
  Alternatives: score normalization + weighted sum (harder to calibrate) or heuristic linear combos (less robust).
- Decision: **Implement namespace isolation at the data layer (namespace keyword field) while allowing single- or multi-index FAISS deployments.** Default is a shared FAISS index with downstream filters; we can shard by namespace if isolation requires it.
- Decision: **Provide full validation harness using the supplied JSONL vectors**, covering ingest integrity, BM25/SPLADE relevance, dense self-hits, namespace filtering, pagination, fusion quality, and backup/restore parity.
  Alternatives: manual spot checks would not meet regression requirements.
- Decision: **Expose configuration through a hot-reloadable config layer (YAML + watch or service-backed)** controlling chunking, FAISS index selection/parameters, search oversampling, fusion/MMR thresholds, dedupe cosine level, and namespace routing.
  Alternatives: hard-coded settings would require redeploys for tuning and impede calibration.
- Decision: **Expose retrieval via a synchronous REST endpoint (`POST /v1/hybrid-search`)** returning fused results, diagnostics, and pagination cursors.
  Alternatives: asynchronous jobs introduce latency and complicate RAG integration; gRPC can be added later if needed.
- Decision: **Instrument ingestion and retrieval with structured logs, Prometheus metrics, and distributed tracing**, feeding dashboards/alerts for throughput, latency, error, and delete-churn thresholds.
  Alternatives: ad-hoc logging would not satisfy operational readiness.

## Architecture Overview
- **Ingestion pipeline**: consumes documents from existing loaders (JSONL bootstrap and future sources), applies tokenizer-aligned chunker, generates SPLADEv3 weights via shared tokenizer, obtains Qwen3-4B embeddings through the model helper (GPU batch inference), and emits chunk payloads onto a work queue for OpenSearch/FAISS writers. Retryable errors (e.g., transient embedding failures) re-queue the document; terminal errors route to a dead-letter topic for manual review.
- **Storage layer**: OpenSearch index templates manage namespace-specific indices with mappings for metadata, text, rank_features, and vector_id. FAISS index manager maintains GPU indexes per namespace group, training IVF/PQ instances with configurable sampling and writing serialized checkpoints for disaster recovery.
- **Retrieval service**: REST API handles authentication, validates request payloads, triggers three modality executors (BM25, SPLADE, FAISS) in parallel using shared configuration, and merges responses through fusion and diversification layers. Post-filtering ensures namespace and metadata filters apply uniformly.
- **Result shaping**: merges metadata from OpenSearch hits keyed by vector_id, computes dedupe heuristics, requests highlight snippets, and assembles context blocks with provenance offsets, token counts, and channel attribution.
- **Configuration & observability**: Config loader watches a versioned config file/service; changes update in-memory parameters atomically. Metrics/traces/logs map to dashboards for ingest throughput, FAISS memory usage, OpenSearch latency, fusion timings, and validation harness outcomes.
- **Validation & calibration**: Dedicated command-line tool orchestrates ingest of fixture data, runs retrieval scenarios, captures metrics, and writes reports (markdown/JSON) under `reports/validation/<timestamp>/`. Automation integrates with CI or cron for regression detection.

## Risks / Trade-offs
- Risk: GPU memory exhaustion when using Flat indexes at multi-million scale. Mitigation: surface guidance on switching to IVF-Flat or IVFPQ with configurables (nlist, nprobe, M, nbits) and monitoring ntotal vs GPU capacity.
- Risk: Delete churn in IVF indexes leading to tombstoned entries. Mitigation: document periodic rebuild procedures and track delete ratios via stats tooling.
- Risk: SPLADE query drift if tokenization diverges between ingest and query time. Mitigation: enforce shared tokenizer config and include validation checks that compare doc/query token alignment.
- Risk: Hybrid fusion may return near-duplicate chunks. Mitigation: enforce dedupe by document ID and near-duplicate cosine thresholds (≥0.98) before packaging context.
- Risk: Configuration drift between ingestion and retrieval services. Mitigation: centralize configuration with versioning and include config hash in metrics and responses for auditing.
- Risk: Validation harness false negatives if fixture coverage is insufficient. Mitigation: allow operators to plug in additional corpora and capture metrics across different datasets.

## Migration Plan
1. Bootstrap OpenSearch mappings and ensure rehearsal data (JSONL) indexes successfully.
2. Implement chunking + feature generation service, ensuring SPLADE and embedding helpers share tokenization and retry semantics.
3. Train FAISS IVF variants with representative samples; snapshot trained states for reuse and register them in configuration.
4. Build batch upsert service that reads legacy blobs, emits chunk records, and populates both OpenSearch and FAISS in batches (50k–100k vectors).
5. Stand up configuration layer, observability pipelines, and dashboards; verify hot-reload flows.
6. Validate ingest with the test harness, confirming ntotal alignment, sparse/dense self-hits, and namespace behavior.
7. Deploy hybrid query service, result shaping, and fusion layers; run performance tuning for nprobe/PQ and oversampling ratios.
8. Enable backups (OpenSearch snapshots, FAISS writes), document rebuild/rollback procedures, and schedule periodic calibration/validation runs.

## Open Questions
- Do we need per-namespace FAISS indexes immediately, or can we launch with a shared index plus downstream filtering?
- What are the production latency and recall targets that should trigger IVF/PQ tuning runs versus Flat operation?
- Where should the hot-reloadable configuration live (file-based with GitOps vs service-backed store) to balance simplicity and operational control?
- What authentication/authorization mechanisms are required on `POST /v1/hybrid-search`, especially when exposing namespaces with ACLs?
