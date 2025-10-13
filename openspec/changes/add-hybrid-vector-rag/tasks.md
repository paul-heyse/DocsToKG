## 1. Ingestion & Storage
- [ ] 1.1 Implement chunking pipeline that produces deterministic UUIDs, overlapping windows, SPLADE weights, and Qwen embeddings with shared tokenization, retry semantics, and dead-letter handling
- [ ] 1.2 Define and deploy OpenSearch index templates/mappings, including analyzers, rank_features fields, metadata keywords/dates, and namespace routing strategy
- [ ] 1.3 Build FAISS GPU index provisioning for Flat, IVF-Flat, and IVFPQ, covering training sample selection, StandardGpuResources pooling, IndexIDMap2 wrapping, serialization, and ntotal vs memory monitoring
- [ ] 1.4 Implement batch upsert/delete workflows that maintain OpenSearch ↔ FAISS consistency, enforce L2 normalization, perform reconciliation retries, and emit ingest metrics

## 2. Retrieval & Fusion
- [ ] 2.1 Implement synchronous `POST /v1/hybrid-search` API contract with request validation, pagination cursors, diversification flags, and per-channel diagnostics in the response
- [ ] 2.2 Implement BM25, SPLADE, and FAISS search executors with configuration-driven parameters, oversampling, namespace filters, and failure isolation to prevent one modality from failing the whole query
- [ ] 2.3 Implement Reciprocal Rank Fusion (configurable `k0`), score normalization options, MMR diversification, and deduplication by doc_id and cosine thresholds
- [ ] 2.4 Implement result shaping that joins metadata, fetches highlights, annotates provenance offsets, enforces token budgets, and emits citation-rich context blocks

## 3. Configuration, Observability, and Operations
- [ ] 3.1 Build configuration management layer (file or service) controlling index type, IVF/PQ parameters, chunk sizes, oversampling ratios, fusion/MMR thresholds, dedupe cosine, and namespace routing with hot-reload support
- [ ] 3.2 Expose structured logging, Prometheus metrics, and tracing spans for ingest and retrieval subsystems; integrate alerts for latency, recall, error rate, and delete-churn thresholds
- [ ] 3.3 Implement ops tooling for stats surfaces, pagination verification, OpenSearch snapshots, FAISS serialization/restore, and delete-churn rebuild triggers
- [ ] 3.4 Author operational runbooks covering calibration sweeps, namespace onboarding, failover/rollback procedures, and backup/restore drills

## 4. Validation & Calibration
- [ ] 4.1 Implement automated validation harness that ingests provided JSONL fixtures, runs integrity checks, dense self-hit tests, sparse relevance queries, namespace filter checks, pagination stability, fusion efficacy, highlight verification, backup/restore parity, and writes human/JSON reports under `reports/validation/<timestamp>/`
- [ ] 4.2 Integrate calibration sweeps for `nprobe`, PQ settings, oversampling ratios into CI or scheduled jobs, and persist results for comparison over time
