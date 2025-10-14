## 1. Ingestion & Storage
- [x] 1.1 Consume pre-chunked JSONL + vector artifacts (chunking happens upstream) and emit `ChunkPayload` objects with consistent UUID alignment and metadata.
- [x] 1.2 Define OpenSearch chunk index templates/mappings with namespace routing, rank_features for SPLADE, and metadata fields, recording chunking parameters.
- [x] 1.3 Build FAISS GPU index provisioning supporting Flat/IVF/PQ, training sample selection, IndexIDMap2 wrapping, serialization, and ntotal monitoring.
- [x] 1.4 Implement batch upsert/delete workflows that keep OpenSearch ↔ FAISS/registry state in sync, enforce L2 normalization, and emit ingest metrics.

## 2. Retrieval & Fusion
- [x] 2.1 Expose a synchronous handler for `POST /v1/hybrid-search` with request validation, pagination cursors, diversification flag, and per-channel diagnostics.
- [x] 2.2 Implement BM25, SPLADE, and FAISS retrieval executors with configuration-driven parameters, oversampling, namespace filters, and failure isolation.
- [x] 2.3 Implement Reciprocal Rank Fusion (configurable `k0`), optional MMR diversification, and dedupe by doc_id and cosine thresholds.
- [x] 2.4 Implement result shaping that joins metadata, produces highlights/fallback snippets, enforces chunk-per-doc budgets, and returns citation-rich payloads.

## 3. Configuration, Observability, and Operations
- [x] 3.1 Provide a file-backed configuration layer controlling index type, IVF/PQ parameters, oversampling, fusion/MMR thresholds, dedupe cosine, and namespace routing with reload support.
- [x] 3.2 Expose structured logging, Prometheus-style metrics collectors, and tracing spans for ingest and retrieval subsystems.
- [x] 3.3 Implement ops tooling for stats snapshots, pagination verification, FAISS serialization/restore, and delete-churn rebuild guidance.
- [x] 3.4 Author operational runbooks covering calibration sweeps, namespace onboarding, failover/rollback, and backup/restore drills.

## 4. Validation & Calibration
- [x] 4.1 Deliver an automated validation harness that ingests JSONL fixtures, runs ingest integrity/self-hit/sparse relevance/namespace/pagination/highlight/backups checks, and writes JSON+text reports.
- [x] 4.2 Integrate calibration sweeps for oversampling/recall into the harness and persist results for regression comparison.
