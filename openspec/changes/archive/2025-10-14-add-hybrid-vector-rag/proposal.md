## Why

DocsToKG currently stores vectors in document-centric blobs without a production-grade retrieval stack. We need to operationalize the new GPU-enabled FAISS build alongside OpenSearch so that hybrid sparse+dense retrieval, namespace isolation, and RAG-ready context assembly can be delivered with predictable performance.

## What Changes

- Build a document-to-chunk ingestion service that applies configurable window sizing, generates deterministic UUIDs, produces SPLADE token weights and Qwen3-4B embeddings, and enforces retry/dead-letter semantics
- Stand up OpenSearch index templates/mappings for chunk text, rank_features, and metadata filters, including namespace routing and analyzer configuration
- Provision GPU FAISS indexes (Flat, IVF-Flat, IVFPQ) with training sample selection, IndexIDMap2 ID control, serialization, and ntotal vs GPU memory monitoring
- Implement batch upsert/delete reconciliation that guarantees OpenSearch and FAISS consistency, enforces L2 normalization, exposes ingest metrics, and guards against partial failures
- Ship a synchronous `POST /v1/hybrid-search` API that orchestrates BM25, SPLADE, and dense searches with configurable parameters, oversampling, and namespace filtering
- Implement fusion middleware that supports Reciprocal Rank Fusion, optional score normalization, MMR diversification, dedupe-by-cosine thresholds, and provenance-aware result shaping
- Add configuration management, hot-reload, observability (structured logs, metrics, tracing), and ops tooling for stats dashboards, pagination verification, and backup/restore workflows
- Deliver an automated validation and calibration harness (JSONL ingest through search/fusion), producing human-readable and machine JSON reports for ingest integrity, relevance checks, namespace isolation, pagination, fusion efficacy, and backup parity

## Impact

- Affected specs: `hybrid-search` (new capability)
- Affected code: ingestion jobs, retrieval services (`retrieval/opensearch`, `retrieval/faiss`), fusion middleware, result shaping layer, ops tooling for backups and validation
