## 1. Artifact Ingestion & Storage
- [x] 1.1 Load pre-chunked JSONL documents and matching vector artifacts, preserving UUID alignment, metadata, and sparse/dense feature weights when emitting `ChunkPayload` objects.
- [x] 1.2 Maintain FAISS/OpenSearch dual writes with pre-computed embeddings, including training sample selection, registry reconciliation, and ingest metrics.

## 2. Retrieval & Fusion
- [x] 2.1 Adapt BM25 and SPLADE executors to consume pre-computed weights with query-time feature generation, maintaining pagination and filter semantics.
- [x] 2.2 Keep dense retrieval, RRF/MMR fusion, and deduplication behaviour intact while honoring configured embedding dimensions.

## 3. Validation & Tooling
- [x] 3.1 Update validation harness and CLI to infer embedding dimensionality from vector artifacts and operate on artifact-backed ingestion.
- [x] 3.2 Provide regression tests and fixtures that exercise ingestion, re-ingestion updates, hybrid retrieval, and report generation using the new artifact-driven workflow.
