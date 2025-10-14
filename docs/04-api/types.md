# Module: types

Core typed structures for hybrid search components.

## Functions

### `copy(self)`

*No documentation available.*

### `passed(self)`

*No documentation available.*

## Classes

### `DocumentInput`

Pre-computed chunk and vector artifacts for a document.

### `ChunkFeatures`

Sparse and dense features computed for a chunk.

### `ChunkPayload`

Fully materialized chunk stored in OpenSearch and FAISS.

### `HybridSearchDiagnostics`

Per-channel diagnostics for a result.

### `HybridSearchResult`

Output item returned to callers of the hybrid search service.

### `HybridSearchRequest`

Validated request payload for `/v1/hybrid-search`.

### `HybridSearchResponse`

Response envelope returned by the hybrid search API.

### `FusionCandidate`

Intermediate structure used by fusion pipeline.

### `ValidationReport`

Structured output for the validation harness.

### `ValidationSummary`

Aggregate of validation reports.
