# Module: retrieval

Hybrid search execution across sparse and dense channels.

## Functions

### `search(self, request)`

*No documentation available.*

### `_execute_bm25(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_execute_splade(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_execute_dense(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_filter_dense_hits(self, hits, filters)`

Return the dense hits that satisfy the active filters along with a pre-fetched
chunk map for downstream processing.

### `_matches_filters(self, chunk, filters)`

*No documentation available.*

### `_dedupe_candidates(self, candidates, fused_scores)`

*No documentation available.*

### `_validate_request(self, request)`

*No documentation available.*

## Classes

### `RequestValidationError`

Raised when the caller submits an invalid search request.

### `ChannelResults`

*No documentation available.*

### `HybridSearchService`

Execute BM25, SPLADE, and dense retrieval with fusion.
