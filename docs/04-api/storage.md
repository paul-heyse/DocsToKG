# Module: storage

In-memory storage simulators for OpenSearch and chunk registry.

## Functions

### `upsert(self, chunks)`

*No documentation available.*

### `delete(self, vector_ids)`

*No documentation available.*

### `get(self, vector_id)`

*No documentation available.*

### `bulk_get(self, vector_ids)`

*No documentation available.*

### `all(self)`

*No documentation available.*

### `count(self)`

*No documentation available.*

### `bulk_upsert(self, chunks)`

*No documentation available.*

### `bulk_delete(self, vector_ids)`

*No documentation available.*

### `fetch(self, vector_ids)`

*No documentation available.*

### `vector_ids(self)`

*No documentation available.*

### `search_bm25(self, query_weights, filters, top_k, cursor)`

*No documentation available.*

### `search_splade(self, query_weights, filters, top_k, cursor)`

*No documentation available.*

### `highlight(self, chunk, query_tokens)`

*No documentation available.*

### `_filtered_chunks(self, filters)`

*No documentation available.*

### `_matches_filters(self, chunk, filters)`

*No documentation available.*

### `_bm25_score(self, stored, query_weights)`

*No documentation available.*

### `_paginate(self, scores, top_k, cursor)`

*No documentation available.*

### `_recompute_avg_length(self)`

*No documentation available.*

### `stats(self)`

*No documentation available.*

## Classes

### `StoredChunk`

Internal representation of a chunk stored in the OpenSearch simulator.

### `ChunkRegistry`

Durable mapping of `vector_id` → `ChunkPayload` for joins and reconciliation.

### `OpenSearchSimulator`

Subset of OpenSearch capabilities required for hybrid retrieval tests.
