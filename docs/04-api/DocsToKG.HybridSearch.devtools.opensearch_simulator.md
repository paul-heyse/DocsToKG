# 1. Module: opensearch_simulator

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.devtools.opensearch_simulator``.

## 1. Overview

In-memory OpenSearch simulator and schema helpers for development.

## 2. Functions

### `matches_filters(chunk, filters)`

Return ``True`` when ``chunk`` satisfies the provided ``filters``.

### `asdict(self)`

Return a dictionary representation of the template.

### `bootstrap_template(self, namespace, chunking)`

*No documentation available.*

### `get_template(self, namespace)`

*No documentation available.*

### `list_templates(self)`

*No documentation available.*

### `bulk_upsert(self, chunks)`

*No documentation available.*

### `bulk_delete(self, vector_ids)`

*No documentation available.*

### `fetch(self, vector_ids)`

*No documentation available.*

### `vector_ids(self)`

*No documentation available.*

### `register_template(self, template)`

*No documentation available.*

### `template_for(self, namespace)`

*No documentation available.*

### `search_bm25(self, query_weights, filters, top_k, cursor)`

*No documentation available.*

### `search_splade(self, query_weights, filters, top_k, cursor)`

*No documentation available.*

### `highlight(self, chunk, query_tokens)`

*No documentation available.*

### `stats(self)`

*No documentation available.*

### `_filtered_chunks(self, filters)`

*No documentation available.*

### `_bm25_score(self, stored, query_weights)`

*No documentation available.*

### `_paginate(self, results, top_k, cursor)`

*No documentation available.*

### `_search_sparse(self, scoring_fn, filters, top_k, cursor)`

*No documentation available.*

### `_recompute_avg_length(self)`

*No documentation available.*

## 3. Classes

### `OpenSearchIndexTemplate`

Representation of a namespace-specific OpenSearch template.

### `OpenSearchSchemaManager`

Manage simulated OpenSearch index templates for tests.

### `_StoredChunk`

*No documentation available.*

### `OpenSearchSimulator`

Simplified OpenSearch-like index used for development and tests.
