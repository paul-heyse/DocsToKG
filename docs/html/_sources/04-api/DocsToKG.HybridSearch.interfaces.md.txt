# 1. Module: interfaces

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.interfaces``.

## 1. Overview

Protocols defining hybrid-search integration points.

Args:
    None

Returns:
    None

Raises:
    None

## 2. Functions

### `bulk_upsert(self, chunks)`

Insert or update chunk payloads.

Args:
chunks: Chunk payloads that should be persisted in the lexical index.

Returns:
None

### `bulk_delete(self, vector_ids)`

Remove chunk payloads for the supplied vector identifiers.

Args:
vector_ids: Identifiers referencing payloads that should be removed.

Returns:
None

Raises:
Exception: Implementations may surface provider-specific failures.

### `search_bm25(self, query_weights, filters, top_k, cursor)`

Execute a BM25-style sparse search.

Args:
query_weights: Sparse query representation (token → weight).
filters: Metadata filters used to restrict the search domain.
top_k: Maximum number of hits the caller expects in the page.
cursor: Optional continuation token from a previous search page.

Returns:
List of ``(chunk, score)`` pairs and an optional cursor for pagination.

### `search_splade(self, query_weights, filters, top_k, cursor)`

Execute a SPLADE sparse search.

Args:
query_weights: Sparse SPLADE activations for the query.
filters: Metadata filters used to restrict the search domain.
top_k: Maximum number of hits the caller expects in the page.
cursor: Optional continuation token from a previous search page.

Returns:
List of ``(chunk, score)`` pairs and an optional cursor for pagination.

### `search_bm25_true(self, query_weights, filters, top_k, cursor)`

Optional Okapi BM25 search supporting DF/length-aware scoring.

### `highlight(self, chunk, query_tokens)`

Return highlight snippets for ``chunk`` given ``query_tokens``.

Args:
chunk: Chunk payload whose text should be highlighted.
query_tokens: Tokens extracted from the query text.

Returns:
Highlights appropriate for presentation alongside the chunk.

### `stats(self)`

Return implementation-defined statistics about the lexical index.

Args:
None

Returns:
Mapping containing statistics relevant to the implementation.

### `dim(self)`

Return the embedding dimensionality.

### `ntotal(self)`

Return the number of stored vectors.

### `device(self)`

Return the CUDA device identifier used by the index.

### `config(self)`

Immutable configuration backing the dense store.

### `gpu_resources(self)`

Return GPU resources when available, otherwise ``None``.

### `add(self, vectors, vector_ids)`

Insert dense vectors.

### `add_batch(self, vectors, vector_ids)`

Insert vectors in batches; defaults mirror FAISS GPU-friendly chunking.

### `remove(self, vector_ids)`

Delete dense vectors referenced by ``vector_ids``.

### `search(self, query, top_k)`

Search for nearest neighbours of ``query``.

### `search_many(self, queries, top_k)`

Search for nearest neighbours of multiple queries.

### `search_batch(self, queries, top_k)`

Optional alias for batched search.

### `serialize(self)`

Return a serialised representation of the index.

### `restore(self, payload)`

Restore index state from ``payload``.

### `stats(self)`

Return implementation-defined statistics.

### `rebuild_if_needed(self)`

Perform compaction when the store indicates a rebuild is required.

### `needs_training(self)`

Return ``True`` when additional training is required.

### `train(self, vectors)`

Train the index with representative vectors.

### `set_id_resolver(self, resolver)`

Register a resolver translating FAISS integer IDs to external IDs.

## 3. Classes

### `LexicalIndex`

Protocol describing the lexical (BM25/SPLADE) index interface.

Implementations are expected to manage chunk persistence, sparse search, and
highlighting in a way that is compatible with the hybrid search pipeline.

Attributes:
None

Examples:
>>> from DocsToKG.HybridSearch.storage import OpenSearchSimulator
>>> simulator: LexicalIndex = OpenSearchSimulator()
>>> simulator.bulk_upsert([])  # doctest: +SKIP

### `DenseVectorStore`

Protocol describing the dense vector index surface area.

Implementations provide GPU-backed vector search with facilities for
ingestion, persistence, and statistics reporting. This protocol allows
tests to swap lightweight stand-ins without relying on FAISS directly.
