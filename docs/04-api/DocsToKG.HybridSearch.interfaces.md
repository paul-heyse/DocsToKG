# 1. Module: interfaces

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.interfaces``.

## 1. Overview

Protocols defining hybrid-search integration points.

## 2. Functions

### `bulk_upsert(self, chunks)`

Insert or update chunk payloads.

### `bulk_delete(self, vector_ids)`

Remove chunk payloads for the supplied vector identifiers.

### `search_bm25(self, query_weights, filters, top_k, cursor)`

Execute a BM25-style sparse search.

### `search_splade(self, query_weights, filters, top_k, cursor)`

Execute a SPLADE sparse search.

### `highlight(self, chunk, query_tokens)`

Return highlight snippets for ``chunk`` given ``query_tokens``.

### `stats(self)`

Return implementation-defined statistics about the lexical index.

## 3. Classes

### `LexicalIndex`

Protocol describing the lexical (BM25/SPLADE) index interface.

Implementations are expected to manage chunk persistence, sparse search, and
highlighting in a way that is compatible with the hybrid search pipeline.
