# 1. Module: storage

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.storage``.

## 1. Overview

Chunk registry helpers and shared filter utilities.

## 2. Functions

### `matches_filters(chunk, filters)`

Check whether ``chunk`` satisfies the provided OpenSearch-style filters.

### `upsert(self, chunks)`

Insert or update registry entries for ``chunks``.

### `delete(self, vector_ids)`

Remove registry entries for the supplied vector identifiers.

### `get(self, vector_id)`

Return the chunk payload for ``vector_id`` when available.

### `bulk_get(self, vector_ids)`

Return chunk payloads for identifiers present in the registry.

### `resolve_faiss_id(self, internal_id)`

Translate a FAISS integer id back to the original vector identifier.

### `all(self)`

Return all cached chunk payloads.

### `count(self)`

Return the number of chunks tracked by the registry.

### `vector_ids(self)`

Return all vector identifiers in insertion order.

## 3. Classes

### `ChunkRegistry`

Durable mapping of vector identifiers to chunk payloads.
