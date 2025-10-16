# 1. Module: storage

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.storage``.

## 1. Overview

Chunk registry helpers and shared filter utilities.

Args:
    None

Returns:
    None

Raises:
    None

## 2. Functions

### `matches_filters(chunk, filters)`

Check whether ``chunk`` satisfies the provided OpenSearch-style filters.

Args:
chunk: Chunk payload whose metadata should be evaluated.
filters: Mapping of filter keys to expected values.

Returns:
``True`` if the chunk matches every filter, otherwise ``False``.

### `upsert(self, chunks)`

Insert or update registry entries for ``chunks``.

Args:
chunks: Chunk payloads that should be tracked by the registry.

Returns:
None

### `delete(self, vector_ids)`

Remove registry entries for the supplied vector identifiers.

Args:
vector_ids: Identifiers associated with chunks to delete.

Returns:
None

Raises:
None

### `get(self, vector_id)`

Return the chunk payload for ``vector_id`` when available.

Args:
vector_id: Identifier of the chunk to retrieve.

Returns:
Matching chunk payload, or ``None`` if the identifier is unknown.

### `bulk_get(self, vector_ids)`

Return chunk payloads for identifiers present in the registry.

Args:
vector_ids: Identifiers of the desired chunk payloads.

Returns:
List of chunk payloads that are present in the registry.

### `resolve_faiss_id(self, internal_id)`

Translate a FAISS integer id back to the original vector identifier.

Args:
internal_id: Integer id assigned by the FAISS vector store.

Returns:
Associated vector identifier when the mapping exists.

### `all(self)`

Return all cached chunk payloads.

Args:
None

Returns:
List containing every chunk payload stored in the registry.

### `iter_all(self)`

Yield chunk payloads without materialising the full list.

### `count(self)`

Return the number of chunks tracked by the registry.

Args:
None

Returns:
Number of chunk payloads stored in the registry.

### `vector_ids(self)`

Return all vector identifiers in insertion order.

Args:
None

Returns:
Vector identifiers ordered by insertion time.

## 3. Classes

### `ChunkRegistry`

Durable mapping of vector identifiers to chunk payloads.

Attributes:
_chunks: Mapping from vector identifier to chunk payload.
_bridge: Mapping from FAISS integer identifier to vector identifier.

Examples:
>>> registry = ChunkRegistry()
>>> registry.upsert([])  # no-op
>>> registry.count()
0
