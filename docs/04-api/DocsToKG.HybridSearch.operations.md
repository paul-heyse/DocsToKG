# Module: operations

Operational tooling for FAISS/OpenSearch maintenance.

## Functions

### `build_stats_snapshot(faiss_index, opensearch, registry)`

Capture a lightweight snapshot of hybrid search storage metrics.

Args:
faiss_index: Dense vector index manager.
opensearch: OpenSearch simulator representing lexical storage.
registry: Chunk registry tracking vector-to-payload mappings.

Returns:
Mapping describing FAISS stats, OpenSearch stats, and chunk counts.

### `verify_pagination(service, request)`

Ensure pagination cursors produce non-duplicated results.

Args:
service: Hybrid search service to execute paginated queries.
request: Initial hybrid search request payload.

Returns:
PaginationCheckResult detailing encountered cursors and duplicates.

### `serialize_state(faiss_index, registry)`

Serialize FAISS state and registry metadata for snapshotting.

Args:
faiss_index: Dense index manager whose state should be captured.
registry: Chunk registry providing vector identifiers.

Returns:
Mapping containing base64-encoded FAISS bytes and registered vector IDs.

### `restore_state(faiss_index, payload)`

Restore FAISS index state from a serialized payload.

Args:
faiss_index: Dense index manager to restore into.
payload: Snapshot mapping produced by `serialize_state`.

Raises:
ValueError: If the payload does not include a FAISS snapshot.

Returns:
None

### `should_rebuild_index(registry, deleted_since_snapshot, threshold)`

Heuristic to determine when FAISS should be rebuilt after deletions.

Args:
registry: Chunk registry reflecting current vector count.
deleted_since_snapshot: Number of vectors deleted since the last snapshot.
threshold: Fraction of deletions that triggers a rebuild.

Returns:
True when the proportion of deletions exceeds `threshold`.

## Classes

### `PaginationCheckResult`

Result of a pagination verification run.

Attributes:
cursor_chain: Sequence of pagination cursors encountered.
duplicate_detected: True when duplicate results were observed.

Examples:
>>> result = PaginationCheckResult(cursor_chain=["cursor1"], duplicate_detected=False)
>>> result.duplicate_detected
False
