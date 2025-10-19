# 1. Module: router

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.router``.

## 1. Overview

Namespace router utilities for hybrid search vector stores.

## 2. Functions

### `per_namespace(self)`

Return ``True`` when a dedicated store is provisioned per namespace.

### `default_store(self)`

Return the default store used when namespaces are disabled.

### `get(self, namespace)`

Return the store serving ``namespace`` (creating one if necessary).

### `stats(self)`

Return stats for all managed stores (namespaced and aggregate).

### `_aggregate_stats(namespaced)`

*No documentation available.*

### `serialize_all(self)`

Serialize every managed store including snapshot metadata.

### `_serialize_with_meta(store)`

*No documentation available.*

### `iter_stores(self)`

Return a snapshot of managed stores keyed by namespace.

### `restore_all(self, payloads)`

Restore stores from serialized payloads and metadata.

### `_extract_payload_and_meta(packed)`

*No documentation available.*

### `rebuild_if_needed(self)`

Attempt to rebuild all managed stores; returns True if any rebuild occurs.

### `set_resolver(self, resolver)`

Register a resolver applied to existing and future stores.

### `set_id_resolver(self, resolver)`

Alias for :meth:`set_resolver` to improve readability.

### `evict_idle(self)`

Serialize and evict stores idle longer than ``max_idle_seconds``.

### `run_maintenance(self)`

Run optional training and rebuild checks across managed stores.

Args:
training_sampler: Callable receiving `(namespace, store)` and returning
representative vectors for training when the store reports that it
requires training. If omitted, training is skipped.

Returns:
Mapping of namespace to maintenance actions performed.

### `_maintain_store(self, namespace, store, training_sampler)`

*No documentation available.*

### `build_entry(payload, meta)`

Package FAISS payload bytes and optional metadata for persistence.

Args:
payload: Serialized FAISS index bytes.
meta: Supplemental metadata returned by the store snapshot.

Returns:
Dict[str, object]: Mapping with ``faiss`` bytes and optional ``meta`` payload.

### `collect(store)`

Extract serialized payload and snapshot metadata from ``store``.

Args:
store: Vector store to snapshot.

Returns:
Tuple[bytes, Optional[Mapping[str, object]]]: Serialized FAISS bytes and metadata.

### `coerce_entry(entry)`

Normalise stored payloads into raw bytes and metadata mapping.

Args:
entry: Persisted snapshot entry in legacy or current format.

Returns:
Tuple[Optional[bytes], Optional[Mapping[str, object]]]: Normalised payload and metadata.

### `restore_store(store, blob, meta)`

Restore a store from serialized payload and optional metadata.

Args:
store: Vector store instance to restore.
blob: Serialized FAISS bytes.
meta: Supplemental metadata to pass to ``restore`` when supported.

## 3. Classes

### `FaissRouter`

Lightweight namespace-aware router for managed FAISS instances.
