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

Serialize every managed store (namespace -> payload).

### `restore_all(self, payloads)`

Restore stores from serialized payloads.

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

## 3. Classes

### `FaissRouter`

Lightweight namespace-aware router for managed FAISS instances.
