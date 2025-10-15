# Module: dense

GPU-only FAISS index management for dense retrieval.

## Functions

### `ntotal(self)`

*No documentation available.*

### `config(self)`

*No documentation available.*

### `set_id_resolver(self, resolver)`

*No documentation available.*

### `train(self, vectors)`

*No documentation available.*

### `needs_training(self)`

*No documentation available.*

### `add(self, vectors, vector_ids)`

*No documentation available.*

### `remove(self, vector_ids)`

*No documentation available.*

### `search(self, query, top_k)`

*No documentation available.*

### `serialize(self)`

*No documentation available.*

### `restore(self, payload)`

*No documentation available.*

### `stats(self)`

*No documentation available.*

### `_create_index(self)`

*No documentation available.*

### `_maybe_to_gpu(self, index)`

*No documentation available.*

### `_to_cpu(self, index)`

*No documentation available.*

### `_apply_search_parameters(self, index)`

*No documentation available.*

### `_log_index_configuration(self, index)`

*No documentation available.*

### `_resolve_vector_id(self, internal_id)`

*No documentation available.*

### `_init_gpu_resources(self)`

*No documentation available.*

### `_ensure_dim(self, vector)`

*No documentation available.*

### `_remove_ids(self, ids)`

*No documentation available.*

### `_rebuild_index(self)`

*No documentation available.*

## Classes

### `FaissSearchResult`

Dense search hit returned by FAISS.

### `FaissIndexManager`

Manage lifecycle of a GPU-resident FAISS index.
