# Module: dense

FAISS index management with GPU-aware fallbacks.

## Functions

### `ntotal(self)`

*No documentation available.*

### `config(self)`

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

### `_init_gpu_resources(self)`

*No documentation available.*

### `_ensure_dim(self, vector)`

*No documentation available.*

### `_remove_ids(self, ids)`

*No documentation available.*

### `_normalize(self, matrix)`

*No documentation available.*

## Classes

### `FaissSearchResult`

Dense search hit returned by FAISS.

### `FaissIndexManager`

Manage lifecycle of a FAISS index with optional GPU acceleration.
