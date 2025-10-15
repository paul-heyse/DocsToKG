# Module: dense

GPU-only FAISS index management for dense retrieval.

## Functions

### `ntotal(self)`

Number of vectors currently stored in the FAISS index.

Args:
None

Returns:
Count of vectors indexed in FAISS.

### `config(self)`

Return the dense index configuration in use.

Args:
None

Returns:
DenseIndexConfig associated with this manager.

### `set_id_resolver(self, resolver)`

Register a callback translating FAISS internal IDs to vector UUIDs.

Args:
resolver: Callable mapping FAISS internal IDs to vector UUIDs.

Returns:
None

### `train(self, vectors)`

Train IVF-style indexes with the provided vectors when required.

Args:
vectors: Sequence of vectors used for training.

Returns:
None

Raises:
ValueError: If training vectors are required but not provided.

### `needs_training(self)`

Return True when the underlying index still requires training.

Args:
None

Returns:
Boolean indicating if training is required.

### `add(self, vectors, vector_ids)`

Add vectors to the index, replacing existing entries when necessary.

Args:
vectors: Sequence of embedding vectors to add.
vector_ids: Corresponding vector identifiers.

Returns:
None

Raises:
ValueError: If the lengths of `vectors` and `vector_ids` differ.

### `remove(self, vector_ids)`

Remove vectors from FAISS and the in-memory cache by vector UUID.

Args:
vector_ids: Identifiers of vectors to remove.

Returns:
None

### `search(self, query, top_k)`

Execute a cosine-similarity search returning the best `top_k` results.

Args:
query: Query vector to search against the index.
top_k: Maximum number of nearest neighbours to return.

Returns:
List of `FaissSearchResult` objects ordered by score.

### `serialize(self)`

Serialize the FAISS index and cached vectors for persistence.

Args:
None

Returns:
Bytes object containing serialized index and vector cache.

### `restore(self, payload)`

Restore FAISS state from bytes produced by `serialize`.

Args:
payload: Bytes previously produced by `serialize`.

Returns:
None

Raises:
ValueError: If the payload is invalid or incompatible.

### `stats(self)`

Expose diagnostic metrics for monitoring.

Args:
None

Returns:
Dictionary containing index configuration and diagnostics.

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

### `_detect_device(self, index)`

*No documentation available.*

### `_resolve_device(self, config)`

Determine the target GPU device, honouring runtime overrides.

Args:
config: Dense index configuration supplying the default device.

Returns:
int: GPU device identifier to use for FAISS operations.

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

Attributes:
vector_id: Identifier of the matched vector.
score: Cosine similarity score returned by FAISS.

Examples:
>>> result = FaissSearchResult(vector_id="vec-1", score=0.92)
>>> result.vector_id
'vec-1'

### `FaissIndexManager`

Manage lifecycle of a GPU-resident FAISS index.

Attributes:
_dim: Dimensionality of vectors stored in the index.
_config: DenseIndexConfig controlling index behaviour.
_device: GPU device identifier used for FAISS operations.
_gpu_resources: FAISS GPU resources allocated for the index.

Examples:
>>> manager = FaissIndexManager(dim=128, config=DenseIndexConfig())  # doctest: +SKIP
>>> manager.ntotal  # doctest: +SKIP
0
