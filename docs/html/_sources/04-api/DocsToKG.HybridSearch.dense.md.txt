# 1. Module: dense

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.dense``.

GPU-only FAISS index management for dense retrieval.

## 1. Functions

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

### `gpu_resources(self)`

Return the FAISS GPU resources backing the index.

Args:
None

Returns:
`faiss.StandardGpuResources` instance when GPU execution is enabled, otherwise ``None``.

### `device(self)`

Return the GPU device identifier assigned to the index manager.

Args:
None

Returns:
Integer CUDA device ordinal used for FAISS kernels.

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

### `remove_ids(self, ids)`

Remove vectors by FAISS integer identifiers with optional batching.

Args:
ids: Array of FAISS internal identifiers to remove.
force_flush: When True, rebuild immediately instead of batching.

Returns:
Number of identifiers scheduled for removal.

### `_current_index_ids(self)`

Return the FAISS internal identifiers currently stored in the index.

Args:
None

Returns:
NumPy array containing the FAISS internal IDs present in the index.

### `_lookup_existing_ids(self, candidate_ids)`

Identify which of the supplied FAISS IDs already exist in the index.

Args:
candidate_ids: Array of FAISS internal IDs to test for membership.

Returns:
NumPy array containing the subset of ``candidate_ids`` currently stored.

### `search(self, query, top_k)`

Execute a cosine-similarity search returning the best `top_k` results.

Args:
query: Query vector to search against the index.
top_k: Maximum number of nearest neighbours to return.

Returns:
List of `FaissSearchResult` objects ordered by score.

### `serialize(self)`

Serialize the FAISS index to bytes.

Args:
None

Returns:
Byte string containing the serialized FAISS index.

Raises:
RuntimeError: If the index has not been initialised.

### `save(self, path)`

Persist the FAISS index to disk without CPU fallbacks.

Args:
path: Destination filepath for the serialized FAISS index.

Returns:
None

Raises:
RuntimeError: If the index has not been initialised.

### `load(cls, path, config, dim)`

Load a FAISS index from disk and ensure it resides on GPU.

Args:
path: Filesystem path pointing to a serialized FAISS index.
config: Dense index configuration used to rebuild runtime properties.
dim: Dimensionality of the vectors contained in the index.

Returns:
Instance of ``FaissIndexManager`` with GPU state initialised.

Raises:
RuntimeError: If the FAISS index cannot be read or promoted to GPU memory.

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

Construct a GPU-native FAISS index based on the configured index_type.

Args:
None

Returns:
Initialised ``faiss.Index`` residing on the configured GPU.

Raises:
RuntimeError: If GPU resources have not been initialised before invocation.

### `replicate_to_all_gpus(self, index)`

Replicate a FAISS index across every visible GPU device.

### `distribute_to_all_gpus(self, index)`

Replicate or shard a FAISS index across all GPUs depending on ``shard``.

When ``shard`` is ``True`` each GPU owns a disjoint partition. Otherwise the full index
is replicated across every GPU.

### `_maybe_to_gpu(self, index)`

Promote a CPU index to GPU while enforcing strict cloning guarantees.

Args:
index: CPU-resident FAISS index to migrate onto the configured GPU(s).

Returns:
GPU-backed FAISS index, optionally replicated across devices.

Raises:
RuntimeError: If GPU resources are unavailable or promotion fails.

### `_maybe_reserve_memory(self, index)`

Reserve GPU memory for the expected corpus size when supported.

Args:
index: Target FAISS index whose underlying storage may be extended.

Returns:
None

### `_to_cpu(self, index)`

Clone a GPU index back to CPU memory for persistence operations.

Args:
index: GPU-resident FAISS index to convert to CPU representation.

Returns:
CPU-backed FAISS index ready for serialization.

Raises:
RuntimeError: If FAISS lacks GPU→CPU conversion support.

### `_set_nprobe(self)`

Apply IVF search breadth configuration to the current index.

Args:
None

Returns:
None

### `_log_index_configuration(self, index)`

Emit structured logging that captures the live index configuration.

Args:
index: FAISS index whose configuration should be summarised.

Returns:
None

### `_resolve_vector_id(self, internal_id)`

Translate a FAISS internal identifier to the original vector UUID.

Args:
internal_id: FAISS-managed integer identifier produced during search.

Returns:
Matching vector UUID when available, otherwise ``None``.

### `_detect_device(self, index)`

Determine the CUDA device hosting the provided index, if any.

Args:
index: FAISS index whose device affinity is being inspected.

Returns:
CUDA device ordinal when discoverable, otherwise ``None``.

### `_resolve_device(self, config)`

Determine the target GPU device, honouring runtime overrides.

Args:
config: Dense index configuration supplying the default device.

Returns:
int: GPU device identifier to use for FAISS operations.

### `init_gpu(self)`

Initialise FAISS GPU resources in line with configuration settings.

Args:
None

Returns:
None

Raises:
RuntimeError: If FAISS lacks GPU support or an unsuitable device is requested.

### `_ensure_dim(self, vector)`

Validate that the provided embedding matches the configured dimensionality.

Args:
vector: Vector whose shape should align with the configured dimension.

Returns:
NumPy array validated to match ``self._dim``.

Raises:
ValueError: If the vector dimensionality differs from ``self._dim``.

### `_flush_pending_deletes(self)`

Rebuild the index when tombstone thresholds or manual flush requests demand it.

Args:
force: When ``True``, rebuild even if thresholds have not been reached.

Returns:
None

### `_remove_ids(self, ids)`

Attempt to delete FAISS IDs directly, falling back to tombstones when required.

Args:
ids: Array of FAISS internal identifiers targeted for removal.

Returns:
None

### `_rebuild_index(self)`

Recreate the GPU index from live vectors after tombstones accumulate.

Args:
None

Returns:
None

## 2. Classes

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
