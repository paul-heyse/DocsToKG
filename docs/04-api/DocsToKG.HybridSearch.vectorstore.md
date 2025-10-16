# 1. Module: vectorstore

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.vectorstore``.

## 1. Overview

Unified FAISS vector store, GPU similarity utilities, and state helpers.

## 2. Functions

### `normalize_rows(matrix)`

L2-normalise each row of ``matrix`` in-place.

Args:
matrix: Contiguous float32 matrix whose rows represent vectors.

Returns:
Normalised matrix (same object as ``matrix``).

Raises:
TypeError: If ``matrix`` is not a contiguous float32 ``ndarray``.

### `cosine_against_corpus_gpu(query, corpus)`

Compute cosine similarity between ``query`` and each vector in ``corpus`` on GPU.

Args:
query: Query vector or matrix (``N x D``) to compare.
corpus: Matrix representing the corpus (``M x D``).
device: CUDA device id used for the computation.
resources: FAISS GPU resources to execute the kernel.

Returns:
numpy.ndarray: Matrix of cosine similarities with shape ``(N, M)``.

Raises:
RuntimeError: If GPU resources are unavailable.
ValueError: When the query and corpus dimensionality mismatch.

### `pairwise_inner_products(a, b)`

Compute cosine similarities between two corpora on GPU.

Args:
a: First matrix of vectors.
b: Optional second matrix; when omitted ``a`` is used (symmetrical case).
device: CUDA device id to run the computation.
resources: FAISS GPU resources backing the computation.

Returns:
numpy.ndarray: Pairwise similarity matrix with rows from ``a`` and columns from ``b``.

Raises:
RuntimeError: If GPU resources are unavailable.
ValueError: When input matrices have mismatched dimensionality.

### `max_inner_product(target, corpus)`

Return the maximum cosine similarity between ``target`` and ``corpus``.

Args:
target: Query vector whose maximum similarity is desired.
corpus: Corpus matrix used for comparison.
device: CUDA device id for FAISS computations.
resources: FAISS GPU resources needed for similarity search.

Returns:
float: Maximum cosine similarity value.

Raises:
RuntimeError: If GPU resources are unavailable.

### `cosine_batch(q, C)`

Helper that normalises and computes cosine similarities on GPU.

Args:
q: Query matrix (``N x D``).
C: Corpus matrix (``M x D``).
device: CUDA device id used for computation.
resources: FAISS GPU resources backing the kernel.

Returns:
numpy.ndarray: Pairwise cosine similarities with shape ``(N, M)``.

### `cosine_topk_blockwise(q, C)`

Return Top-K cosine similarities between ``q`` and ``C`` using GPU tiling.

The helper avoids materialising the full ``(N × M)`` similarity matrix by
iterating over ``C`` in row blocks and maintaining a running Top-K per query
row. Inputs are copied and normalised inside the routine so callers retain
ownership of their buffers.

Args:
q: Query vector or matrix (``N × D``).
C: Corpus matrix (``M × D``).
k: Number of neighbours to return per query row.
device: CUDA device ordinal used for FAISS kernels.
resources: FAISS GPU resources backing ``pairwise_distance_gpu``.
block_rows: Number of corpus rows processed per iteration.

Returns:
Tuple ``(scores, indices)`` where each has shape ``(N × K)``. Scores are
sorted in descending order for every query row and indices reference rows
within ``C``.

### `serialize_state(faiss_index, registry)`

Serialize the vector store and chunk registry to a JSON-safe payload.

Args:
faiss_index: Vector store whose state should be captured.
registry: Chunk registry providing vector identifier mappings.

Returns:
dict[str, object]: Dictionary containing base64-encoded FAISS bytes and registry ids.

### `restore_state(faiss_index, payload)`

Restore the vector store from a payload produced by :func:`serialize_state`.

Args:
faiss_index: Vector store receiving the restored state.
payload: Mapping with ``faiss`` (base64) and registry vector ids.

Returns:
None

Raises:
ValueError: If the payload is missing the FAISS byte stream.

### `ntotal(self)`

Number of vectors currently stored in the FAISS index.

Args:
None

Returns:
int: Total number of stored vectors.

### `config(self)`

Return the resolved dense index configuration.

Args:
None

Returns:
DenseIndexConfig: Active dense index configuration.

### `dim(self)`

Return the dimensionality of vectors stored in the FAISS index.

Args:
None

Returns:
int: Dimensionality of embeddings managed by the index.

### `gpu_resources(self)`

Expose the underlying FAISS GPU resource manager.

Args:
None

Returns:
faiss.StandardGpuResources | None: GPU resource manager when initialised.

### `device(self)`

Return the CUDA device id used for the FAISS index.

Args:
None

Returns:
int: Configured CUDA device identifier.

### `set_id_resolver(self, resolver)`

Register a callback that maps FAISS internal ids to external ids.

Args:
resolver: Callable receiving the FAISS integer id and returning the
application-level identifier (or ``None`` when unresolved).

Returns:
None

### `train(self, vectors)`

Train IVF style indexes using the provided sample vectors.

Args:
vectors: Sequence of dense vectors used to train IVF quantizers.

Raises:
ValueError: If no vectors are supplied for a trainable index type.

Returns:
None

### `needs_training(self)`

Return ``True`` when the current FAISS index still requires training.

Args:
None

Returns:
bool: ``True`` when additional training is required.

### `add(self, vectors, vector_ids)`

Add vectors to the FAISS index, replacing existing ids if present.

Args:
vectors: Embeddings to insert into the index.
vector_ids: Application-level identifiers paired with each vector.

Raises:
ValueError: When the vector and id sequences differ in length.

Returns:
None

### `remove(self, vector_ids)`

Remove vectors from the index using application-level identifiers.

Args:
vector_ids: Sequence of vector ids scheduled for deletion.

Returns:
None

### `remove_ids(self, ids)`

Remove vectors using FAISS internal ids.

Args:
ids: Array of FAISS integer ids to delete.
force_flush: Force immediate rebuild when tombstones accumulate.

Returns:
Number of ids scheduled for removal.

### `_current_index_ids(self)`

*No documentation available.*

### `_lookup_existing_ids(self, candidate_ids)`

*No documentation available.*

### `search(self, query, top_k)`

Search the index for the ``top_k`` nearest neighbours of ``query``.

Args:
query: Dense query vector with dimensionality ``self._dim``.
top_k: Maximum number of nearest neighbours to return.

Returns:
Ranked list of :class:`FaissSearchResult` objects.

### `serialize(self)`

Return a CPU-serialised representation of the FAISS index.

Args:
None

Returns:
Byte payload produced by :func:`faiss.serialize_index`.

Raises:
RuntimeError: If the index has not been initialised.

### `save(self, path)`

Persist the FAISS index to ``path`` when persistence is enabled.

Args:
path: Filesystem destination for the serialised index.

Returns:
None

Raises:
RuntimeError: If the index has not been initialised.

### `load(cls, path, config, dim)`

Restore a vector store from disk.

Args:
path: Filesystem path containing a previously saved index payload.
config: Dense index configuration to apply to the reloaded store.
dim: Dimensionality of vectors stored in the index.

Returns:
Fresh :class:`FaissVectorStore` instance initialised from ``path``.

Raises:
OSError: If ``path`` cannot be read from disk.

### `restore(self, payload)`

Load an index from ``payload`` and promote it to the GPU.

Args:
payload: Bytes produced by :meth:`serialize`.

Raises:
ValueError: If the payload is empty.

Returns:
None

### `stats(self)`

Return diagnostic metrics describing the active FAISS index.

Args:
None

Returns:
Mapping of human-readable metric names to values (counts or strings).

### `rebuild_needed(self)`

Return ``True`` when tombstones require a full FAISS rebuild.

Args:
None

Returns:
bool: ``True`` when a rebuild should be triggered.

### `init_gpu(self)`

Initialise FAISS GPU resources for the configured CUDA device.

Args:
None

Raises:
RuntimeError: When GPU support is unavailable or the requested
device id is invalid.

Returns:
None

### `distribute_to_all_gpus(self, index)`

Clone ``index`` across available GPUs when the build supports it.

Args:
index: FAISS index to replicate or shard.
shard: When ``True`` attempt sharded replication if supported.

Returns:
FAISS index after attempting replication/sharding.

Raises:
RuntimeError: If sharding is requested but unsupported by the
linked FAISS build.

### `_maybe_to_gpu(self, index)`

*No documentation available.*

### `_maybe_reserve_memory(self, index)`

*No documentation available.*

### `_to_cpu(self, index)`

*No documentation available.*

### `_set_nprobe(self)`

*No documentation available.*

### `_log_index_configuration(self, index)`

*No documentation available.*

### `_create_index(self)`

*No documentation available.*

### `_probe_remove_support(self)`

*No documentation available.*

### `_remove_ids(self, ids)`

*No documentation available.*

### `_flush_pending_deletes(self)`

*No documentation available.*

### `_rebuild_index(self)`

*No documentation available.*

### `_resolve_vector_id(self, internal_id)`

*No documentation available.*

### `_ensure_dim(self, vector)`

*No documentation available.*

### `_detect_device(self, index)`

*No documentation available.*

### `_resolve_device(self, config)`

*No documentation available.*

## 3. Classes

### `FaissSearchResult`

Dense search hit returned by FAISS.

Attributes:
vector_id: External identifier resolved from the FAISS internal id.
score: Similarity score reported by FAISS (inner product).

Examples:
>>> FaissSearchResult(vector_id="doc-123", score=0.82)
FaissSearchResult(vector_id='doc-123', score=0.82)

### `FaissVectorStore`

Manage a GPU-backed FAISS index for dense retrieval.

The store encapsulates FAISS initialisation, GPU resource management,
vector ingestion, deletion, and similarity search. It understands the
custom CUDA-enabled FAISS build shipped with DocsToKG (``faiss-1.12``)
and automatically mirrors configuration options defined in
:class:`DenseIndexConfig`.

Attributes:
_dim: Dimensionality of stored vectors.
_config: Active dense index configuration.
_index: Underlying FAISS GPU index (possibly wrapped in ``IndexIDMap2``).
_gpu_resources: Lazily initialised ``StandardGpuResources`` instance.
_device: CUDA device id derived from configuration.

Examples:
>>> store = FaissVectorStore(dim=1536, config=DenseIndexConfig())
>>> store.add([np.random.rand(1536)], ["chunk-1"])
>>> results = store.search(np.random.rand(1536), top_k=5)
>>> [hit.vector_id for hit in results]
['chunk-1']
