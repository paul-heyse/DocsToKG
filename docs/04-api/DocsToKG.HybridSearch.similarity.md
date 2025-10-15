# Module: similarity

GPU-accelerated cosine similarity helpers for HybridSearch.

## Functions

### `get_gpu_resources()`

Return a module-level `StandardGpuResources` singleton.

Args:
None

Returns:
faiss.StandardGpuResources: Shared GPU resource manager instance.

Raises:
RuntimeError: If FAISS GPU helpers are unavailable or initialisation fails.

### `_as_f32(x)`

*No documentation available.*

### `normalize_rows(matrix)`

Normalise rows in-place for cosine similarity operations.

Args:
matrix: Contiguous float32 array whose rows should be L2 normalised.

Returns:
np.ndarray: The same matrix with rows normalised to unit length.

Raises:
TypeError: If the input array is not contiguous float32.

### `cosine_against_corpus_gpu(query, corpus)`

Compute cosine similarities between a query vector and a corpus on GPU.

Args:
query: Query vector or matrix to compare against the corpus.
corpus: Corpus matrix containing candidate vectors.
device: GPU device index to execute the computation.
resources: Optional pre-created FAISS GPU resources.

Returns:
np.ndarray: Matrix of cosine similarities between query and corpus rows.

Raises:
RuntimeError: If FAISS GPU helpers are unavailable.
ValueError: If the query and corpus dimensionalities differ.

### `pairwise_inner_products(a, b)`

Return pairwise cosine similarities between rows of `a` and `b` on GPU.

Args:
a: Matrix containing source vectors.
b: Optional matrix containing comparison vectors (defaults to `a`).
device: GPU device index to execute the computation.
resources: Optional FAISS GPU resources to reuse.

Returns:
np.ndarray: Matrix of cosine similarities between each row of `a` and `b`.

Raises:
RuntimeError: If FAISS GPU helpers are unavailable.
ValueError: If the input matrices have different dimensionality.

### `max_inner_product(target, corpus)`

Return the maximum cosine similarity between a target vector and corpus rows.

Args:
target: Vector whose similarity against the corpus is evaluated.
corpus: Matrix containing candidate vectors.
device: GPU device index used for computation.

Returns:
float: Maximum cosine similarity score.

Raises:
RuntimeError: If FAISS GPU helpers are unavailable.
