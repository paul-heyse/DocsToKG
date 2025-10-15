# 1. Module: similarity

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.vectorstore``.

GPU-accelerated cosine similarity helpers for HybridSearch.

## 1. Functions

### `normalize_rows(matrix)`

Normalise rows in-place for cosine similarity operations.

Args:
matrix: Contiguous ``float32`` array whose rows will be normalised.

Returns:
The same array instance with each row scaled to unit length.

Raises:
TypeError: If ``matrix`` is not a contiguous ``float32`` array.

### `cosine_against_corpus_gpu(query, corpus)`

Compute cosine similarities between a query vector and a corpus on GPU.

Args:
query: 1D or 2D array containing the query vector(s).
corpus: 2D array of candidate vectors to compare against ``query``.
device: GPU device ordinal passed to FAISS.
resources: Initialised FAISS GPU resources object.

Returns:
``float32`` matrix of cosine similarities shaped ``(len(query), len(corpus))``.

Raises:
RuntimeError: If GPU resources are not provided.
ValueError: If ``query`` and ``corpus`` dimensions are incompatible.

### `pairwise_inner_products(a, b)`

Return pairwise cosine similarities between rows of ``a`` and ``b`` on GPU.

Args:
a: Matrix holding the first set of vectors to compare.
b: Optional matrix of comparison vectors; defaults to ``a`` when omitted.
device: GPU device ordinal supplied to FAISS.
resources: Initialised FAISS GPU resources object.

Returns:
``float32`` matrix of cosine similarities.

Raises:
RuntimeError: If GPU resources are not provided.
ValueError: When ``a`` and ``b`` have mismatching dimensionality.

### `max_inner_product(target, corpus)`

Return the maximum cosine similarity between ``target`` and rows in ``corpus``.

Args:
target: Vector whose similarity to the corpus is evaluated.
corpus: Matrix containing comparison vectors.
device: GPU device ordinal supplied to FAISS.
resources: Initialised FAISS GPU resources object.

Returns:
Maximum cosine similarity value as a ``float``. Returns ``-inf`` for an empty corpus.

Raises:
RuntimeError: If GPU resources are not provided.
