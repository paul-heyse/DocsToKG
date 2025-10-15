# 1. Module: similarity_gpu

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.vectorstore_gpu``.

GPU-native cosine similarity helpers leveraged by HybridSearch pipelines.

## 1. Functions

### `cosine_batch(q, C)`

Return batched cosine similarities using FAISS GPU kernels.

Args:
q: Query vectors (1D or 2D) to normalise and compare.
C: Corpus matrix providing comparison vectors.
device: GPU device ordinal supplied to FAISS operations.
resources: Initialised FAISS GPU resources reused across calls.

Returns:
``float32`` matrix containing cosine similarities for each query/corpus pair.
