"""GPU-accelerated cosine similarity helpers for HybridSearch."""

from __future__ import annotations

from typing import Optional

import numpy as np

import faiss  # type: ignore

__all__ = [
    "normalize_rows",
    "cosine_against_corpus_gpu",
    "pairwise_inner_products",
    "max_inner_product",
    "cosine_batch",
]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalise rows in-place for cosine similarity operations.

    Args:
        matrix: Contiguous ``float32`` array whose rows will be normalised.

    Returns:
        The same array instance with each row scaled to unit length.

    Raises:
        TypeError: If ``matrix`` is not a contiguous ``float32`` array.
    """

    if matrix.dtype != np.float32 or not matrix.flags.c_contiguous:
        raise TypeError("normalize_rows expects a contiguous float32 array")
    if hasattr(faiss, "normalize_L2"):
        faiss.normalize_L2(matrix)
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix /= norms
    return matrix


def cosine_against_corpus_gpu(
    query: np.ndarray,
    corpus: np.ndarray,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> np.ndarray:
    """Compute cosine similarities between a query vector and a corpus on GPU.

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
    """

    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.shape[1] != corpus.shape[1]:
        raise ValueError("Query and corpus dimensionality must match")
    sims = cosine_batch(query, corpus, device=device, resources=resources)
    return np.asarray(sims, dtype=np.float32)


def pairwise_inner_products(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> np.ndarray:
    """Return pairwise cosine similarities between rows of ``a`` and ``b`` on GPU.

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
    """

    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if a.size == 0:
        if b is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((0, b.shape[0]), dtype=np.float32)
    if b is None:
        b = a
    if a.shape[1] != b.shape[1]:
        raise ValueError("Input matrices must share the same dimensionality")
    sims = cosine_batch(a, b, device=device, resources=resources)
    return np.asarray(sims, dtype=np.float32)


def max_inner_product(
    target: np.ndarray,
    corpus: np.ndarray,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> float:
    """Return the maximum cosine similarity between ``target`` and rows in ``corpus``.

    Args:
        target: Vector whose similarity to the corpus is evaluated.
        corpus: Matrix containing comparison vectors.
        device: GPU device ordinal supplied to FAISS.
        resources: Initialised FAISS GPU resources object.

    Returns:
        Maximum cosine similarity value as a ``float``. Returns ``-inf`` for an empty corpus.

    Raises:
        RuntimeError: If GPU resources are not provided.
    """

    if resources is None:
        raise RuntimeError("FAISS GPU resources are required for cosine comparisons")
    if corpus.size == 0:
        return float("-inf")
    sims = cosine_against_corpus_gpu(target, corpus, device=device, resources=resources)
    return float(np.max(sims))


def cosine_batch(
    q: np.ndarray,
    C: np.ndarray,
    *,
    device: int,
    resources: "faiss.StandardGpuResources",
) -> np.ndarray:
    """Return batched cosine similarities using FAISS GPU kernels.

    Args:
        q: Query vectors to compare against the corpus.
        C: Corpus vectors providing candidate matches.
        device: GPU device identifier used for computation.
        resources: Preallocated FAISS GPU resource manager.

    Returns:
        np.ndarray: Cosine similarity matrix sized ``(len(q), len(C))``.
    """

    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = np.ascontiguousarray(q, dtype=np.float32).copy()
    C = np.ascontiguousarray(C, dtype=np.float32).copy()
    faiss.normalize_L2(q)
    faiss.normalize_L2(C)
    return faiss.pairwise_distance_gpu(
        resources,
        q,
        C,
        metric=faiss.METRIC_INNER_PRODUCT,
        device=int(device),
    )
