"""GPU-native cosine similarity helpers leveraged by HybridSearch pipelines."""

from __future__ import annotations

import numpy as np

import faiss  # type: ignore


def cosine_batch(
    q: np.ndarray,
    C: np.ndarray,
    *,
    device: int,
    resources: faiss.StandardGpuResources,
) -> np.ndarray:
    """Return batched cosine similarities using FAISS GPU kernels.

    Args:
        q: Query vectors (1D or 2D) to normalise and compare.
        C: Corpus matrix providing comparison vectors.
        device: GPU device ordinal supplied to FAISS operations.
        resources: Initialised FAISS GPU resources reused across calls.

    Returns:
        ``float32`` matrix containing cosine similarities for each query/corpus pair.
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
