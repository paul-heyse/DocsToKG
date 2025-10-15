"""Shared helpers for cosine similarity using FAISS GPU primitives."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - exercised indirectly in GPU environments
    import faiss  # type: ignore

    _FAISS_AVAILABLE = all(
        hasattr(faiss, attr)
        for attr in (
            "pairwise_distance_gpu",
            "StandardGpuResources",
        )
    )
except Exception:  # pragma: no cover - dependency may be absent in CI
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

_GPU_RESOURCES: Optional["faiss.StandardGpuResources"] = None


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return L2-normalised rows for cosine similarity operations.

    Args:
        matrix: Input matrix whose rows represent embedding vectors.

    Returns:
        Matrix with each row scaled to unit L2 norm. Zero vectors are preserved.
    """

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def pairwise_inner_products(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise inner products using FAISS GPU helpers.

    Args:
        matrix: L2-normalised embedding matrix.

    Returns:
        Square matrix of inner products between all row pairs.

    Raises:
        RuntimeError: If FAISS GPU operations fail or are unavailable.
    """

    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = matrix.astype(np.float32, copy=False)
    resources = _require_gpu_resources()
    try:
        sims = faiss.pairwise_distance_gpu(  # type: ignore[attr-defined]
            resources,
            matrix,
            matrix,
            metric=faiss.METRIC_INNER_PRODUCT,
            device=0,
        )
    except Exception as exc:  # pragma: no cover - GPU helper failures propagate
        raise RuntimeError("FAISS pairwise_distance_gpu failed") from exc
    return np.asarray(sims, dtype=np.float32)


def max_inner_product(target: np.ndarray, corpus: np.ndarray) -> float:
    """Return the maximum inner product between a target vector and corpus rows.

    Args:
        target: Query vector expected to be L2-normalised.
        corpus: Matrix of candidate vectors for comparison.

    Returns:
        Maximum inner product value between the target and corpus rows.

    Raises:
        RuntimeError: If FAISS GPU operations fail or are unavailable.
    """

    if corpus.size == 0:
        return 0.0
    target = target.astype(np.float32, copy=False)
    corpus = corpus.astype(np.float32, copy=False)
    resources = _require_gpu_resources()
    try:
        sims = faiss.pairwise_distance_gpu(  # type: ignore[attr-defined]
            resources,
            target.reshape(1, -1),
            corpus,
            metric=faiss.METRIC_INNER_PRODUCT,
            device=0,
        )
    except Exception as exc:  # pragma: no cover - GPU helper failures propagate
        raise RuntimeError("FAISS pairwise_distance_gpu failed") from exc
    return float(np.max(np.asarray(sims)))


def _require_gpu_resources() -> "faiss.StandardGpuResources":
    """Initialise FAISS GPU resources if they are available.

    Returns:
        Shared FAISS `StandardGpuResources` instance.

    Raises:
        RuntimeError: If FAISS GPU support is missing or resources cannot be initialised.
    """
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS GPU helpers are unavailable")
    global _GPU_RESOURCES
    if _GPU_RESOURCES is None:
        try:  # pragma: no cover - GPU path depends on host environment
            _GPU_RESOURCES = faiss.StandardGpuResources()
        except Exception as exc:
            raise RuntimeError("Unable to initialise FAISS GPU resources") from exc
    return _GPU_RESOURCES
