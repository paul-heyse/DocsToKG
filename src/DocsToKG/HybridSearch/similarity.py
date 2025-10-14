"""Shared helpers for cosine similarity using FAISS GPU fallbacks."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - exercised indirectly in GPU environments
    import faiss  # type: ignore

    _FAISS_AVAILABLE = hasattr(faiss, "pairwise_distance_gpu")
except Exception:  # pragma: no cover - fallback for CPU-only test rigs
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

_GPU_RESOURCES: Optional["faiss.StandardGpuResources"] = None


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return L2-normalised rows for cosine similarity operations."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def pairwise_inner_products(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise inner products, using GPU helpers when available."""

    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = matrix.astype(np.float32, copy=False)
    resources = _get_gpu_resources()
    if resources is not None and _FAISS_AVAILABLE:
        try:
            sims = faiss.pairwise_distance_gpu(  # type: ignore[attr-defined]
                resources,
                matrix,
                matrix,
                metric=faiss.METRIC_INNER_PRODUCT,
                device=0,
            )
            return np.asarray(sims, dtype=np.float32)
        except Exception:  # pragma: no cover - GPU helper may fail if device busy
            pass
    return matrix @ matrix.T


def max_inner_product(target: np.ndarray, corpus: np.ndarray) -> float:
    """Return the maximum inner product between a target vector and corpus rows."""

    if corpus.size == 0:
        return 0.0
    target = target.astype(np.float32, copy=False)
    corpus = corpus.astype(np.float32, copy=False)
    resources = _get_gpu_resources()
    if resources is not None and _FAISS_AVAILABLE:
        try:
            sims = faiss.pairwise_distance_gpu(  # type: ignore[attr-defined]
                resources,
                target.reshape(1, -1),
                corpus,
                metric=faiss.METRIC_INNER_PRODUCT,
                device=0,
            )
            return float(np.max(np.asarray(sims)))
        except Exception:  # pragma: no cover - GPU helper may fail if device busy
            pass
    return float(np.max(target @ corpus.T))


def _get_gpu_resources() -> Optional["faiss.StandardGpuResources"]:
    global _GPU_RESOURCES
    if not _FAISS_AVAILABLE or not hasattr(faiss, "StandardGpuResources"):
        return None
    if _GPU_RESOURCES is None:
        try:  # pragma: no cover - GPU path depends on host environment
            _GPU_RESOURCES = faiss.StandardGpuResources()
        except Exception:
            _GPU_RESOURCES = None
    return _GPU_RESOURCES

