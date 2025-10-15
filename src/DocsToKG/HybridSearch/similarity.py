"""GPU-accelerated cosine similarity helpers for HybridSearch."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

try:  # pragma: no cover - exercised indirectly in GPU environments
    import faiss  # type: ignore

    _FAISS_AVAILABLE = all(
        hasattr(faiss, attr)
        for attr in (
            "pairwise_distance_gpu",
            "StandardGpuResources",
            "normalize_L2",
        )
    )
except Exception:  # pragma: no cover - dependency may be absent in CI
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

__all__ = [
    "get_gpu_resources",
    "normalize_rows",
    "cosine_against_corpus_gpu",
    "pairwise_inner_products",
    "max_inner_product",
]

_GPU_RES_LOCK = threading.Lock()
_GPU_RES: Optional["faiss.StandardGpuResources"] = None


def get_gpu_resources() -> "faiss.StandardGpuResources":
    """Return a module-level `StandardGpuResources` singleton."""

    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS GPU helpers are unavailable")
    global _GPU_RES
    if _GPU_RES is None:
        with _GPU_RES_LOCK:
            if _GPU_RES is None:
                try:  # pragma: no cover - GPU path depends on host environment
                    _GPU_RES = faiss.StandardGpuResources()
                except Exception as exc:
                    raise RuntimeError("Unable to initialise FAISS GPU resources") from exc
    return _GPU_RES


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalise rows in-place for cosine similarity operations."""

    if matrix.dtype != np.float32 or not matrix.flags.c_contiguous:
        raise TypeError("normalize_rows expects a contiguous float32 array")
    if _FAISS_AVAILABLE and hasattr(faiss, "normalize_L2"):
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
    """Compute cosine similarities between a query vector and a corpus on GPU."""

    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS GPU helpers are unavailable")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.shape[1] != corpus.shape[1]:
        raise ValueError("Query and corpus dimensionality must match")

    q = _as_f32(query.copy())
    c = _as_f32(corpus.copy())
    normalize_rows(q)
    normalize_rows(c)
    res = resources or get_gpu_resources()
    sims = faiss.pairwise_distance_gpu(
        res,
        q,
        c,
        metric=faiss.METRIC_INNER_PRODUCT,
        device=int(device),
    )
    return np.asarray(sims, dtype=np.float32)


def pairwise_inner_products(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> np.ndarray:
    """Return pairwise cosine similarities between rows of `a` and `b` on GPU."""

    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS GPU helpers are unavailable")
    if a.size == 0:
        if b is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.zeros((0, b.shape[0]), dtype=np.float32)
    if b is None:
        b = a
    if a.shape[1] != b.shape[1]:
        raise ValueError("Input matrices must share the same dimensionality")

    same_input = b is a
    A = _as_f32(a.copy())
    normalize_rows(A)

    if same_input:
        B = A
    else:
        B = _as_f32(b.copy())
        normalize_rows(B)

    res = resources or get_gpu_resources()
    sims = faiss.pairwise_distance_gpu(
        res,
        A,
        B,
        metric=faiss.METRIC_INNER_PRODUCT,
        device=int(device),
    )
    return np.asarray(sims, dtype=np.float32)


def max_inner_product(target: np.ndarray, corpus: np.ndarray, *, device: int = 0) -> float:
    """Return the maximum cosine similarity between a target vector and corpus rows."""

    if corpus.size == 0:
        return 0.0
    sims = cosine_against_corpus_gpu(target, corpus, device=device)
    return float(np.max(sims))
