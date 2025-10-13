# datasets.py
import os
import numpy as np

def generate_synthetic(nb: int, d: int, nq: int, seed: int = 123):
    """
    Generate synthetic base and query sets with float32 dtype.
    Shapes: xb: (nb, d), xq: (nq, d)
    """
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((nb, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    return xb, xq

def maybe_normalize_cosine(x: np.ndarray, enable: bool):
    """
    In-place L2 normalization for cosine similarity use-cases.
    """
    if not enable:
        return x
    # Avoid divide-by-zero
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1.0
    x /= norms
    return x

def load_npy_pair(xb_path: str, xq_path: str):
    """
    Load (xb, xq) from .npy files. Ensures float32 and C-contiguous layout.
    """
    xb = np.load(xb_path, mmap_mode=None)
    xq = np.load(xq_path, mmap_mode=None)
    xb = np.ascontiguousarray(xb.astype(np.float32))
    xq = np.ascontiguousarray(xq.astype(np.float32))
    return xb, xq

def ensure_ids(nb: int, use_ids: bool):
    """
    Return int64 IDs [0..nb-1] if use_ids is True; else None.
    """
    if not use_ids:
        return None
    return np.arange(nb, dtype=np.int64)
