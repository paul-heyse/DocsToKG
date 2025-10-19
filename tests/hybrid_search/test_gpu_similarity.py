"""Regression tests for GPU cosine similarity helpers."""

from __future__ import annotations

import numpy as np
import pytest

from DocsToKG.HybridSearch.store import cosine_batch


def _fake_pairwise(resources, queries, corpus, *, metric, device):  # pragma: no cover - exercised via tests
    """Minimal FAISS kernel stand-in returning cosine similarities."""

    return queries @ corpus.T


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_cosine_batch_preserves_inputs(shape):
    faiss = pytest.importorskip("faiss", reason="FAISS runtime is required for cosine_batch")

    q = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    C = np.linspace(0.1, 0.9, num=6, dtype=np.float32).reshape(2, 3)

    q_before = q.copy()
    C_before = C.copy()

    result = cosine_batch(
        q,
        C,
        device=0,
        resources=object(),
        pairwise_fn=_fake_pairwise,
    )

    assert result.shape == (1, 2) if len(shape) == 1 else (shape[0], 2)
    np.testing.assert_allclose(q, q_before)
    np.testing.assert_allclose(C, C_before)

    # Ensure the normalization ran by checking the cosine similarity result against FAISS reference.
    normalized_q = q_before.astype(np.float32, copy=True)
    normalized_C = C_before.astype(np.float32, copy=True)
    faiss.normalize_L2(normalized_q.reshape(1, -1) if normalized_q.ndim == 1 else normalized_q)
    faiss.normalize_L2(normalized_C)
    expected = normalized_q.reshape(1, -1) @ normalized_C.T if normalized_q.ndim == 1 else normalized_q @ normalized_C.T
    np.testing.assert_allclose(result, expected)
