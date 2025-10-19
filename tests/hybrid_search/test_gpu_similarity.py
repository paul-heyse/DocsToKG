"""Regression tests for GPU cosine similarity helpers."""

from __future__ import annotations

import numpy as np
import pytest

from DocsToKG.HybridSearch.store import cosine_batch


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_cosine_batch_normalizes_in_place_and_reuses_contiguous_buffers(shape):
    faiss = pytest.importorskip("faiss", reason="FAISS runtime is required for cosine_batch")

    q = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    C = np.linspace(0.1, 0.9, num=6, dtype=np.float32).reshape(2, 3)

    q_before = q.copy()
    C_before = C.copy()
    captured: dict[str, np.ndarray] = {}

    def _tracking_pairwise(resources, queries, corpus, *, metric, device):
        captured["queries"] = queries
        captured["corpus"] = corpus
        return queries @ corpus.T

    result = cosine_batch(
        q,
        C,
        device=0,
        resources=object(),
        pairwise_fn=_tracking_pairwise,
    )

    assert result.shape == (1, 2) if len(shape) == 1 else (shape[0], 2)

    normalized_q = np.asarray(q_before, dtype=np.float32)
    q_view = normalized_q.reshape(1, -1) if normalized_q.ndim == 1 else normalized_q
    normalized_C = np.asarray(C_before, dtype=np.float32)
    faiss.normalize_L2(q_view)
    faiss.normalize_L2(normalized_C)
    expected = q_view @ normalized_C.T

    np.testing.assert_allclose(result, expected)
    if normalized_q.ndim == 1:
        np.testing.assert_allclose(q, q_view.reshape(-1))
        assert captured["queries"].base is q
    else:
        np.testing.assert_allclose(q, q_view)
        assert captured["queries"] is q
    np.testing.assert_allclose(C, normalized_C)
    assert captured["corpus"] is C
    assert np.shares_memory(captured["queries"], q)
    assert np.shares_memory(captured["corpus"], C)
