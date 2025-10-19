"""Unit tests for ``FaissVectorStore`` convenience behaviours."""

from __future__ import annotations

import types

import numpy as np

from DocsToKG.HybridSearch import store as store_module
from DocsToKG.HybridSearch.pipeline import Observability


def test_search_batch_preserves_input_array(monkeypatch) -> None:
    """Ensure ``search_batch`` does not mutate the provided query matrix."""

    store = store_module.FaissVectorStore.__new__(store_module.FaissVectorStore)
    store._dim = 3
    store._search_coalescer = None
    store._observability = Observability()

    def fake_search_batch_impl(self, matrix: np.ndarray, top_k: int):
        return [
            [store_module.FaissSearchResult(vector_id="stub", score=0.0)]
            for _ in range(matrix.shape[0])
        ]

    store._search_batch_impl = types.MethodType(fake_search_batch_impl, store)

    def fake_normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix /= norms
        return matrix

    monkeypatch.setattr(store_module, "normalize_rows", fake_normalize_rows)

    queries = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    original = queries.copy()

    store.search_batch(queries, top_k=1)

    np.testing.assert_array_equal(queries, original)
