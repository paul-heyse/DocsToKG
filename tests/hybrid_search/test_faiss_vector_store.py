"""Unit tests for :mod:`DocsToKG.HybridSearch.store` FaissVectorStore helpers."""

from __future__ import annotations

from types import MethodType

import numpy as np

from DocsToKG.HybridSearch import store as store_module
from DocsToKG.HybridSearch.store import FaissVectorStore


def test_faiss_vector_store_search_batch_preserves_queries(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Ensure ``search_batch`` does not mutate the caller-provided query matrix."""

    def fake_normalize_rows(matrix: np.ndarray) -> np.ndarray:
        matrix += 1.0
        return matrix

    monkeypatch.setattr(store_module, "normalize_rows", fake_normalize_rows)

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 3  # type: ignore[attr-defined]
    store._search_coalescer = None  # type: ignore[attr-defined]

    captured: dict[str, np.ndarray] = {}

    def fake_search_batch_impl(self: FaissVectorStore, matrix: np.ndarray, top_k: int):
        captured["matrix"] = matrix
        return [[] for _ in range(matrix.shape[0])]

    store._search_batch_impl = MethodType(fake_search_batch_impl, store)  # type: ignore[attr-defined]

    queries = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    original = queries.copy()

    store.search_batch(queries, top_k=1)

    np.testing.assert_array_equal(queries, original)
    assert "matrix" in captured
    assert not np.may_share_memory(captured["matrix"], queries)
