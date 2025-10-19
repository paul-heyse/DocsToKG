"""Unit tests for :mod:`DocsToKG.HybridSearch.store` FaissVectorStore helpers."""

from __future__ import annotations

from types import MethodType

import numpy as np

from DocsToKG.HybridSearch import store as store_module
from DocsToKG.HybridSearch.store import FaissVectorStore


class _DummyMetrics:
    """Collect metric observations emitted during coalescer tests."""

    def __init__(self) -> None:
        self.observations: list[tuple[str, float]] = []
        self.gauges: list[tuple[str, float]] = []

    def observe(self, name: str, value: float) -> None:
        self.observations.append((name, value))

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges.append((name, value))


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


def test_search_coalescer_iterative_execution_handles_many_micro_batches() -> None:
    """Ensure the coalescer drains large queues without recursive overflow."""

    class _DummyObservability:
        def __init__(self) -> None:
            self.metrics = _DummyMetrics()

    class _DummyStore:
        def __init__(self) -> None:
            self._dim = 1
            self._observability = _DummyObservability()
            self._counter = 0

        def _ensure_dim(self, vector: np.ndarray) -> np.ndarray:  # noqa: PLW0211 - test shim
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1 or arr.size != self._dim:
                raise AssertionError(f"unexpected vector shape {arr.shape}")
            return arr

        def _search_batch_impl(self, matrix: np.ndarray, top_k: int):  # noqa: PLW0211 - test shim
            rows = matrix.shape[0]
            results = []
            for _ in range(rows):
                self._counter += 1
                hits = [
                    store_module.FaissSearchResult(
                        vector_id=f"id-{self._counter}-{j}", score=float(top_k - j)
                    )
                    for j in range(top_k)
                ]
                results.append(hits)
            return results

    store = _DummyStore()
    coalescer = store_module._SearchCoalescer(store, window_ms=0.0, max_batch=1)

    total_requests = 1050
    pending = [
        store_module._PendingSearch(np.array([float(i)], dtype=np.float32), top_k=1)
        for i in range(total_requests)
    ]

    first_batch = [pending[0]]
    with coalescer._lock:  # type: ignore[attr-defined]
        coalescer._pending = pending[1:]  # type: ignore[attr-defined]

    coalescer._execute(first_batch)

    for request in pending:
        results = request.wait()
        assert len(results) == request.top_k == 1

    metrics = store._observability.metrics
    assert len(metrics.observations) == total_requests
    assert metrics.gauges and all(value == 0.0 for _, value in metrics.gauges)
