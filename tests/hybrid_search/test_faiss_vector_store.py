"""Unit tests for :mod:`DocsToKG.HybridSearch.store` FaissVectorStore helpers."""

from __future__ import annotations

from threading import Event, RLock, Thread
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


def test_remove_ids_is_atomic_across_threads() -> None:
    """Verify ``remove_ids`` holds the adapter lock until operations complete."""

    class DummyStore:
        remove_ids = FaissVectorStore.remove_ids

        def __init__(self) -> None:
            self._lock = RLock()
            self._events: list[tuple[str, object]] = []
            self._remove_started = Event()
            self._allow_remove_finish = Event()
            self._search_attempted = Event()
            self._search_entered = Event()
            self._allow_search_finish = Event()

        def _remove_ids(self, ids: np.ndarray) -> None:
            self._events.append(("remove", tuple(int(v) for v in ids.tolist())))
            self._remove_started.set()
            assert self._allow_remove_finish.wait(timeout=1.0), "remove_ids wait timed out"

        def _flush_pending_deletes(self, *, force: bool) -> None:
            self._events.append(("flush", bool(force)))

        def _update_gpu_metrics(self) -> None:
            self._events.append(("update", None))

        def _maybe_refresh_snapshot(self, *, writes_delta: int, reason: str) -> None:
            self._events.append(("snapshot", int(writes_delta), str(reason)))

        def search(self, query: np.ndarray, top_k: int) -> list[object]:
            self._search_attempted.set()
            with self._lock:
                self._search_entered.set()
                assert self._allow_search_finish.wait(
                    timeout=1.0
                ), "search wait timed out"
            return []

    store = DummyStore()
    ids = np.array([1, 2], dtype=np.int64)
    remove_counts: list[int] = []

    def run_remove() -> None:
        remove_counts.append(
            store.remove_ids(ids, force_flush=True, reason="test_atomic_remove")
        )

    def run_search() -> None:
        store.search(np.zeros((1,), dtype=np.float32), top_k=1)

    remover = Thread(target=run_remove)
    remover.start()
    assert store._remove_started.wait(timeout=1.0)

    searcher = Thread(target=run_search)
    searcher.start()
    assert store._search_attempted.wait(timeout=1.0)
    assert not store._search_entered.wait(timeout=0.1)

    store._allow_remove_finish.set()
    remover.join(timeout=1.0)
    assert not remover.is_alive()

    assert store._search_entered.wait(timeout=1.0)
    store._allow_search_finish.set()
    searcher.join(timeout=1.0)
    assert not searcher.is_alive()

    assert remove_counts == [2]
    assert store._events == [
        ("remove", (1, 2)),
        ("flush", True),
        ("update", None),
        ("snapshot", 2, "test_atomic_remove"),
    ]
