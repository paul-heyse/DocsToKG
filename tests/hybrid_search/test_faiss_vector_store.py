"""Unit tests for :mod:`DocsToKG.HybridSearch.store` FaissVectorStore helpers."""

from __future__ import annotations

import time
from threading import Event, RLock, Thread
from types import MethodType, SimpleNamespace

import numpy as np

from DocsToKG.HybridSearch.config import DenseIndexConfig
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


class _NullContext:
    """Minimal context manager that performs no additional work."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


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


def test_coerce_batch_allows_opt_out_of_normalization(monkeypatch: "pytest.MonkeyPatch") -> None:
    """``_coerce_batch`` should only normalise rows when explicitly requested."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 4  # type: ignore[attr-defined]

    original_normalize = store_module.normalize_rows
    calls: list[np.ndarray] = []

    def recording_normalize(matrix: np.ndarray) -> np.ndarray:
        calls.append(matrix.copy())
        return original_normalize(matrix)

    monkeypatch.setattr(store_module, "normalize_rows", recording_normalize)

    if store_module.faiss is None:
        monkeypatch.setattr(store_module, "faiss", SimpleNamespace(), raising=False)

    def fake_normalize(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms

    monkeypatch.setattr(store_module.faiss, "normalize_L2", fake_normalize, raising=False)

    vectors = np.array([[1.0, 2.0, 0.5, 0.75], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    normalised = store._coerce_batch(vectors, normalize=True)
    np.testing.assert_allclose(
        np.linalg.norm(normalised, axis=1), np.ones(normalised.shape[0]), rtol=1e-5
    )
    assert len(calls) == 1

    bypassed = store._coerce_batch(vectors, normalize=False)
    assert len(calls) == 1, "normalize_rows must not run when normalize=False"
    np.testing.assert_array_equal(bypassed, vectors)
    norms = np.linalg.norm(bypassed, axis=1)
    assert not np.allclose(norms, np.ones_like(norms)), "Vectors should remain unnormalised"


def test_add_calls_faiss_normalize_once(monkeypatch: "pytest.MonkeyPatch") -> None:
    """``add`` should rely on FAISS to normalise vectors exactly once per batch."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 3  # type: ignore[attr-defined]
    store._config = SimpleNamespace(
        ingest_dedupe_threshold=0.0, nlist=1, ivf_train_factor=1
    )  # type: ignore[attr-defined]
    store._lock = RLock()  # type: ignore[attr-defined]
    store._observability = SimpleNamespace(  # type: ignore[attr-defined]
        trace=lambda *a, **k: _NullContext(),
        metrics=SimpleNamespace(increment=lambda *a, **k: None),
    )
    store._as_pinned = MethodType(lambda self, matrix: matrix, store)  # type: ignore[attr-defined]
    store._release_pinned_buffers = MethodType(lambda self: None, store)  # type: ignore[attr-defined]
    store._flush_pending_deletes = MethodType(lambda self, *, force: None, store)  # type: ignore[attr-defined]
    store._probe_remove_support = MethodType(lambda self: False, store)  # type: ignore[attr-defined]
    store._lookup_existing_ids = MethodType(
        lambda self, ids: np.empty(0, dtype=np.int64), store
    )  # type: ignore[attr-defined]
    store._update_gpu_metrics = MethodType(lambda self: None, store)  # type: ignore[attr-defined]
    store._maybe_refresh_snapshot = MethodType(
        lambda self, *, writes_delta, reason: None, store
    )  # type: ignore[attr-defined]
    store._dirty_deletes = 0  # type: ignore[attr-defined]
    store._needs_rebuild = False  # type: ignore[attr-defined]
    store._supports_remove_ids = False  # type: ignore[attr-defined]
    store._search_coalescer = None  # type: ignore[attr-defined]

    class _Index:
        def __init__(self) -> None:
            self.index = self
            self.ntotal = 0
            self.is_trained = True
            self.add_calls: list[tuple[np.ndarray, np.ndarray]] = []

        def add_with_ids(self, matrix: np.ndarray, ids: np.ndarray) -> None:
            self.add_calls.append((matrix.copy(), ids.copy()))

    fake_index = _Index()
    store._index = fake_index  # type: ignore[attr-defined]

    # Ensure FAISS helpers exist even when the optional dependency is absent.
    if store_module.faiss is None:
        monkeypatch.setattr(store_module, "faiss", SimpleNamespace(), raising=False)

    normalize_calls: list[np.ndarray] = []

    def fake_normalize(matrix: np.ndarray) -> None:
        copied = matrix.copy()
        norms = np.linalg.norm(copied, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        copied /= norms
        matrix[:] = copied
        normalize_calls.append(copied)

    monkeypatch.setattr(store_module.faiss, "normalize_L2", fake_normalize, raising=False)
    monkeypatch.setattr(store_module.faiss, "downcast_index", lambda index: index, raising=False)

    original_coerce = FaissVectorStore._coerce_batch
    normalize_flags: list[bool] = []

    def recording_coerce(self: FaissVectorStore, xb: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        normalize_flags.append(bool(normalize))
        return original_coerce(self, xb, normalize=normalize)

    store._coerce_batch = MethodType(recording_coerce, store)  # type: ignore[attr-defined]

    vectors = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    ids = ["00000000-0000-0000-0000-000000000001"]

    store.add(vectors, ids)

    assert normalize_flags == [False]
    assert len(normalize_calls) == 1
    assert fake_index.add_calls, "Index ingestion should occur"
    stored_matrix, stored_ids = fake_index.add_calls[0]
    np.testing.assert_allclose(np.linalg.norm(stored_matrix, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(
        stored_ids, np.array([store_module._vector_uuid_to_faiss_int(ids[0])])
    )


def test_search_batch_impl_normalizes_once(monkeypatch: "pytest.MonkeyPatch") -> None:
    """``_search_batch_impl`` should bypass redundant normalisation steps."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 2  # type: ignore[attr-defined]
    store._observability = SimpleNamespace(  # type: ignore[attr-defined]
        trace=lambda *a, **k: _NullContext(),
        metrics=SimpleNamespace(increment=lambda *a, **k: None, observe=lambda *a, **k: None),
    )
    store._as_pinned = MethodType(lambda self, matrix: matrix, store)  # type: ignore[attr-defined]
    store._release_pinned_buffers = MethodType(lambda self: None, store)  # type: ignore[attr-defined]

    captured_search_inputs: list[np.ndarray] = []

    def fake_search_matrix(self: FaissVectorStore, matrix: np.ndarray, top_k: int):
        captured_search_inputs.append(matrix.copy())
        distances = np.full((matrix.shape[0], top_k), 0.5, dtype=np.float32)
        indices = np.arange(top_k, dtype=np.int64)
        tiled_indices = np.tile(indices, (matrix.shape[0], 1))
        return distances, tiled_indices

    def fake_resolve(
        self: FaissVectorStore, distances: np.ndarray, indices: np.ndarray
    ) -> list[list[store_module.FaissSearchResult]]:
        results: list[list[store_module.FaissSearchResult]] = []
        for row_scores, row_ids in zip(distances, indices):
            hits = [
                store_module.FaissSearchResult(vector_id=f"id-{int(idx)}", score=float(score))
                for score, idx in zip(row_scores, row_ids)
            ]
            results.append(hits)
        return results

    store._search_matrix = MethodType(fake_search_matrix, store)  # type: ignore[attr-defined]
    store._resolve_search_results = MethodType(fake_resolve, store)  # type: ignore[attr-defined]

    if store_module.faiss is None:
        monkeypatch.setattr(store_module, "faiss", SimpleNamespace(), raising=False)

    normalize_calls: list[np.ndarray] = []

    def fake_normalize(matrix: np.ndarray) -> None:
        copied = matrix.copy()
        norms = np.linalg.norm(copied, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        copied /= norms
        matrix[:] = copied
        normalize_calls.append(copied)

    monkeypatch.setattr(store_module.faiss, "normalize_L2", fake_normalize, raising=False)

    original_coerce = FaissVectorStore._coerce_batch
    normalize_flags: list[bool] = []

    def recording_coerce(self: FaissVectorStore, xb: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        normalize_flags.append(bool(normalize))
        return original_coerce(self, xb, normalize=normalize)

    store._coerce_batch = MethodType(recording_coerce, store)  # type: ignore[attr-defined]

    queries = np.array([[1.0, 0.0], [0.1, 0.2]], dtype=np.float32)

    results = store._search_batch_impl(queries, top_k=2)

    assert normalize_flags == [False]
    assert len(normalize_calls) == 1
    assert captured_search_inputs, "FAISS search should be invoked"
    np.testing.assert_allclose(
        np.linalg.norm(captured_search_inputs[0], axis=1),
        np.ones(queries.shape[0]),
        rtol=1e-6,
    )
    assert results and all(results)


def test_coerce_batch_normalize_flag_benchmark(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Skipping in-method normalisation should yield a measurable CPU win."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 8  # type: ignore[attr-defined]

    def slow_normalize(matrix: np.ndarray) -> np.ndarray:
        time.sleep(0.0002)
        return matrix

    monkeypatch.setattr(store_module, "normalize_rows", slow_normalize)

    batch = np.random.rand(256, store._dim).astype(np.float32)

    start = time.perf_counter()
    for _ in range(5):
        store._coerce_batch(batch, normalize=True)
    normalized_duration = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(5):
        store._coerce_batch(batch, normalize=False)
    bypass_duration = time.perf_counter() - start

    assert bypass_duration < normalized_duration * 0.5


def test_set_nprobe_initializes_gpu_parameter_space(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Ensure GPU parameter tuning initializes replica shards before configuration."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = DenseIndexConfig(index_type="ivf_flat", nprobe=11)

    class RecordingShard:
        def __init__(self, label: str) -> None:
            self.label = label
            self.nprobe: int | None = None

    shards = [RecordingShard(label) for label in ("a", "b", "c")]

    class RecordingReplicaIndex:
        def __init__(self, shard_list: list[RecordingShard]) -> None:
            object.__setattr__(self, "replicas", shard_list)
            object.__setattr__(self, "shards", shard_list)
            object.__setattr__(self, "index", None)
            object.__setattr__(self, "_fallback_assignments", 0)
            object.__setattr__(self, "_nprobe", "unset")

        @property
        def nprobe(self) -> object:
            return object.__getattribute__(self, "_nprobe")

        @nprobe.setter
        def nprobe(self, value: object) -> None:
            object.__setattr__(
                self,
                "_fallback_assignments",
                object.__getattribute__(self, "_fallback_assignments") + 1,
            )
            object.__setattr__(self, "_nprobe", value)

        @property
        def fallback_assignments(self) -> int:
            return object.__getattribute__(self, "_fallback_assignments")

    replica_index = RecordingReplicaIndex(shards)
    store._index = replica_index  # type: ignore[attr-defined]

    set_calls: list[tuple[object, str, int]] = []

    class FakeGpuParameterSpace:
        def __init__(self) -> None:
            self.initialized: object | None = None

        def initialize(self, index: object) -> None:
            self.initialized = index

        def set_index_parameter(self, index: object, name: str, value: int) -> None:
            assert self.initialized is index, "GpuParameterSpace.initialize must be invoked first"
            set_calls.append((index, name, value))
            seen: set[object] = set()
            for attr in ("replicas", "shards"):
                collection = getattr(index, attr, None)
                if not collection:
                    continue
                for shard in collection:
                    if shard in seen:
                        continue
                    shard.nprobe = value
                    seen.add(shard)

    fake_faiss = SimpleNamespace(
        GpuParameterSpace=FakeGpuParameterSpace,
        describe_index=lambda index: "recording-index",
    )
    monkeypatch.setattr(store_module, "faiss", fake_faiss, raising=False)

    store._log_index_configuration = MethodType(lambda self, _: None, store)  # type: ignore[attr-defined]

    store._set_nprobe()

    assert set_calls == [(replica_index, "nprobe", store._config.nprobe)]  # type: ignore[attr-defined]
    assert [shard.nprobe for shard in shards] == [store._config.nprobe] * len(shards)
    assert replica_index.fallback_assignments == 0


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
