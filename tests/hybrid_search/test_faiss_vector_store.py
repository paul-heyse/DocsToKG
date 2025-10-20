"""Regression suite for `FaissVectorStore` on the custom FAISS GPU wheel.

Grounds the guarantees documented in `faiss-gpu-wheel-reference.md` and the
README by stress-testing concurrency guards, cuVS detection, retry logic,
metrics wiring, and vector normalisation. Ensures dense ingestion remains
thread-safe, resource-aware (CUDA 12 + OpenBLAS), and resilient to loader
quirks exposed by the `faiss-1.12.0` wheel. Additional coverage exercises
concurrency guards, vector normalisation, retry loops, background threads, and
observability counters so the dense store remains robust under load.
"""

from __future__ import annotations

import logging
import time
from threading import Event, RLock, Thread
from types import MethodType, SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pytest

from DocsToKG.HybridSearch import store as store_module
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.pipeline import Observability
from DocsToKG.HybridSearch.store import FaissVectorStore
from tests.conftest import PatchManager


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


def test_resolve_cuvs_state_handles_missing_should_use(patcher: "PatchManager") -> None:
    """cuVS availability should be reported as unavailable when the probe is missing."""

    patcher.setattr(store_module, "_FAISS_AVAILABLE", True, raising=False)
    stub_faiss = SimpleNamespace(knn_gpu=object())
    patcher.setattr(store_module, "faiss", stub_faiss, raising=False)

    enabled, available, reported = store_module.resolve_cuvs_state(requested=None)

    assert (enabled, available, reported) == (False, False, None)


def test_resolve_cuvs_state_overrides_requested_when_probe_false(
    patcher: "PatchManager",
) -> None:
    """A `False` cuVS probe must disable cuVS even when requested."""

    patcher.setattr(store_module, "_FAISS_AVAILABLE", True, raising=False)
    stub_faiss = SimpleNamespace(knn_gpu=object(), should_use_cuvs=lambda *_args, **_kwargs: False)
    patcher.setattr(store_module, "faiss", stub_faiss, raising=False)

    enabled, available, reported = store_module.resolve_cuvs_state(requested=True)

    assert not enabled
    assert not available
    assert reported is False


def test_apply_use_cuvs_parameter_never_sets_true_when_probe_false(
    patcher: "PatchManager",
) -> None:
    """The cuVS flag must not be applied when FAISS rejects the probe."""

    patcher.setattr(store_module, "_FAISS_AVAILABLE", True, raising=False)
    created_spaces: list[object] = []

    class RecordingGpuParameterSpace:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str, bool]] = []
            self._value: Optional[bool] = None
            created_spaces.append(self)

        def initialize(self, _index: object) -> None:
            return None

        def set_index_parameter(self, index: object, name: str, value: bool) -> None:
            self.calls.append((index, name, bool(value)))
            self._value = bool(value)

        def get_index_parameter(self, _index: object, _name: str) -> Optional[bool]:
            return self._value

    stub_faiss = SimpleNamespace(
        knn_gpu=object(),
        should_use_cuvs=lambda *_args, **_kwargs: False,
        GpuParameterSpace=RecordingGpuParameterSpace,
    )
    patcher.setattr(store_module, "faiss", stub_faiss, raising=False)

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = SimpleNamespace(use_cuvs=True)  # type: ignore[attr-defined]
    patcher.setattr(
        store,
        "_iter_gpu_index_variants",
        MethodType(lambda self, root: [root], store),
        raising=False,
    )

    index = SimpleNamespace()
    store._apply_use_cuvs_parameter(index)

    assert created_spaces, "GpuParameterSpace should have been constructed"
    recorded_calls = created_spaces[0].calls
    assert recorded_calls, "use_cuvs should be set via GpuParameterSpace"
    assert all(call[2] is False for call in recorded_calls)
    assert store._last_applied_cuvs is False


def test_faiss_vector_store_search_batch_preserves_queries(
    patcher: "PatchManager",
) -> None:
    """Ensure ``search_batch`` does not mutate the caller-provided query matrix."""

    def fake_normalize_rows(matrix: np.ndarray) -> np.ndarray:
        matrix += 1.0
        return matrix

    patcher.setattr(store_module, "normalize_rows", fake_normalize_rows)

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


def test_coerce_batch_allows_opt_out_of_normalization(patcher: "PatchManager") -> None:
    """``_coerce_batch`` should only normalise rows when explicitly requested."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 4  # type: ignore[attr-defined]

    original_normalize = store_module.normalize_rows
    calls: list[np.ndarray] = []

    def recording_normalize(matrix: np.ndarray) -> np.ndarray:
        calls.append(matrix.copy())
        return original_normalize(matrix)

    patcher.setattr(store_module, "normalize_rows", recording_normalize)

    if store_module.faiss is None:
        patcher.setattr(store_module, "faiss", SimpleNamespace(), raising=False)

    def fake_normalize(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms

    patcher.setattr(store_module.faiss, "normalize_L2", fake_normalize, raising=False)

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


def test_apply_cloner_reservation_scales_per_gpu(patcher: "PatchManager") -> None:
    """Reserve vectors should scale based on GPU participation in shard mode."""

    class FakeIntVector(list):
        def push_back(self, value: int) -> None:
            self.append(int(value))

    class RecordingLogger:
        def __init__(self) -> None:
            self.events: list[dict[str, object] | None] = []

        def info(self, _message: str, *, extra: dict[str, object] | None = None) -> None:
            self.events.append(extra)

    class RecordingCloner:
        def __init__(self) -> None:
            self.reserveVecs: int | None = None
            self.eachReserveVecs: list[int] | None = None
            self.set_calls: list[list[int]] = []

        def setReserveVecs(self, values: Sequence[int]) -> None:
            self.set_calls.append(list(int(v) for v in values))

    fake_faiss = SimpleNamespace(IntVector=FakeIntVector)
    patcher.setattr(store_module, "faiss", fake_faiss, raising=False)

    def make_store(mode: str, expected: int) -> FaissVectorStore:
        store = FaissVectorStore.__new__(FaissVectorStore)
        store._multi_gpu_mode = mode  # type: ignore[attr-defined]
        store._expected_ntotal = expected  # type: ignore[attr-defined]
        store._observability = SimpleNamespace(logger=RecordingLogger())  # type: ignore[attr-defined]
        return store

    replicate_store = make_store("replicate", expected=120)
    replicate_cloner = RecordingCloner()
    replicate_store._apply_cloner_reservation(replicate_cloner, gpu_ids=[0, 1, 2])
    assert replicate_cloner.reserveVecs == 120
    assert replicate_cloner.eachReserveVecs == [120, 120, 120]
    assert replicate_cloner.set_calls == []

    shard_store = make_store("shard", expected=101)
    shard_cloner = RecordingCloner()
    shard_store._apply_cloner_reservation(shard_cloner, gpu_ids=[0, 1, 2, 3])
    assert shard_cloner.reserveVecs == 26
    assert shard_cloner.eachReserveVecs == [26, 26, 26, 26]

    tiny_store = make_store("shard", expected=2)
    tiny_cloner = RecordingCloner()
    tiny_store._apply_cloner_reservation(tiny_cloner, gpu_ids=[0, 1, 2])
    assert tiny_cloner.reserveVecs == 1
    assert tiny_cloner.eachReserveVecs == [1, 1, 1]


def test_add_calls_faiss_normalize_once(patcher: "PatchManager") -> None:
    """``add`` should rely on FAISS to normalise vectors exactly once per batch."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 3  # type: ignore[attr-defined]
    store._config = SimpleNamespace(ingest_dedupe_threshold=0.0, nlist=1, ivf_train_factor=1)  # type: ignore[attr-defined]
    store._lock = RLock()  # type: ignore[attr-defined]
    store._observability = SimpleNamespace(  # type: ignore[attr-defined]
        trace=lambda *a, **k: _NullContext(),
        metrics=SimpleNamespace(increment=lambda *a, **k: None),
    )
    store._as_pinned = MethodType(lambda self, matrix: matrix, store)  # type: ignore[attr-defined]
    store._release_pinned_buffers = MethodType(lambda self: None, store)  # type: ignore[attr-defined]
    store._flush_pending_deletes = MethodType(lambda self, *, force: None, store)  # type: ignore[attr-defined]
    store._probe_remove_support = MethodType(lambda self: False, store)  # type: ignore[attr-defined]
    store._lookup_existing_ids = MethodType(lambda self, ids: np.empty(0, dtype=np.int64), store)  # type: ignore[attr-defined]
    store._update_gpu_metrics = MethodType(lambda self: None, store)  # type: ignore[attr-defined]
    store._maybe_refresh_snapshot = MethodType(lambda self, *, writes_delta, reason: None, store)  # type: ignore[attr-defined]
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
        patcher.setattr(store_module, "faiss", SimpleNamespace(), raising=False)

    normalize_calls: list[np.ndarray] = []

    def fake_normalize(matrix: np.ndarray) -> None:
        copied = matrix.copy()
        norms = np.linalg.norm(copied, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        copied /= norms
        matrix[:] = copied
        normalize_calls.append(copied)

    patcher.setattr(store_module.faiss, "normalize_L2", fake_normalize, raising=False)
    patcher.setattr(store_module.faiss, "downcast_index", lambda index: index, raising=False)

    original_coerce = FaissVectorStore._coerce_batch
    normalize_flags: list[bool] = []

    def recording_coerce(
        self: FaissVectorStore, xb: np.ndarray, *, normalize: bool = True
    ) -> np.ndarray:
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


@pytest.mark.parametrize("requested, expected", [(True, True), (False, False)])
def test_maybe_distribute_applies_cuvs_flag(
    patcher: "PatchManager", requested: bool, expected: bool
) -> None:
    """GpuParameterSpace should receive ``use_cuvs`` updates for every index variant."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = SimpleNamespace(use_cuvs=requested, index_type="flat", nprobe=1)  # type: ignore[attr-defined]
    store._multi_gpu_mode = "single"  # type: ignore[attr-defined]
    store._replication_enabled = False  # type: ignore[attr-defined]
    store._observability = SimpleNamespace(  # type: ignore[attr-defined]
        logger=SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None),
        metrics=SimpleNamespace(increment=lambda *a, **k: None, set_gauge=lambda *a, **k: None),
    )
    store._last_applied_cuvs = None  # type: ignore[attr-defined]

    class RecordingGpuParameterSpace:
        def __init__(self) -> None:
            self.initialize_calls: list[object] = []
            self.set_calls: list[tuple[object, str, object]] = []
            self._values: dict[int, object] = {}

        def initialize(self, index: object) -> None:
            self.initialize_calls.append(index)

        def set_index_parameter(self, index: object, key: str, value: object) -> None:
            self.set_calls.append((index, key, value))
            self._values[id(index)] = value

        def get_index_parameter(self, index: object, key: str) -> object:
            return self._values.get(id(index), False)

    gps_instances: list[RecordingGpuParameterSpace] = []

    def gps_factory() -> RecordingGpuParameterSpace:
        instance = RecordingGpuParameterSpace()
        gps_instances.append(instance)
        return instance

    fake_faiss = SimpleNamespace(
        GpuParameterSpace=gps_factory,
        downcast_index=lambda index: index,
        describe_index=lambda index: "mock",
    )
    patcher.setattr(store_module, "faiss", fake_faiss, raising=False)

    patcher.setattr(
        store_module,
        "resolve_cuvs_state",
        lambda value: (expected, True, expected),
        raising=False,
    )

    class DummyIndex:
        pass

    gpu_index = DummyIndex()

    result = store._maybe_distribute_multi_gpu(gpu_index)

    assert result is gpu_index
    assert gps_instances, "GpuParameterSpace should be constructed"
    gps = gps_instances[0]
    assert gpu_index in gps.initialize_calls
    assert (gpu_index, "use_cuvs", expected) in gps.set_calls
    assert store._last_applied_cuvs == expected


def test_add_dedupe_rejects_scaled_duplicates(patcher: "PatchManager") -> None:
    """Scaled duplicates should be rejected when the dedupe threshold is strict."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 3  # type: ignore[attr-defined]
    store._config = SimpleNamespace(ingest_dedupe_threshold=0.99, nlist=1, ivf_train_factor=1)  # type: ignore[attr-defined]
    store._observability = SimpleNamespace(  # type: ignore[attr-defined]
        trace=lambda *a, **k: _NullContext(),
        metrics=SimpleNamespace(increment=lambda *a, **k: None),
    )

    metrics_calls: list[tuple[str, float]] = []

    def recording_increment(name: str, amount: float = 1.0) -> None:
        metrics_calls.append((name, float(amount)))

    store._observability.metrics.increment = recording_increment  # type: ignore[attr-defined]

    class RecordingIndex:
        def __init__(self) -> None:
            self.ntotal = 1
            self.add_calls: list[tuple[np.ndarray, np.ndarray]] = []

        def add_with_ids(
            self, matrix: np.ndarray, ids: np.ndarray
        ) -> None:  # pragma: no cover - sanity guard
            self.add_calls.append((matrix.copy(), ids.copy()))

    store._index = RecordingIndex()  # type: ignore[attr-defined]

    search_inputs: list[np.ndarray] = []

    def fake_search_matrix(self: FaissVectorStore, matrix: np.ndarray, top_k: int):
        search_inputs.append(matrix.copy())
        distances = np.array([[0.9995]], dtype=np.float32)
        indices = np.array([[123]], dtype=np.int64)
        return distances, indices

    store._search_matrix = MethodType(fake_search_matrix, store)  # type: ignore[attr-defined]

    normalize_calls: list[np.ndarray] = []

    def fake_normalize(matrix: np.ndarray) -> None:
        copied = matrix.copy()
        norms = np.linalg.norm(copied, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        copied /= norms
        matrix[:] = copied
        normalize_calls.append(copied)

    patcher.setattr(
        store_module,
        "faiss",
        SimpleNamespace(normalize_L2=fake_normalize),
        raising=False,
    )

    original_vector = np.array([10.0, -4.0, 0.5], dtype=np.float32)
    payload = original_vector.copy()
    vector_id = "00000000-0000-0000-0000-000000000099"

    store.add([payload], [vector_id])

    assert not store._index.add_calls  # type: ignore[attr-defined]
    assert metrics_calls == [("faiss_ingest_deduped", 1.0)]
    assert normalize_calls, "dedupe path should normalise the copied queries"
    assert search_inputs, "dedupe path should perform a search"
    np.testing.assert_allclose(np.linalg.norm(search_inputs[0], axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(payload, original_vector)


def test_search_batch_impl_normalizes_once(patcher: "PatchManager") -> None:
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
        patcher.setattr(store_module, "faiss", SimpleNamespace(), raising=False)

    normalize_calls: list[np.ndarray] = []

    def fake_normalize(matrix: np.ndarray) -> None:
        copied = matrix.copy()
        norms = np.linalg.norm(copied, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        copied /= norms
        matrix[:] = copied
        normalize_calls.append(copied)

    patcher.setattr(store_module.faiss, "normalize_L2", fake_normalize, raising=False)

    original_coerce = FaissVectorStore._coerce_batch
    normalize_flags: list[bool] = []

    def recording_coerce(
        self: FaissVectorStore, xb: np.ndarray, *, normalize: bool = True
    ) -> np.ndarray:
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


def test_coerce_batch_normalize_flag_benchmark(patcher: "PatchManager") -> None:
    """Skipping in-method normalisation should yield a measurable CPU win."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._dim = 8  # type: ignore[attr-defined]

    def slow_normalize(matrix: np.ndarray) -> np.ndarray:
        time.sleep(0.0002)
        return matrix

    patcher.setattr(store_module, "normalize_rows", slow_normalize)

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


def test_set_nprobe_initializes_gpu_parameter_space(patcher: "PatchManager") -> None:
    """Ensure GPU parameter tuning initializes replica shards before configuration."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = DenseIndexConfig(index_type="ivf_flat", nprobe=11)
    store._last_applied_nprobe = None
    store._last_applied_nprobe_monotonic = 0.0

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
                    if name == "nprobe":
                        shard.nprobe = value
                    seen.add(shard)

    fake_faiss = SimpleNamespace(
        GpuParameterSpace=FakeGpuParameterSpace,
        describe_index=lambda index: "recording-index",
    )
    patcher.setattr(store_module, "faiss", fake_faiss, raising=False)

    store._log_index_configuration = MethodType(lambda self, _: None, store)  # type: ignore[attr-defined]

    store._set_nprobe()

    expected_pairs = [
        (replica_index, "nprobe", store._config.nprobe),  # type: ignore[attr-defined]
        (replica_index, "use_cuvs", False),
    ]
    assert set_calls == expected_pairs
    assert [shard.nprobe for shard in shards] == [store._config.nprobe] * len(shards)
    assert replica_index.fallback_assignments == 0


def test_set_nprobe_short_circuits_when_value_unchanged(
    patcher: "PatchManager",
) -> None:
    """``_set_nprobe`` should avoid redundant FAISS parameter updates."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = DenseIndexConfig(index_type="ivf_flat", nprobe=7)
    store._index = SimpleNamespace()
    store._log_index_configuration = MethodType(lambda self, _: None, store)  # type: ignore[attr-defined]

    set_calls: list[tuple[object, str, int]] = []

    class FakeGpuParameterSpace:
        def __init__(self) -> None:
            self._initialised = False

        def initialize(self, index: object) -> None:
            self._initialised = True

        def set_index_parameter(self, index: object, name: str, value: int) -> None:
            assert self._initialised, "GpuParameterSpace.initialize must run before set"
            set_calls.append((index, name, value))

    fake_faiss = SimpleNamespace(
        GpuParameterSpace=FakeGpuParameterSpace,
        describe_index=lambda index: "recording-index",
    )
    patcher.setattr(store_module, "faiss", fake_faiss, raising=False)

    store._reset_nprobe_cache()  # type: ignore[attr-defined]
    store._set_nprobe()  # type: ignore[attr-defined]
    first_timestamp = store._last_applied_nprobe_monotonic  # type: ignore[attr-defined]

    initial_expected = [
        (store._index, "nprobe", store._config.nprobe),  # type: ignore[attr-defined]
        (store._index, "use_cuvs", False),
    ]
    assert set_calls == initial_expected
    assert store._last_applied_nprobe == store._config.nprobe  # type: ignore[attr-defined]
    assert first_timestamp > 0.0

    store._set_nprobe()  # type: ignore[attr-defined]
    assert set_calls == initial_expected
    assert store._last_applied_nprobe_monotonic == first_timestamp  # type: ignore[attr-defined]

    store._reset_nprobe_cache()  # type: ignore[attr-defined]
    store._set_nprobe()  # type: ignore[attr-defined]
    repeat_expected = [
        (store._index, "nprobe", store._config.nprobe),  # type: ignore[attr-defined]
        (store._index, "use_cuvs", False),
        (store._index, "nprobe", store._config.nprobe),
        (store._index, "use_cuvs", False),
    ]
    assert set_calls == repeat_expected
    assert store._last_applied_nprobe_monotonic > first_timestamp  # type: ignore[attr-defined]


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
                assert self._allow_search_finish.wait(timeout=1.0), "search wait timed out"
            return []

    store = DummyStore()
    ids = np.array([1, 2], dtype=np.int64)
    remove_counts: list[int] = []

    def run_remove() -> None:
        remove_counts.append(store.remove_ids(ids, force_flush=True, reason="test_atomic_remove"))

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


@pytest.mark.parametrize("use_all_devices", [False, True])
def test_init_gpu_configures_resource_knobs(patcher, caplog, use_all_devices: bool) -> None:
    """Ensure GPU init applies DenseIndexConfig resource tuning on single GPU."""

    temp_memory = 8 << 20
    config = DenseIndexConfig(
        device=0,
        gpu_temp_memory_bytes=temp_memory,
    )

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = config  # type: ignore[attr-defined]
    store._dim = 16  # type: ignore[attr-defined]
    store._multi_gpu_mode = "single"  # type: ignore[attr-defined]
    store._indices_32_bit = True  # type: ignore[attr-defined]
    store._temp_memory_bytes = temp_memory  # type: ignore[attr-defined]
    store._gpu_use_default_null_stream_all_devices = use_all_devices  # type: ignore[attr-defined]
    store._gpu_use_default_null_stream = True  # type: ignore[attr-defined]
    store._expected_ntotal = 0  # type: ignore[attr-defined]
    store._rebuild_delete_threshold = 10000  # type: ignore[attr-defined]
    store._force_64bit_ids = False  # type: ignore[attr-defined]
    store._force_remove_ids_fallback = False  # type: ignore[attr-defined]
    store._replication_enabled = False  # type: ignore[attr-defined]
    store._replication_gpu_ids = None  # type: ignore[attr-defined]
    store._replica_gpu_resources = []  # type: ignore[attr-defined]
    store._pinned_buffers = []  # type: ignore[attr-defined]
    store._observability = Observability()  # type: ignore[attr-defined]
    store._gpu_resources = None  # type: ignore[attr-defined]

    class RecordingResource:
        def __init__(self) -> None:
            self.temp_memory_calls: list[int] = []
            self.null_stream_calls: list[object | None] = []
            self.null_stream_all_calls: int = 0

        def setTempMemory(self, value: int) -> None:
            self.temp_memory_calls.append(value)

        def setDefaultNullStreamAllDevices(self) -> None:  # pragma: no cover - defensive
            self.null_stream_all_calls += 1

        def setDefaultNullStream(
            self, device: int | None = None
        ) -> None:  # pragma: no cover - defensive
            self.null_stream_calls.append(device)

    fake_faiss = SimpleNamespace(
        StandardGpuResources=RecordingResource,
        get_num_gpus=lambda: 1,
    )
    patcher.setattr(store_module, "faiss", fake_faiss, raising=False)

    caplog.set_level(logging.INFO, logger="DocsToKG.HybridSearch")

    store.init_gpu()

    resource = store._gpu_resources  # type: ignore[attr-defined]
    assert isinstance(resource, RecordingResource)
    assert resource.temp_memory_calls == [temp_memory]
    if use_all_devices:
        assert resource.null_stream_all_calls == 1
        assert resource.null_stream_calls == []
    else:
        assert resource.null_stream_all_calls == 0
        assert resource.null_stream_calls == [config.device]

    gauges = {sample.name: sample.value for sample in store._observability.metrics.export_gauges()}
    assert gauges.get("faiss_gpu_temp_memory_bytes") == float(temp_memory)
    assert gauges.get("faiss_gpu_default_null_stream") == 1.0
    expected_all_devices = 1.0 if use_all_devices else 0.0
    assert gauges.get("faiss_gpu_default_null_stream_all_devices") == expected_all_devices

    records = [
        record
        for record in caplog.records
        if record.getMessage() == "faiss-gpu-resource-configured"
    ]
    assert records, "resource configuration log event missing"
    payload = getattr(records[-1], "event", {})
    assert payload.get("temp_memory_bytes") == temp_memory
    assert payload.get("default_null_stream_all_devices") is use_all_devices


def test_maybe_to_gpu_applies_expected_reserve_vecs(patcher: "PatchManager") -> None:
    """``_maybe_to_gpu`` should propagate expected reservations to cloner options."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._observability = Observability()  # type: ignore[attr-defined]
    store._expected_ntotal = 256  # type: ignore[attr-defined]
    store._force_64bit_ids = False  # type: ignore[attr-defined]
    store._indices_32_bit = True  # type: ignore[attr-defined]
    store._gpu_resources = object()  # type: ignore[attr-defined]
    store._multi_gpu_mode = "single"  # type: ignore[attr-defined]
    store._replication_enabled = False  # type: ignore[attr-defined]
    store._maybe_distribute_multi_gpu = MethodType(lambda self, idx: idx, store)  # type: ignore[attr-defined]
    store.init_gpu = MethodType(lambda self: None, store)  # type: ignore[attr-defined]
    store._config = SimpleNamespace(device=1, flat_use_fp16=False)  # type: ignore[attr-defined]

    captured: dict[str, object] = {}

    class RecordingCloner:
        def __init__(self) -> None:
            self.device = None
            self.verbose = False
            self.allowCpuCoarseQuantizer = True
            self.indicesOptions = None
            self.reserveVecs = None

    def fake_index_cpu_to_gpu(resources, device, index, co):  # type: ignore[override]
        captured["co"] = co
        captured["device"] = device
        captured["resources"] = resources
        captured["index"] = index
        return f"gpu-{device}"

    fake_faiss = SimpleNamespace(
        GpuClonerOptions=RecordingCloner,
        index_cpu_to_gpu=fake_index_cpu_to_gpu,
        INDICES_32_BIT=13,
    )

    patcher.setattr(store_module, "faiss", fake_faiss, raising=False)

    result = store._maybe_to_gpu(object())

    assert result == "gpu-1"
    assert captured["resources"] is store._gpu_resources
    assert captured["device"] == 1
    cloner = captured["co"]
    assert isinstance(cloner, RecordingCloner)
    assert cloner.device == 1
    assert cloner.reserveVecs == store._expected_ntotal


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
