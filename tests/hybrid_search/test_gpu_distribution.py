"""Tests for GPU replication helpers in :mod:`DocsToKG.HybridSearch.store`."""

from __future__ import annotations

import logging

import pytest

from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.pipeline import Observability
from DocsToKG.HybridSearch.store import FaissVectorStore

try:  # pragma: no cover - optional dependency in CI
    import faiss  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing wheel
    faiss = None  # type: ignore


temp_memory = 1 << 20


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
@pytest.mark.skipif(
    faiss is not None
    and (
        not hasattr(faiss, "GpuMultipleClonerOptions")
        or not hasattr(faiss, "index_cpu_to_all_gpus")
        or not hasattr(faiss, "index_cpu_to_gpus_list")
    ),
    reason="faiss build does not expose GPU replication helpers",
)
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("force_legacy_path", [False, True])
def test_distribute_to_all_gpus_configures_cloner_options(
    monkeypatch, caplog, shard: bool, force_legacy_path: bool
) -> None:
    """Ensure the GPU distributor forwards supported arguments only."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig()
    store._has_explicit_replication_ids = False
    store._replication_gpu_ids = None
    store._multi_gpu_mode = "shard" if shard else "replicate"
    store._observability = Observability()
    store._temp_memory_bytes = temp_memory
    store._gpu_use_default_null_stream = False
    store._gpu_use_default_null_stream_all_devices = False
    store._replica_gpu_resources = []

    cpu_index = faiss.IndexFlatIP(4)

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 2, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    captured: dict[str, object] = {}
    caplog.set_level(logging.INFO, logger="DocsToKG.HybridSearch")

    class RecordingResources:
        def __init__(self) -> None:
            self.temp_memory_calls: list[int] = []
            self.null_stream_calls: list[tuple[str, object | None]] = []

        def setTempMemory(self, value: int) -> None:
            self.temp_memory_calls.append(value)

        def setDefaultNullStreamAllDevices(self) -> None:  # pragma: no cover - stubbed
            self.null_stream_calls.append(("all", None))

        def setDefaultNullStream(self, device: int | None = None) -> None:  # pragma: no cover - stubbed
            self.null_stream_calls.append(("device", device))

    class RecordingVector:
        def __init__(self) -> None:
            self.resources: list[RecordingResources] = []

        def push_back(self, resource: RecordingResources) -> None:
            self.resources.append(resource)

    monkeypatch.setattr(faiss, "StandardGpuResources", RecordingResources)
    monkeypatch.setattr(faiss, "GpuResourcesVector", RecordingVector)

    store._gpu_resources = faiss.StandardGpuResources()

    class DummyClonerOptions:
        def __init__(self) -> None:
            self.shard = False
            self.common_ivf_quantizer = False

    monkeypatch.setattr(faiss, "GpuMultipleClonerOptions", DummyClonerOptions, raising=False)

    if force_legacy_path:
        monkeypatch.delattr(faiss, "index_cpu_to_gpu_multiple", raising=False)

        def fake_index_cpu_to_gpus_list(
            index_arg, *, gpus=None, co=None, resources=None
        ):  # type: ignore[override]
            captured["co"] = co
            captured["gpus"] = list(gpus or [])
            captured["resources_vector"] = resources
            return index_arg

        monkeypatch.setattr(
            faiss,
            "index_cpu_to_gpus_list",
            fake_index_cpu_to_gpus_list,
            raising=False,
        )
        monkeypatch.setattr(
            faiss,
            "index_cpu_to_all_gpus",
            lambda *args, **kwargs: pytest.fail("manual resource fallback should be used"),
        )
    else:

        def fake_index_cpu_to_gpu_multiple(
            resources_vector, gpu_ids, index_arg, co=None
        ):  # type: ignore[override]
            captured["co"] = co
            captured["gpu_ids"] = list(gpu_ids)
            captured["resources_vector"] = resources_vector
            return index_arg

        monkeypatch.setattr(
            faiss, "index_cpu_to_gpu_multiple", fake_index_cpu_to_gpu_multiple
        )
        monkeypatch.setattr(
            faiss,
            "index_cpu_to_gpus_list",
            lambda *args, **kwargs: pytest.fail("gpu_multiple path should be used"),
            raising=False,
        )
        monkeypatch.setattr(
            faiss,
            "index_cpu_to_all_gpus",
            lambda *args, **kwargs: pytest.fail("legacy replication path invoked"),
        )

    replicated = store.distribute_to_all_gpus(cpu_index, shard=shard)

    assert replicated is cpu_index
    assert store._replicated is True
    assert isinstance(captured["co"], faiss.GpuMultipleClonerOptions)
    vector = captured["resources_vector"]
    assert isinstance(vector, RecordingVector)
    expected_gpu_ids = [0, 1]
    assert len(vector.resources) == len(expected_gpu_ids)
    for resource in vector.resources:
        assert resource.temp_memory_calls == [temp_memory]
    assert store._replica_gpu_resources == [vector.resources[1]]
    cloner = captured["co"]
    assert cloner.shard is shard
    if shard and hasattr(cloner, "common_ivf_quantizer"):
        assert cloner.common_ivf_quantizer is True

    counters = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for sample in store._observability.metrics.export_counters()
    }
    if force_legacy_path:
        assert captured["gpus"] == expected_gpu_ids
        assert counters.get(("faiss_gpu_manual_resource_path", ())) == 1.0
        assert any(
            record.getMessage() == "faiss-manual-resource-path-engaged"
            for record in caplog.records
        )
    else:
        assert captured["gpu_ids"] == expected_gpu_ids
        assert ("faiss_gpu_manual_resource_path", ()) not in counters


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
@pytest.mark.skipif(
    faiss is not None
    and (
        not hasattr(faiss, "GpuMultipleClonerOptions")
        or not hasattr(faiss, "index_cpu_to_gpu_multiple")
        or not hasattr(faiss, "GpuResourcesVector")
    ),
    reason="faiss build missing multi-GPU replication helpers",
)
def test_distribute_to_all_gpus_uses_non_contiguous_ids(monkeypatch) -> None:
    """Ensure explicit GPU ids are preserved when constructing resources."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig(replication_gpu_ids=(1, 3))
    store._has_explicit_replication_ids = True
    store._replication_gpu_ids = (1, 3)
    store._multi_gpu_mode = "replicate"
    store._observability = Observability()
    store._temp_memory_bytes = temp_memory
    store._gpu_use_default_null_stream = False
    store._gpu_use_default_null_stream_all_devices = False
    store._replica_gpu_resources = []
    store._gpu_resources = None

    cpu_index = object()

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 4, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    configured_devices: list[int | None] = []

    def recording_configure(self, resource, *, device=None):
        configured_devices.append(device)
        if hasattr(resource, "setTempMemory"):
            resource.setTempMemory(temp_memory)

    monkeypatch.setattr(
        FaissVectorStore, "_configure_gpu_resource", recording_configure, raising=False
    )

    class RecordingResources:
        def __init__(self) -> None:
            self.temp_memory_calls: list[int] = []

        def setTempMemory(self, value: int) -> None:
            self.temp_memory_calls.append(value)

        def setDefaultNullStreamAllDevices(self) -> None:  # pragma: no cover - stubbed
            return None

        def setDefaultNullStream(self, device: int | None = None) -> None:  # pragma: no cover
            return None

    class RecordingVector:
        def __init__(self) -> None:
            self.resources: list[RecordingResources] = []

        def push_back(self, resource: RecordingResources) -> None:
            self.resources.append(resource)

    monkeypatch.setattr(faiss, "StandardGpuResources", RecordingResources)
    monkeypatch.setattr(faiss, "GpuResourcesVector", RecordingVector)

    class DummyClonerOptions:
        def __init__(self) -> None:
            self.shard = False
            self.common_ivf_quantizer = False

    monkeypatch.setattr(faiss, "GpuMultipleClonerOptions", DummyClonerOptions, raising=False)

    captured: dict[str, object] = {}

    def fake_index_cpu_to_gpu_multiple(resources_vector, gpu_ids, index_arg, co=None):
        captured["gpu_ids"] = list(gpu_ids)
        captured["resources_vector"] = resources_vector
        captured["co"] = co
        return index_arg

    monkeypatch.setattr(faiss, "index_cpu_to_gpu_multiple", fake_index_cpu_to_gpu_multiple)
    monkeypatch.setattr(
        faiss,
        "index_cpu_to_all_gpus",
        lambda *args, **kwargs: pytest.fail("index_cpu_to_all_gpus should not run"),
    )

    replicated = store.distribute_to_all_gpus(cpu_index, shard=False)

    assert replicated is cpu_index
    assert store._replicated is True
    assert captured["gpu_ids"] == [1, 3]
    vector = captured["resources_vector"]
    assert isinstance(vector, RecordingVector)
    assert len(vector.resources) == 2
    assert configured_devices == [1, 3]
    assert store._replica_gpu_resources == vector.resources
    for resource in vector.resources:
        assert resource.temp_memory_calls == [temp_memory]
    cloner = captured["co"]
    assert isinstance(cloner, DummyClonerOptions)
    assert cloner.shard is False


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_distribute_to_all_gpus_manual_path_without_resources(monkeypatch) -> None:
    """Manual replication without explicit resources must honour requested GPUs."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig(replication_gpu_ids=(1, 3))
    store._has_explicit_replication_ids = True
    store._replication_gpu_ids = (1, 3)
    store._multi_gpu_mode = "replicate"
    store._observability = Observability()
    store._replica_gpu_resources = []
    store._gpu_resources = None
    store._temp_memory_bytes = None
    store._gpu_use_default_null_stream = False
    store._gpu_use_default_null_stream_all_devices = False

    cpu_index = object()

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 4, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    if hasattr(faiss, "GpuResourcesVector"):
        monkeypatch.delattr(faiss, "GpuResourcesVector", raising=False)

    captured_gpus: list[int] = []
    sentinel_index = object()

    def fake_index_cpu_to_gpus_list(index_arg, *, gpus=None, co=None):  # type: ignore[override]
        captured_gpus.extend(list(gpus or []))
        return sentinel_index

    monkeypatch.setattr(
        faiss, "index_cpu_to_gpus_list", fake_index_cpu_to_gpus_list, raising=False
    )
    monkeypatch.setattr(
        faiss,
        "index_cpu_to_all_gpus",
        lambda *args, **kwargs: pytest.fail("index_cpu_to_all_gpus should not run"),
        raising=False,
    )

    replicated = store.distribute_to_all_gpus(cpu_index, shard=False)

    assert replicated is sentinel_index
    assert store._replicated is True
    assert captured_gpus == [1, 3]
    counters = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for sample in store._observability.metrics.export_counters()
    }
    assert counters.get(("faiss_gpu_manual_resource_path", ())) == 1.0
    assert store._replica_gpu_resources == []


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
@pytest.mark.parametrize("shard", [False, True])
def test_distribute_to_all_gpus_respects_explicit_gpu_list(monkeypatch, shard: bool) -> None:
    """Explicit replication ids should use ``index_cpu_to_gpus_list`` with filtering."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig(replication_gpu_ids=(0, 2, 5))
    store._has_explicit_replication_ids = True
    store._replication_gpu_ids = (0, 2, 5)
    store._multi_gpu_mode = "shard" if shard else "replicate"
    store._observability = Observability()

    cpu_index = object()

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 3, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    captured: dict[str, object] = {}

    def fake_index_cpu_to_gpus_list(index_arg, *, gpus=None, co=None):  # type: ignore[override]
        captured["gpus"] = list(gpus or [])
        captured["co"] = co
        return index_arg

    called_all = False

    def fake_index_cpu_to_all_gpus(index_arg, **kwargs):  # type: ignore[override]
        nonlocal called_all
        called_all = True
        return index_arg

    class DummyCloner:
        def __init__(self) -> None:
            self.shard = False
            self.common_ivf_quantizer = False

    monkeypatch.setattr(faiss, "GpuMultipleClonerOptions", DummyCloner, raising=False)
    monkeypatch.setattr(
        faiss, "index_cpu_to_gpus_list", fake_index_cpu_to_gpus_list, raising=False
    )
    monkeypatch.setattr(
        faiss, "index_cpu_to_all_gpus", fake_index_cpu_to_all_gpus, raising=False
    )

    replicated = store.distribute_to_all_gpus(cpu_index, shard=shard)

    assert replicated is cpu_index
    assert store._replicated is True
    assert captured["gpus"] == [0, 2]
    assert called_all is False
    cloner = captured["co"]
    assert isinstance(cloner, DummyCloner)
    assert cloner.shard is shard
    if shard and hasattr(cloner, "common_ivf_quantizer"):
        assert cloner.common_ivf_quantizer is True


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_distribute_to_all_gpus_skips_invalid_targets(monkeypatch, caplog) -> None:
    """Invalid GPU ids should be ignored without invoking replication helpers."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig(replication_gpu_ids=(5,))
    store._has_explicit_replication_ids = True
    store._replication_gpu_ids = (5,)
    store._multi_gpu_mode = "replicate"
    store._observability = Observability()

    cpu_index = object()

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 2, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    def fail_index_cpu_to_gpus_list(*args, **kwargs):  # type: ignore[override]
        pytest.fail("index_cpu_to_gpus_list should not be called when no valid GPUs remain")

    def fail_index_cpu_to_all_gpus(*args, **kwargs):  # type: ignore[override]
        pytest.fail("index_cpu_to_all_gpus should not be called for explicit GPU ids")

    monkeypatch.setattr(faiss, "index_cpu_to_gpus_list", fail_index_cpu_to_gpus_list, raising=False)
    monkeypatch.setattr(faiss, "index_cpu_to_all_gpus", fail_index_cpu_to_all_gpus, raising=False)

    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="DocsToKG.HybridSearch")

    replicated = store.distribute_to_all_gpus(cpu_index, shard=False)

    assert replicated is cpu_index
    assert store._replicated is False

    counters = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for sample in store._observability.metrics.export_counters()
    }
    assert counters.get(
        (
            "faiss_gpu_explicit_target_unavailable",
            (("reason", "insufficient_filtered_targets"),),
        )
    ) == 1.0

    assert any(
        record.getMessage() == "faiss-explicit-gpu-targets-partially-unavailable"
        for record in caplog.records
        if record.name == "DocsToKG.HybridSearch"
    )
    assert any("Insufficient GPU targets" in record.getMessage() for record in caplog.records)


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
@pytest.mark.parametrize(
    "replication_ids, available, expected_count, expect_fallback, expected_selected",
    [
        (None, 3, 3, True, []),
        ((0, 2, 7), 4, 2, False, [0, 2]),
    ],
)
def test_distribute_to_all_gpus_fallback_uses_gpu_count(
    monkeypatch,
    replication_ids: tuple[int, ...] | None,
    available: int,
    expected_count: int,
    expect_fallback: bool,
    expected_selected: list[int],
) -> None:
    """Fallback replication should respect the resolved GPU ids/count."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = (
        DenseIndexConfig(replication_gpu_ids=replication_ids)
        if replication_ids is not None
        else DenseIndexConfig()
    )
    store._has_explicit_replication_ids = replication_ids is not None
    store._replication_gpu_ids = tuple(replication_ids) if replication_ids is not None else None
    store._multi_gpu_mode = "replicate"
    store._observability = Observability()
    store._replica_gpu_resources = []
    store._gpu_resources = None
    store._temp_memory_bytes = None
    store._gpu_use_default_null_stream = False
    store._gpu_use_default_null_stream_all_devices = False
    store._device = 0

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: available, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    if hasattr(faiss, "GpuResourcesVector"):
        monkeypatch.delattr(faiss, "GpuResourcesVector", raising=False)

    fallback_calls: list[dict[str, object]] = []

    def fake_index_cpu_to_all_gpus(index_arg, *, co=None, ngpu=None):  # type: ignore[override]
        fallback_calls.append({"ngpu": ngpu, "co": co})
        return index_arg

    monkeypatch.setattr(faiss, "index_cpu_to_all_gpus", fake_index_cpu_to_all_gpus, raising=False)

    captured_gpus: list[int] = []

    if replication_ids is not None:

        def fake_index_cpu_to_gpus_list(index_arg, *, gpus=None, co=None):  # type: ignore[override]
            captured_gpus.extend(list(gpus or []))
            return index_arg

        monkeypatch.setattr(
            faiss,
            "index_cpu_to_gpus_list",
            fake_index_cpu_to_gpus_list,
            raising=False,
        )

    cpu_index = object()

    replicated = store.distribute_to_all_gpus(cpu_index, shard=False)

    assert replicated is cpu_index
    assert store._replicated is True

    if expect_fallback:
        assert fallback_calls, "Expected fallback replication to invoke index_cpu_to_all_gpus"
        assert fallback_calls[-1]["ngpu"] == expected_count
        assert captured_gpus == []
    else:
        assert fallback_calls == []
        assert captured_gpus == expected_selected


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_distribute_to_all_gpus_reports_missing_manual_helper(monkeypatch, caplog) -> None:
    """Explicit GPU targets should emit telemetry when helpers are unavailable."""

    if not hasattr(faiss, "index_cpu_to_gpus_list"):
        pytest.skip("faiss build already missing index_cpu_to_gpus_list")

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig(replication_gpu_ids=(0, 1))
    store._has_explicit_replication_ids = True
    store._replication_gpu_ids = (0, 1)
    store._multi_gpu_mode = "replicate"
    store._observability = Observability()

    cpu_index = object()

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 2, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    caplog.set_level(logging.WARNING)
    caplog.set_level(logging.WARNING, logger="DocsToKG.HybridSearch")

    monkeypatch.delattr(faiss, "index_cpu_to_gpus_list", raising=False)

    replicated = store.distribute_to_all_gpus(cpu_index, shard=False)

    assert replicated is cpu_index
    assert store._replicated is False

    counters = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for sample in store._observability.metrics.export_counters()
    }
    assert counters.get(
        (
            "faiss_gpu_explicit_target_unavailable",
            (("reason", "missing_index_cpu_to_gpus_list"),),
        )
    ) == 1.0

    assert any(
        record.getMessage() == "faiss-explicit-gpu-targets-unavailable"
        for record in caplog.records
        if record.name == "DocsToKG.HybridSearch"
    )


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
def test_distribute_to_all_gpus_manual_path_without_resources_vector(monkeypatch, caplog) -> None:
    """Manual GPU replication should respect ids even without GpuResourcesVector."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig(replication_gpu_ids=(1, 3))
    store._has_explicit_replication_ids = True
    store._replication_gpu_ids = (1, 3)
    store._multi_gpu_mode = "replicate"
    store._observability = Observability()
    store._temp_memory_bytes = temp_memory
    store._gpu_use_default_null_stream = False
    store._gpu_use_default_null_stream_all_devices = False
    store._replica_gpu_resources = []
    store._gpu_resources = None

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 4, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    if hasattr(faiss, "GpuResourcesVector"):
        monkeypatch.delattr(faiss, "GpuResourcesVector", raising=False)

    caplog.set_level(logging.INFO, logger="DocsToKG.HybridSearch")

    captured_gpus: list[int] = []
    sentinel = object()

    def fake_index_cpu_to_gpus_list(index_arg, *, gpus=None, co=None):  # type: ignore[override]
        captured_gpus.extend(list(gpus or []))
        return sentinel

    monkeypatch.setattr(
        faiss,
        "index_cpu_to_gpus_list",
        fake_index_cpu_to_gpus_list,
        raising=False,
    )
    monkeypatch.setattr(
        faiss,
        "index_cpu_to_all_gpus",
        lambda *args, **kwargs: pytest.fail("index_cpu_to_all_gpus should not run"),
        raising=False,
    )

    configured_devices: list[int | None] = []
    created_resources: list[object] = []

    class StubResource:
        def __init__(self) -> None:
            self.temp_memory_calls: list[int] = []

        def setTempMemory(self, value: int) -> None:
            self.temp_memory_calls.append(value)

        def setDefaultNullStreamAllDevices(self) -> None:  # pragma: no cover - stubbed
            return None

        def setDefaultNullStream(self, device: int | None = None) -> None:  # pragma: no cover
            return None

    def fake_create(self, *, device=None):
        resource = StubResource()
        created_resources.append(resource)
        return resource

    def fake_configure(self, resource, *, device=None):
        configured_devices.append(device)
        if hasattr(resource, "setTempMemory"):
            resource.setTempMemory(temp_memory)

    monkeypatch.setattr(
        FaissVectorStore,
        "_create_gpu_resources",
        fake_create,
        raising=False,
    )
    monkeypatch.setattr(
        FaissVectorStore,
        "_configure_gpu_resource",
        fake_configure,
        raising=False,
    )

    replicated = store.distribute_to_all_gpus(object(), shard=False)

    assert replicated is sentinel
    assert store._replicated is True
    assert captured_gpus == [1, 3]
    assert configured_devices == [1, 3]
    assert store._replica_gpu_resources == created_resources

    counters = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for sample in store._observability.metrics.export_counters()
    }
    assert counters.get(("faiss_gpu_manual_resource_path", ())) == 1.0
    assert any(
        record.getMessage() == "faiss-manual-resource-path-engaged"
        for record in caplog.records
        if record.name == "DocsToKG.HybridSearch"
    )
