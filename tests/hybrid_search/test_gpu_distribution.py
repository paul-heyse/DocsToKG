"""Tests for GPU replication helpers in :mod:`DocsToKG.HybridSearch.store`."""

from __future__ import annotations

import pytest

from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.pipeline import Observability
from DocsToKG.HybridSearch.store import FaissVectorStore

try:  # pragma: no cover - optional dependency in CI
    import faiss  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing wheel
    faiss = None  # type: ignore


@pytest.mark.skipif(faiss is None, reason="faiss not installed")
@pytest.mark.skipif(
    faiss is not None
    and (
        not hasattr(faiss, "GpuMultipleClonerOptions")
        or not hasattr(faiss, "GpuResourcesVector")
        or not (
            hasattr(faiss, "index_cpu_to_gpu_multiple")
            or hasattr(faiss, "index_cpu_to_gpus_list")
        )
    ),
    reason="faiss build does not expose GPU replication helpers",
)
@pytest.mark.parametrize("shard", [False, True])
def test_distribute_to_all_gpus_configures_cloner_options(monkeypatch, shard: bool) -> None:
    """Ensure the GPU distributor forwards supported arguments only."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    temp_memory = 8_192
    store._config = DenseIndexConfig(
        device=0,
        multi_gpu_mode="shard" if shard else "replicate",
        gpu_temp_memory_bytes=temp_memory,
    )
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

    def fake_index_cpu_to_gpu_multiple(
        resources_vector, gpu_ids, index_arg, co=None
    ):  # type: ignore[override]
        captured["co"] = co
        captured["gpu_ids"] = list(gpu_ids)
        captured["resources_vector"] = resources_vector
        return index_arg

    monkeypatch.setattr(faiss, "index_cpu_to_gpu_multiple", fake_index_cpu_to_gpu_multiple)
    monkeypatch.setattr(
        faiss,
        "index_cpu_to_all_gpus",
        lambda *args, **kwargs: pytest.fail("legacy replication path invoked"),
    )

    replicated = store.distribute_to_all_gpus(cpu_index, shard=shard)

    assert replicated is cpu_index
    assert store._replicated is True
    assert isinstance(captured["co"], faiss.GpuMultipleClonerOptions)
    assert captured["gpu_ids"] == [0, 1]
    vector = captured["resources_vector"]
    assert isinstance(vector, RecordingVector)
    assert len(vector.resources) == 2
    for resource in vector.resources:
        assert resource.temp_memory_calls == [temp_memory]
    assert store._replica_gpu_resources == [vector.resources[1]]
    cloner = captured["co"]
    assert cloner.shard is shard
    if shard and hasattr(cloner, "common_ivf_quantizer"):
        assert cloner.common_ivf_quantizer is True
