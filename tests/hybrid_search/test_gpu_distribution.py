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
def test_distribute_to_all_gpus_configures_cloner_options(monkeypatch, shard: bool) -> None:
    """Ensure the GPU distributor forwards supported arguments only."""

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._replication_enabled = True
    store._replicated = False
    store._config = DenseIndexConfig()
    store._has_explicit_replication_ids = False
    store._replication_gpu_ids = None
    store._multi_gpu_mode = "shard" if shard else "replicate"
    store._observability = Observability()

    cpu_index = faiss.IndexFlatIP(4)

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 2, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", lambda idx: idx, raising=False)

    captured: dict[str, object] = {}

    def fake_index_cpu_to_all_gpus(index_arg, co=None, ngpu: int = 0):  # type: ignore[override]
        captured["co"] = co
        captured["ngpu"] = ngpu
        return index_arg

    monkeypatch.setattr(faiss, "index_cpu_to_all_gpus", fake_index_cpu_to_all_gpus)

    replicated = store.distribute_to_all_gpus(cpu_index, shard=shard)

    assert replicated is cpu_index
    assert store._replicated is True
    assert isinstance(captured["co"], faiss.GpuMultipleClonerOptions)
    assert captured["ngpu"] == 2
    cloner = captured["co"]
    assert cloner.shard is shard
    if shard and hasattr(cloner, "common_ivf_quantizer"):
        assert cloner.common_ivf_quantizer is True


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

    replicated = store.distribute_to_all_gpus(cpu_index, shard=False)

    assert replicated is cpu_index
    assert store._replicated is False
    assert any("Insufficient GPU targets" in record.getMessage() for record in caplog.records)
