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
        or not hasattr(faiss, "index_cpu_to_all_gpus")
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
