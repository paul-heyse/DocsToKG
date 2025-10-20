"""Targeted tests for cuVS capability detection across HybridSearch helpers."""

from __future__ import annotations

import logging
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from DocsToKG.HybridSearch import store as store_module
from DocsToKG.HybridSearch.store import FaissVectorStore
from tests.conftest import PatchManager


class _NullContext:
    def __enter__(self) -> None:  # pragma: no cover - trivial context manager
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial context manager
        return False


@pytest.mark.parametrize("reported", [True, False])
def test_cuvs_detection_propagates_probe_results(
    patcher: PatchManager, reported: bool
) -> None:
    """Ensure cuVS discovery honours FAISS's `should_use_cuvs` verdict."""

    captured: dict[str, object] = {"configs": []}

    def normalize_L2(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms

    def knn_gpu(_resources, queries, corpus, k, **kwargs):
        captured["knn_params"] = kwargs.get("params")
        sims = queries.astype(np.float32) @ corpus.astype(np.float32).T
        order = np.argsort(sims, axis=1)[:, ::-1][:, :k]
        scores = np.take_along_axis(sims, order, axis=1)
        return scores.astype(np.float32), order.astype(np.int64)

    gps_instances: list[object] = []

    class RecordingGpuParameterSpace:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str, object]] = []
            self._value: object | None = None
            gps_instances.append(self)

        def initialize(self, _index: object) -> None:
            return None

        def set_index_parameter(self, index: object, name: str, value: object) -> None:
            self.calls.append((index, name, value))
            self._value = value

        def get_index_parameter(self, _index: object, _name: str) -> object | None:
            return self._value

    def should_use_cuvs(config: object | None = None) -> bool:
        captured.setdefault("configs", []).append(config)
        return reported

    stub_faiss = SimpleNamespace(
        knn_gpu=knn_gpu,
        should_use_cuvs=should_use_cuvs,
        GpuDistanceParams=lambda: SimpleNamespace(
            xType=None,
            yType=None,
            use_cuvs=None,
            metricType=None,
        ),
        DistanceDataType_F16="f16",
        GpuIndexFlatConfig=lambda: SimpleNamespace(
            device=None,
            d=None,
            dims=None,
            dim=None,
            useFloat16=None,
            useFloat16LookupTables=None,
            metricType=None,
        ),
        GpuParameterSpace=RecordingGpuParameterSpace,
        normalize_L2=normalize_L2,
        METRIC_INNER_PRODUCT=1,
    )

    patcher.setattr(store_module, "_FAISS_AVAILABLE", True, raising=False)
    patcher.setattr(store_module, "faiss", stub_faiss, raising=False)

    enabled, available, reported_available = store_module.resolve_cuvs_state(requested=None)

    assert enabled is reported
    assert available is reported
    assert reported_available is reported
    assert captured["configs"], "should_use_cuvs should receive a probe config"
    assert captured["configs"][-1] is not None

    q = np.random.default_rng(0).random((2, 4), dtype=np.float32)
    c = np.random.default_rng(1).random((8, 4), dtype=np.float32)

    scores, indices = store_module.cosine_topk_blockwise(
        q,
        c,
        k=2,
        device=0,
        resources=object(),
        use_fp16=True,
    )

    assert scores.shape == (2, 2)
    assert indices.shape == (2, 2)
    params = captured.get("knn_params")
    assert params is not None, "cosine_topk_blockwise should pass params to knn_gpu"
    assert getattr(params, "use_cuvs", None) is reported

    store = FaissVectorStore.__new__(FaissVectorStore)
    store._config = SimpleNamespace(  # type: ignore[attr-defined]
        use_cuvs=True,
        flat_use_fp16=True,
        index_type="flat",
        nprobe=1,
        device=0,
    )
    store._multi_gpu_mode = "single"  # type: ignore[attr-defined]
    store._replicated = False  # type: ignore[attr-defined]
    store._gpu_resources = None  # type: ignore[attr-defined]
    store._last_applied_cuvs = None  # type: ignore[attr-defined]
    store._dim = 4  # type: ignore[attr-defined]
    store._observability = SimpleNamespace(  # type: ignore[attr-defined]
        trace=lambda *_args, **_kwargs: _NullContext(),
        metrics=SimpleNamespace(increment=lambda *_a, **_k: None, set_gauge=lambda *_a, **_k: None),
        logger=logging.getLogger(__name__),
    )
    store._index = SimpleNamespace(ntotal=0)  # type: ignore[attr-defined]
    patcher.setattr(
        store,
        "_iter_gpu_index_variants",
        MethodType(lambda self, root: [root], store),
        raising=False,
    )
    patcher.setattr(
        FaissVectorStore,
        "_describe_index",
        lambda self, _index: "stub-index",
        raising=False,
    )
    patcher.setattr(
        store,
        "_current_nprobe_value",
        MethodType(lambda self: 1, store),
        raising=False,
    )

    stats = store.adapter_stats
    assert stats.cuvs_enabled is reported
    assert stats.cuvs_available is reported

    index = SimpleNamespace()
    store._apply_use_cuvs_parameter(index)

    assert gps_instances, "GpuParameterSpace should be instantiated"
    recorded = gps_instances[0].calls
    assert recorded, "use_cuvs should be propagated to the FAISS index"
    assert bool(recorded[0][2]) is reported
    assert store._last_applied_cuvs is reported
*** End of File
