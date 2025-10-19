"""Regression tests for GPU cosine similarity helpers."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from DocsToKG.HybridSearch import store as store_module
from DocsToKG.HybridSearch.store import cosine_batch, cosine_topk_blockwise


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_cosine_batch_normalizes_copies_and_reuses_contiguous_buffers(shape):
    faiss = pytest.importorskip("faiss", reason="FAISS runtime is required for cosine_batch")

    q = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    C = np.linspace(0.1, 0.9, num=6, dtype=np.float32).reshape(2, 3)

    q_before = q.copy()
    C_before = C.copy()
    captured: dict[str, np.ndarray] = {}

    def _tracking_pairwise(resources, queries, corpus, *, metric, device):
        captured["queries"] = queries
        captured["corpus"] = corpus
        return queries @ corpus.T

    result = cosine_batch(
        q,
        C,
        device=0,
        resources=object(),
        pairwise_fn=_tracking_pairwise,
    )

    assert result.shape == (1, 2) if len(shape) == 1 else (shape[0], 2)

    normalized_q = np.asarray(q_before, dtype=np.float32, copy=True)
    q_view = normalized_q.reshape(1, -1) if normalized_q.ndim == 1 else normalized_q
    normalized_C = np.asarray(C_before, dtype=np.float32, copy=True)
    faiss.normalize_L2(q_view)
    faiss.normalize_L2(normalized_C)
    expected = q_view @ normalized_C.T

    np.testing.assert_allclose(result, expected)
    # Verify that input arrays are not modified (cosine_batch copies internally)
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(C, C_before)
    # Verify that captured arrays are normalized versions
    np.testing.assert_allclose(captured["queries"], q_view)
    np.testing.assert_allclose(captured["corpus"], normalized_C)


def _make_faiss_stub(
    *, enable_knn: bool, with_params: bool = False, require_params: bool = False
) -> tuple[SimpleNamespace, dict[str, object]]:
    state: dict[str, object] = {"pairwise_calls": 0}

    def normalize_L2(arr: np.ndarray) -> None:
        view = np.asarray(arr, dtype=np.float32)
        if view.ndim == 1:
            norm = np.linalg.norm(view)
            if norm:
                view /= norm
            return
        norms = np.linalg.norm(view, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        view /= norms

    def pairwise_distance_gpu(
        resources,
        queries,
        corpus,
        *,
        metric,
        device,
        params=None,
    ):
        state["pairwise_calls"] = state.get("pairwise_calls", 0) + 1
        state["pairwise_params"] = params
        state["pairwise_queries_dtype"] = getattr(queries, "dtype", None)
        state["pairwise_corpus_dtype"] = getattr(corpus, "dtype", None)
        if require_params and with_params and params is None:
            raise TypeError("params argument is required for this stub")
        return queries.astype(np.float32) @ corpus.astype(np.float32).T

    def knn_gpu(resources, queries, corpus, k, **kwargs):
        state["kwargs"] = kwargs
        state["knn_params"] = kwargs.get("params")
        state["knn_queries_dtype"] = getattr(queries, "dtype", None)
        state["knn_corpus_dtype"] = getattr(corpus, "dtype", None)
        if require_params and with_params and kwargs.get("params") is None:
            raise TypeError("params argument is required for this stub")
        sims = queries.astype(np.float32) @ corpus.astype(np.float32).T
        order = np.argsort(sims, axis=1)[:, ::-1][:, :k]
        scores = np.take_along_axis(sims, order, axis=1)
        return scores.astype(np.float32), order.astype(np.int64)

    if enable_knn:

        def pairwise_disabled(*_args, **_kwargs):  # pragma: no cover - guard rail
            raise AssertionError("pairwise path disabled")

        pairwise_distance = pairwise_disabled  # type: ignore[assignment]
    else:
        pairwise_distance = pairwise_distance_gpu

    namespace_kwargs = dict(
        METRIC_INNER_PRODUCT=1,
        normalize_L2=normalize_L2,
        pairwise_distance_gpu=pairwise_distance,
    )
    if with_params:
        namespace_kwargs.update(
            {
                "DistanceDataType_F16": "f16",
                "GpuDistanceParams": lambda: SimpleNamespace(
                    xType=None,
                    yType=None,
                    use_cuvs=None,
                ),
            }
        )
    stub = SimpleNamespace(**namespace_kwargs)
    if enable_knn:
        stub.knn_gpu = knn_gpu  # type: ignore[attr-defined]
    return stub, state


def test_cosine_topk_blockwise_prefers_knn_gpu(monkeypatch):
    stub, state = _make_faiss_stub(enable_knn=True)
    monkeypatch.setattr(store_module, "faiss", stub, raising=False)

    q = np.random.default_rng(0).random((3, 4), dtype=np.float32)
    C = np.random.default_rng(1).random((128, 4), dtype=np.float32)
    block_rows = 32

    def forbid_concat(*_args, **_kwargs):  # pragma: no cover - sanity guard
        raise AssertionError("np.concatenate should not run in knn path")

    monkeypatch.setattr(store_module.np, "concatenate", forbid_concat)

    scores, indices = cosine_topk_blockwise(
        q,
        C,
        k=5,
        device=0,
        resources=object(),
        block_rows=block_rows,
    )

    q_norm = q.astype(np.float32, copy=True)
    C_norm = C.astype(np.float32, copy=True)
    stub.normalize_L2(q_norm)
    stub.normalize_L2(C_norm)
    expected = q_norm @ C_norm.T
    expected_idx = np.argsort(expected, axis=1)[:, ::-1][:, :5]
    expected_scores = np.take_along_axis(expected, expected_idx, axis=1)

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_array_equal(indices, expected_idx)

    kwargs = state.get("kwargs")
    assert isinstance(kwargs, dict)
    vector_limit = block_rows * C.shape[1] * np.dtype(np.float32).itemsize
    query_limit = max(block_rows, q.shape[0]) * q.shape[1] * np.dtype(np.float32).itemsize
    assert kwargs["vectorsMemoryLimit"] == vector_limit
    assert kwargs["queriesMemoryLimit"] == query_limit
    assert "use_cuvs" not in kwargs


def test_cosine_topk_blockwise_falls_back_without_knn(monkeypatch):
    stub, state = _make_faiss_stub(enable_knn=False)
    monkeypatch.setattr(store_module, "faiss", stub, raising=False)

    q = np.random.default_rng(2).random((2, 3), dtype=np.float32)
    C = np.random.default_rng(3).random((17, 3), dtype=np.float32)

    scores, indices = cosine_topk_blockwise(
        q,
        C,
        k=4,
        device=0,
        resources=object(),
        use_fp16=True,
    )

    q_norm = q.astype(np.float32, copy=True)
    C_norm = C.astype(np.float32, copy=True)
    stub.normalize_L2(q_norm)
    stub.normalize_L2(C_norm)
    q_fp16 = q_norm.astype(np.float16)
    C_fp16 = C_norm.astype(np.float16)
    expected = q_fp16.astype(np.float32) @ C_fp16.astype(np.float32).T
    expected_idx = np.argsort(expected, axis=1)[:, ::-1][:, :4]
    expected_scores = np.take_along_axis(expected, expected_idx, axis=1)

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_array_equal(indices, expected_idx)
    assert scores.dtype == np.float32
    assert indices.dtype == np.int64
    assert state["pairwise_calls"]


def test_cosine_topk_blockwise_honors_cuvs_request(monkeypatch):
    faiss = pytest.importorskip("faiss", reason="cuVS integration test requires faiss GPU runtime")
    if not hasattr(faiss, "knn_gpu") or not hasattr(faiss, "StandardGpuResources"):
        pytest.skip("faiss GPU helpers unavailable")
    get_num_gpus = getattr(faiss, "get_num_gpus", None)
    if callable(get_num_gpus) and int(get_num_gpus()) <= 0:
        pytest.skip("no GPUs available for cuVS test")
    should_use = getattr(faiss, "should_use_cuvs", None)
    if not callable(should_use):
        pytest.skip("faiss.should_use_cuvs unavailable")
    try:
        if not bool(should_use()):
            pytest.skip("cuVS runtime not available")
    except Exception as exc:
        pytest.skip(f"faiss.should_use_cuvs failed: {exc}")
    try:
        resources = faiss.StandardGpuResources()
    except Exception as exc:
        pytest.skip(f"unable to allocate GPU resources: {exc}")

    q = np.random.default_rng(4).random((2, 8), dtype=np.float32)
    C = np.random.default_rng(5).random((32, 8), dtype=np.float32)

    original_knn = faiss.knn_gpu
    captured: dict[str, object] = {}

    def wrapped_knn(*args, **kwargs):
        captured["kwargs"] = kwargs
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(store_module.faiss, "knn_gpu", wrapped_knn, raising=False)
    try:
        cosine_topk_blockwise(
            q,
            C,
            k=3,
            device=0,
            resources=resources,
            use_cuvs=True,
        )
    finally:
        monkeypatch.setattr(store_module.faiss, "knn_gpu", original_knn, raising=False)

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("use_cuvs") is True


def test_cosine_topk_blockwise_fp16_knn_uses_distance_params(monkeypatch):
    stub, state = _make_faiss_stub(
        enable_knn=True,
        with_params=True,
        require_params=True,
    )
    monkeypatch.setattr(store_module, "faiss", stub, raising=False)
    monkeypatch.setattr(
        store_module,
        "resolve_cuvs_state",
        lambda requested: (True, True, True),
        raising=False,
    )

    q = np.random.default_rng(6).random((2, 4), dtype=np.float32)
    C = np.random.default_rng(7).random((16, 4), dtype=np.float32)

    scores, indices = cosine_topk_blockwise(
        q,
        C,
        k=3,
        device=0,
        resources=object(),
        use_fp16=True,
        use_cuvs=True,
    )

    q_norm = q.astype(np.float32, copy=True)
    C_norm = C.astype(np.float32, copy=True)
    stub.normalize_L2(q_norm)
    stub.normalize_L2(C_norm)
    expected = q_norm @ C_norm.T
    expected_idx = np.argsort(expected, axis=1)[:, ::-1][:, :3]
    expected_scores = np.take_along_axis(expected, expected_idx, axis=1)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(indices, expected_idx)
    assert state.get("pairwise_calls", 0) == 0
    params = state.get("knn_params")
    assert params is not None
    assert params.xType == stub.DistanceDataType_F16
    assert params.yType == stub.DistanceDataType_F16
    assert getattr(params, "use_cuvs", None) is True
    assert state.get("knn_queries_dtype") == np.float16
    assert state.get("knn_corpus_dtype") == np.float16
    assert scores.dtype == np.float32
    assert indices.dtype == np.int64


def test_cosine_topk_blockwise_fp16_pairwise_uses_distance_params(monkeypatch):
    stub, state = _make_faiss_stub(
        enable_knn=False,
        with_params=True,
        require_params=True,
    )
    monkeypatch.setattr(store_module, "faiss", stub, raising=False)
    monkeypatch.setattr(
        store_module,
        "resolve_cuvs_state",
        lambda requested: (True, True, True),
        raising=False,
    )

    q = np.random.default_rng(8).random((3, 5), dtype=np.float32)
    C = np.random.default_rng(9).random((21, 5), dtype=np.float32)

    scores, indices = cosine_topk_blockwise(
        q,
        C,
        k=4,
        device=0,
        resources=object(),
        block_rows=10,
        use_fp16=True,
        use_cuvs=True,
    )

    q_norm = q.astype(np.float32, copy=True)
    C_norm = C.astype(np.float32, copy=True)
    stub.normalize_L2(q_norm)
    stub.normalize_L2(C_norm)
    expected = q_norm @ C_norm.T
    expected_idx = np.argsort(expected, axis=1)[:, ::-1][:, :4]
    expected_scores = np.take_along_axis(expected, expected_idx, axis=1)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(indices, expected_idx)

    assert state.get("pairwise_calls", 0) > 0
    params = state.get("pairwise_params")
    assert params is not None
    assert params.xType == stub.DistanceDataType_F16
    assert params.yType == stub.DistanceDataType_F16
    assert getattr(params, "use_cuvs", None) is True
    assert state.get("pairwise_queries_dtype") == np.float16
    assert state.get("pairwise_corpus_dtype") == np.float16
    assert scores.dtype == np.float32
    assert indices.dtype == np.int64
