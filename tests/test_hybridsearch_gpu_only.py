import os
import uuid
from typing import Callable

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # type: ignore

pytestmark = pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="FAISS GPU device required")

from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.ids import vector_uuid_to_faiss_int
from DocsToKG.HybridSearch.similarity import cosine_against_corpus_gpu


def _toy_data(n: int = 2048, d: int = 128) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((64, d), dtype=np.float32)
    return xb, xq


def _target_device() -> int:
    return int(os.getenv("HYBRIDSEARCH_FAISS_DEVICE", "0"))


def _make_id_resolver(vector_ids: list[str]) -> Callable[[int], str | None]:
    bridge = {vector_uuid_to_faiss_int(vid): vid for vid in vector_ids}
    return bridge.get


def _emit_vectors(xb: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    vectors = [row.copy() for row in xb]
    vector_ids = [str(uuid.uuid4()) for _ in vectors]
    return vectors, vector_ids


def _assert_gpu_index(manager: FaissIndexManager) -> None:
    stats = manager.stats()
    device = stats.get("device")
    assert device not in (None, "*"), f"Expected GPU device assignment, stats={stats}"
    assert int(device) == _target_device(), f"Index promoted to unexpected device: {stats}"


def test_gpu_flat_end_to_end() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(index_type="flat", nprobe=1, device=_target_device())
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


def test_gpu_ivf_flat_build_and_search() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(index_type="ivf_flat", nlist=256, nprobe=8, device=_target_device())
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


def test_gpu_ivfpq_build_and_search() -> None:
    xb, xq = _toy_data()
    cfg = DenseIndexConfig(
        index_type="ivf_pq",
        nlist=256,
        nprobe=8,
        pq_m=16,
        pq_bits=8,
        device=_target_device(),
    )
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    results = manager.search(xq[0], top_k=5)
    assert len(results) == 5
    _assert_gpu_index(manager)


def test_gpu_cosine_against_corpus() -> None:
    xb, xq = _toy_data(n=512)
    query = xq[0]
    sims = cosine_against_corpus_gpu(query, xb, device=_target_device())
    assert sims.shape == (1, xb.shape[0])
    self_sim = float(cosine_against_corpus_gpu(query, query.reshape(1, -1))[0, 0])
    assert 0.98 <= self_sim <= 1.001
