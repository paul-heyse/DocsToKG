import os
import uuid
from typing import Callable

import numpy as np
import pytest

from DocsToKG.HybridSearch.config import DenseIndexConfig, FusionConfig
from DocsToKG.HybridSearch.dense import FaissIndexManager
from DocsToKG.HybridSearch.ids import vector_uuid_to_faiss_int
from DocsToKG.HybridSearch.results import ResultShaper
from DocsToKG.HybridSearch.similarity import cosine_against_corpus_gpu
from DocsToKG.HybridSearch.storage import OpenSearchSimulator
from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload, HybridSearchRequest

faiss = pytest.importorskip("faiss")  # type: ignore

if not hasattr(faiss, "get_num_gpus"):
    pytestmark = pytest.mark.skip(reason="FAISS GPU utilities not available in this build")
else:
    pytestmark = pytest.mark.skipif(
        faiss.get_num_gpus() < 1, reason="FAISS GPU device required"
    )


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
    base = manager._index
    if hasattr(base, "index"):
        base = base.index
    if hasattr(faiss, "downcast_index"):
        try:
            base = faiss.downcast_index(base)
        except Exception:
            pass
    assert "Gpu" in type(base).__name__, f"Expected GPU index type, got {type(base)}"
    assert stats.get("gpu_base") is True
    assert float(stats.get("gpu_remove_fallbacks", 0.0)) == 0.0


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


def test_gpu_clone_strict_coarse_quantizer() -> None:
    cfg = DenseIndexConfig(index_type="flat", device=_target_device())
    manager = FaissIndexManager(dim=32, config=cfg)
    cpu_index = faiss.IndexFlatIP(32)
    mapped = faiss.IndexIDMap2(cpu_index)
    gpu_index = manager._maybe_to_gpu(mapped)
    base = gpu_index.index if hasattr(gpu_index, "index") else gpu_index
    if hasattr(faiss, "downcast_index"):
        base = faiss.downcast_index(base)
    assert "Gpu" in type(base).__name__, "Expected GPU index after strict cloning"


def test_gpu_near_duplicate_detection_filters_duplicates() -> None:
    embedding = np.ones(16, dtype=np.float32)
    features = ChunkFeatures({}, {}, embedding)
    chunk_a = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-1",
        vector_id="vec-1",
        namespace="default",
        text="Hybrid search test chunk",
        metadata={},
        features=features,
        token_count=4,
        source_chunk_idxs=[0],
        doc_items_refs=[],
    )
    chunk_b = ChunkPayload(
        doc_id="doc-1",
        chunk_id="chunk-2",
        vector_id="vec-2",
        namespace="default",
        text="Hybrid search test chunk",
        metadata={},
        features=ChunkFeatures({}, {}, embedding.copy()),
        token_count=4,
        source_chunk_idxs=[1],
        doc_items_refs=[],
    )
    opensearch = OpenSearchSimulator()
    opensearch.bulk_upsert([chunk_a, chunk_b])
    shaper = ResultShaper(
        opensearch,
        FusionConfig(cosine_dedupe_threshold=0.9),
        device=_target_device(),
        resources=faiss.StandardGpuResources(),
    )
    request = HybridSearchRequest(query="hybrid search", namespace=None, filters={}, page_size=5)
    fused_scores = {chunk_a.vector_id: 1.0, chunk_b.vector_id: 0.95}
    channel_scores = {"dense": fused_scores}
    results = shaper.shape([chunk_a, chunk_b], fused_scores, request, channel_scores)
    assert len(results) == 1, "GPU near-duplicate detection should filter duplicates"


def test_gpu_nprobe_applied_during_search() -> None:
    xb, xq = _toy_data(n=512, d=64)
    cfg = DenseIndexConfig(index_type="ivf_flat", nlist=64, nprobe=32, device=_target_device())
    manager = FaissIndexManager(dim=xb.shape[1], config=cfg)
    vectors, vector_ids = _emit_vectors(xb)
    manager.set_id_resolver(_make_id_resolver(vector_ids))
    manager.add(vectors, vector_ids)
    manager.search(xq[0], top_k=5)
    base = manager._index.index if hasattr(manager._index, "index") else manager._index
    if hasattr(faiss, "downcast_index"):
        base = faiss.downcast_index(base)
    assert hasattr(base, "nprobe")
    assert int(base.nprobe) == cfg.nprobe


def test_gpu_similarity_uses_supplied_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DenseIndexConfig(index_type="flat", device=_target_device())
    manager = FaissIndexManager(dim=32, config=cfg)

    captured: dict[str, object] = {}

    def fake_pairwise(resources, A, B, metric, device):  # type: ignore[no-untyped-def]
        captured["resources"] = resources
        captured["device"] = device
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)

    monkeypatch.setattr(faiss, "pairwise_distance_gpu", fake_pairwise)

    q = np.ones(32, dtype=np.float32)
    corpus = np.ones((3, 32), dtype=np.float32)
    cosine_against_corpus_gpu(q, corpus, device=manager.device, resources=manager.gpu_resources)

    assert captured.get("device") == manager.device
    assert captured.get("resources") is manager.gpu_resources
