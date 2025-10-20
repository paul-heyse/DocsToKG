"""HybridSearchValidator resource-budget tests against the FAISS GPU wheel.

Confirms validation runs configure `StandardGpuResources` according to config
(temp/pinned memory, null streams) and fall back cleanly when FAISS is absent.
Prevents validation harnesses from over-subscribing CUDA memory while using the
custom CUDA 12/OpenBLAS wheel documented in `faiss-gpu-wheel-reference.md`.
Checks GPU memory limits, pinned memory configuration, null stream handling,
and fallback behaviour when FAISS utilities are unavailable so validation
pipelines do not over-subscribe accelerator resources.
"""

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

import DocsToKG.HybridSearch.service as service_module
from DocsToKG.HybridSearch.config import DenseIndexConfig, RetrievalConfig
from DocsToKG.HybridSearch.pipeline import Observability
from DocsToKG.HybridSearch.service import HybridSearchValidator
from DocsToKG.HybridSearch.store import FaissSearchResult
from DocsToKG.HybridSearch.types import (
    ChunkFeatures,
    ChunkPayload,
    HybridSearchResult,
)


class _RecordingResources:
    def __init__(self) -> None:
        self.temp_memory = None
        self.pinned_memory = None
        self.null_stream_all_devices = False
        self.default_null_stream_calls: list[int | None] = []

    def setTempMemory(self, value: int) -> None:  # noqa: N802 - faiss naming
        self.temp_memory = value

    def setPinnedMemory(self, value: int) -> None:  # noqa: N802 - faiss naming
        self.pinned_memory = value

    def setDefaultNullStreamAllDevices(self) -> None:  # noqa: N802 - faiss naming
        self.null_stream_all_devices = True

    def setDefaultNullStream(self, device: int | None = None) -> None:  # noqa: N802 - faiss naming
        self.default_null_stream_calls.append(device)


@pytest.fixture
def recording_faiss(patcher):
    stub = SimpleNamespace(StandardGpuResources=_RecordingResources)
    patcher.setattr(service_module, "faiss", stub, raising=False)
    return stub


def test_validation_resources_apply_dense_config(caplog, recording_faiss):
    dense_cfg = DenseIndexConfig(
        device=2,
        gpu_temp_memory_bytes=512,
        gpu_pinned_memory_bytes=2048,
        gpu_use_default_null_stream_all_devices=True,
    )
    config = SimpleNamespace(dense=dense_cfg)
    config_manager = SimpleNamespace(get=lambda: config)

    observability = Observability()
    service = SimpleNamespace(
        _config_manager=config_manager,
        _faiss=SimpleNamespace(get_gpu_resources=lambda: None),
        _observability=observability,
    )
    ingestion = SimpleNamespace(faiss_index=SimpleNamespace(get_gpu_resources=lambda: None))

    validator = HybridSearchValidator(
        ingestion=ingestion,
        service=service,
        registry=SimpleNamespace(),
        opensearch=SimpleNamespace(),
    )

    caplog.set_level(logging.INFO, logger=observability.logger.name)

    resource = validator._ensure_validation_resources()
    assert isinstance(resource, _RecordingResources)
    assert resource.temp_memory == 512
    assert resource.pinned_memory == 2048
    assert resource.null_stream_all_devices is True
    assert resource.default_null_stream_calls == []

    gauges = {
        (sample["name"], tuple(sorted(sample.get("labels", {}).items()))): sample["value"]
        for sample in observability.metrics_snapshot()["gauges"]
    }
    assert gauges[("faiss_gpu_temp_memory_bytes", (("scope", "validation"),))] == pytest.approx(
        512.0
    )
    assert gauges[("faiss_gpu_default_null_stream", (("scope", "validation"),))] == pytest.approx(
        0.0
    )
    assert gauges[
        ("faiss_gpu_default_null_stream_all_devices", (("scope", "validation"),))
    ] == pytest.approx(1.0)

    records = [record for record in caplog.records if record.msg == "faiss-gpu-resource-configured"]
    assert records, "expected observability log for validation GPU resource configuration"
    payload = getattr(records[-1], "event", {})
    assert payload.get("scope") == "validation"
    assert payload.get("default_null_stream_all_devices") is True

    # Ensure cached resource is reused
    assert validator._ensure_validation_resources() is resource


def test_request_for_query_respects_recall_first_flag():
    validator = HybridSearchValidator(
        ingestion=SimpleNamespace(),
        service=SimpleNamespace(),
        registry=SimpleNamespace(),
        opensearch=SimpleNamespace(),
    )

    payload = {"query": "example", "recall_first": True}
    request = validator._request_for_query(payload)

    assert request.recall_first is True
    assert request.diagnostics is True

    payload = {"query": "example", "diagnostics": False}
    request = validator._request_for_query(payload)

    assert request.diagnostics is False


def test_calibration_batches_queries_and_preserves_accuracy():
    matches = {
        "vec-0": True,
        "vec-1": False,
        "vec-2": True,
        "vec-3": True,
        "vec-4": False,
    }
    embedding_dim = 4

    class _RecordingFaissIndex:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.last_vector_ids: list[str] = []

        def search_batch(self, queries: np.ndarray, top_k: int):
            assert queries.dtype == np.float32
            assert queries.flags.c_contiguous
            self.calls.append({"shape": tuple(queries.shape), "top_k": top_k})
            results = []
            for vector_id in self.last_vector_ids:
                hits = []
                if matches.get(vector_id, True):
                    hits.append(FaissSearchResult(vector_id=vector_id, score=1.0))
                else:
                    hits.append(FaissSearchResult(vector_id=f"miss-{vector_id}", score=1.0))
                for extra in range(1, top_k):
                    hits.append(
                        FaissSearchResult(
                            vector_id=f"noise-{vector_id}-{extra}", score=0.0
                        )
                    )
                results.append(hits)
            return results

        def search(self, *_args, **_kwargs):  # pragma: no cover - defensive
            raise AssertionError("batch search must be used during calibration")

    class _RecordingRegistry:
        def __init__(self, index: _RecordingFaissIndex) -> None:
            self._chunks = [SimpleNamespace(vector_id=vid) for vid in matches]
            self._embeddings = {
                vid: np.full(embedding_dim, float(idx + 1), dtype=np.float32)
                for idx, vid in enumerate(matches)
            }
            self._dim = embedding_dim
            self._index = index
            self.all_calls = 0

        def all(self):
            self.all_calls += 1
            return list(self._chunks)

        def count(self) -> int:
            return len(self._chunks)

        def resolve_embeddings(self, vector_ids, *, cache=None, dtype=np.float32):
            if not vector_ids:
                return np.empty((0, self._dim), dtype=dtype)
            matrix = np.asarray(
                [self._embeddings[vid] for vid in vector_ids], dtype=np.float32
            )
            if cache is not None:
                for vid, row in zip(vector_ids, matrix, strict=False):
                    cache[vid] = row
            self._index.last_vector_ids = list(vector_ids)
            return matrix

    index = _RecordingFaissIndex()
    registry = _RecordingRegistry(index)
    ingestion = SimpleNamespace(faiss_index=index)
    config = SimpleNamespace(retrieval=RetrievalConfig(dense_calibration_batch_size=2))
    config_manager = SimpleNamespace(get=lambda: config)
    service = SimpleNamespace(_config_manager=config_manager)

    validator = HybridSearchValidator(
        ingestion=ingestion,
        service=service,
        registry=registry,
        opensearch=SimpleNamespace(),
    )

    report = validator._run_calibration([])

    expected_accuracy = sum(1 for flag in matches.values() if flag) / max(1, len(matches))
    dense_results = report.details["dense"]
    assert all(math.isclose(entry["self_hit_accuracy"], expected_accuracy) for entry in dense_results)
    assert registry.all_calls == 1

    batches_per_sweep = math.ceil(len(matches) / 2)
    assert len(index.calls) == len(dense_results) * batches_per_sweep
    expected_topks = []
    for oversample in (1, 2, 3):
        expected_topks.extend([max(1, oversample * 3)] * batches_per_sweep)
    assert [entry["top_k"] for entry in index.calls] == expected_topks
    assert report.passed is False


def test_calibration_reuses_cached_embeddings_across_oversamples():
    vector_ids = [f"vec-{idx}" for idx in range(4)]
    embedding_dim = 3
    embeddings = {
        vector_id: np.full(embedding_dim, float(idx + 1), dtype=np.float32)
        for idx, vector_id in enumerate(vector_ids)
    }

    class _RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def reconstruct_batch(self, vector_ids):
            self.calls.append(tuple(vector_ids))
            rows = [embeddings[vector_id] for vector_id in vector_ids]
            return np.asarray(rows, dtype=np.float32)

    class _RecordingIndex:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.last_vector_ids: list[str] = []

        def search_batch(self, queries: np.ndarray, top_k: int):
            assert queries.dtype == np.float32
            assert queries.flags.c_contiguous
            self.calls.append({"shape": tuple(queries.shape), "top_k": top_k})
            results = []
            for vector_id in self.last_vector_ids:
                hits = [FaissSearchResult(vector_id=vector_id, score=1.0)]
                for extra in range(1, top_k):
                    hits.append(
                        FaissSearchResult(
                            vector_id=f"noise-{vector_id}-{extra}", score=0.0
                        )
                    )
                results.append(hits)
            return results

    class _RecordingRegistry:
        def __init__(self, store: _RecordingStore, index: _RecordingIndex) -> None:
            self._store = store
            self._index = index
            self._chunks = [SimpleNamespace(vector_id=vector_id) for vector_id in vector_ids]

        def all(self):
            return list(self._chunks)

        def resolve_embeddings(self, vector_ids, *, cache=None, dtype=np.float32):
            dtype = np.dtype(dtype)
            results = [None] * len(vector_ids)
            missing = []
            missing_positions = []
            if cache is not None:
                for idx, vector_id in enumerate(vector_ids):
                    cached = cache.get(vector_id)
                    if cached is not None:
                        results[idx] = np.asarray(cached, dtype=np.float32)
                    else:
                        missing.append(vector_id)
                        missing_positions.append(idx)
            else:
                missing = list(vector_ids)
                missing_positions = list(range(len(vector_ids)))
            if missing:
                reconstructed = self._store.reconstruct_batch(missing)
                if reconstructed.shape[0] != len(missing):
                    raise RuntimeError(
                        "Dense store returned mismatched reconstruction rows "
                        f"(expected {len(missing)}, received {reconstructed.shape[0]})"
                    )
                for offset, idx in enumerate(missing_positions):
                    row = np.asarray(reconstructed[offset], dtype=np.float32)
                    results[idx] = row
                    if cache is not None:
                        cache[vector_ids[idx]] = row
            resolved = []
            missing_ids = []
            for idx, candidate in enumerate(results):
                if candidate is None:
                    missing_ids.append(vector_ids[idx])
                else:
                    resolved.append(candidate)
            if missing_ids:
                raise KeyError(f"Embeddings missing for {missing_ids!r}")
            matrix = np.ascontiguousarray(np.stack(resolved), dtype=dtype)
            self._index.last_vector_ids = list(vector_ids)
            return matrix

        def resolve_embedding(self, vector_id: str, *, cache=None, dtype=np.float32):
            matrix = self.resolve_embeddings([vector_id], cache=cache, dtype=dtype)
            return matrix[0]

    store = _RecordingStore()
    index = _RecordingIndex()
    registry = _RecordingRegistry(store, index)

    ingestion = SimpleNamespace(faiss_index=index)
    config = SimpleNamespace(retrieval=RetrievalConfig(dense_calibration_batch_size=2))
    service = SimpleNamespace(_config_manager=SimpleNamespace(get=lambda: config))

    validator = HybridSearchValidator(
        ingestion=ingestion,
        service=service,
        registry=registry,
        opensearch=SimpleNamespace(),
    )

    report = validator._run_calibration([])

    expected_batches = math.ceil(len(vector_ids) / 2)
    assert len(store.calls) == expected_batches
    assert store.calls == [tuple(vector_ids[i : i + 2]) for i in range(0, len(vector_ids), 2)]
    dense_results = report.details["dense"]
    assert len(dense_results) == 3
    assert all(math.isclose(entry["self_hit_accuracy"], 1.0) for entry in dense_results)


def test_calibration_skips_missing_vectors_without_redundant_reconstruction():
    vector_ids = ["vec-available", "vec-missing"]
    embeddings = {
        "vec-available": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }

    class _PartiallyMissingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def reconstruct_batch(self, vector_ids):
            self.calls.append(tuple(vector_ids))
            for vector_id in vector_ids:
                if vector_id not in embeddings:
                    raise KeyError(vector_id)
            rows = [embeddings[vector_id] for vector_id in vector_ids]
            return np.asarray(rows, dtype=np.float32)

    class _RecordingIndex:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.last_vector_ids: list[str] = []

        def search_batch(self, queries: np.ndarray, top_k: int):
            assert queries.dtype == np.float32
            assert queries.flags.c_contiguous
            self.calls.append({"shape": tuple(queries.shape), "top_k": top_k})
            results = []
            for vector_id in self.last_vector_ids:
                hits = [FaissSearchResult(vector_id=vector_id, score=1.0)]
                for extra in range(1, top_k):
                    hits.append(
                        FaissSearchResult(
                            vector_id=f"noise-{vector_id}-{extra}", score=0.0
                        )
                    )
                results.append(hits)
            return results

    class _RecordingRegistry:
        def __init__(self, store: _PartiallyMissingStore, index: _RecordingIndex) -> None:
            self._store = store
            self._index = index
            self._chunks = [SimpleNamespace(vector_id=vector_id) for vector_id in vector_ids]

        def all(self):
            return list(self._chunks)

        def resolve_embeddings(self, vector_ids, *, cache=None, dtype=np.float32):
            dtype = np.dtype(dtype)
            results = [None] * len(vector_ids)
            missing = []
            missing_positions = []
            if cache is not None:
                for idx, vector_id in enumerate(vector_ids):
                    cached = cache.get(vector_id)
                    if cached is not None:
                        results[idx] = np.asarray(cached, dtype=np.float32)
                    else:
                        missing.append(vector_id)
                        missing_positions.append(idx)
            else:
                missing = list(vector_ids)
                missing_positions = list(range(len(vector_ids)))
            if missing:
                reconstructed = self._store.reconstruct_batch(missing)
                if reconstructed.shape[0] != len(missing):
                    raise RuntimeError(
                        "Dense store returned mismatched reconstruction rows "
                        f"(expected {len(missing)}, received {reconstructed.shape[0]})"
                    )
                for offset, idx in enumerate(missing_positions):
                    row = np.asarray(reconstructed[offset], dtype=np.float32)
                    results[idx] = row
                    if cache is not None:
                        cache[vector_ids[idx]] = row
            resolved = []
            missing_ids = []
            for idx, candidate in enumerate(results):
                if candidate is None:
                    missing_ids.append(vector_ids[idx])
                else:
                    resolved.append(candidate)
            if missing_ids:
                raise KeyError(f"Embeddings missing for {missing_ids!r}")
            matrix = np.ascontiguousarray(np.stack(resolved), dtype=dtype)
            self._index.last_vector_ids = list(vector_ids)
            return matrix

        def resolve_embedding(self, vector_id: str, *, cache=None, dtype=np.float32):
            matrix = self.resolve_embeddings([vector_id], cache=cache, dtype=dtype)
            return matrix[0]

    store = _PartiallyMissingStore()
    index = _RecordingIndex()
    registry = _RecordingRegistry(store, index)

    ingestion = SimpleNamespace(faiss_index=index)
    config = SimpleNamespace(retrieval=RetrievalConfig(dense_calibration_batch_size=1))
    service = SimpleNamespace(_config_manager=SimpleNamespace(get=lambda: config))

    validator = HybridSearchValidator(
        ingestion=ingestion,
        service=service,
        registry=registry,
        opensearch=SimpleNamespace(),
    )

    report = validator._run_calibration([])

    assert store.calls[0] == tuple(vector_ids)
    assert store.calls.count(("vec-available",)) == 1
    assert store.calls.count(("vec-missing",)) == 1
    assert len(store.calls) == 3
    dense_results = report.details["dense"]
    assert len(dense_results) == 3
    expected_accuracy = 1 / max(1, len(vector_ids))
    assert all(
        math.isclose(entry["self_hit_accuracy"], expected_accuracy) for entry in dense_results
    )


@pytest.fixture
def duplicate_namespace_registry() -> tuple[SimpleNamespace, list[ChunkPayload], dict[str, np.ndarray]]:
    research_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    support_embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    chunks = [
        ChunkPayload(
            doc_id="doc-1",
            chunk_id="0",
            vector_id="vec-research",
            namespace="research",
            text="research chunk",
            metadata={},
            features=ChunkFeatures({}, {}, research_embedding),
            token_count=3,
            source_chunk_idxs=(0,),
            doc_items_refs=("doc:0",),
        ),
        ChunkPayload(
            doc_id="doc-1",
            chunk_id="0",
            vector_id="vec-support",
            namespace="support",
            text="support chunk",
            metadata={},
            features=ChunkFeatures({}, {}, support_embedding),
            token_count=3,
            source_chunk_idxs=(0,),
            doc_items_refs=("doc:0",),
        ),
    ]

    embeddings = {
        "vec-research": research_embedding,
        "vec-support": support_embedding,
    }

    class _Registry(SimpleNamespace):
        def all(self):  # type: ignore[override]
            return chunks

        def resolve_embedding(self, vector_id: str, *, cache=None, dtype=np.float32):
            return embeddings[vector_id]

    return _Registry(), chunks, embeddings


def test_embeddings_for_results_respect_namespace(duplicate_namespace_registry):
    registry, _chunks, embeddings = duplicate_namespace_registry

    validator = HybridSearchValidator(
        ingestion=SimpleNamespace(),
        service=SimpleNamespace(),
        registry=registry,
        opensearch=SimpleNamespace(),
    )

    chunk_lookup = {
        (chunk.namespace, chunk.doc_id, chunk.chunk_id): chunk for chunk in registry.all()
    }

    results = [
        HybridSearchResult(
            doc_id="doc-1",
            chunk_id="0",
            namespace="research",
            vector_id="vec-research",
            score=1.0,
            fused_rank=0,
            text="research chunk",
            highlights=(),
            provenance_offsets=(),
            diagnostics=None,
            metadata={},
        ),
        HybridSearchResult(
            doc_id="doc-1",
            chunk_id="0",
            namespace="support",
            vector_id="vec-support",
            score=1.0,
            fused_rank=1,
            text="support chunk",
            highlights=(),
            provenance_offsets=(),
            diagnostics=None,
            metadata={},
        ),
    ]

    resolved = validator._embeddings_for_results(results, chunk_lookup, limit=2)

    assert np.array_equal(resolved[0], embeddings["vec-research"])
    assert np.array_equal(resolved[1], embeddings["vec-support"])
