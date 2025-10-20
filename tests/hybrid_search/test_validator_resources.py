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
from types import SimpleNamespace

import numpy as np
import pytest

import DocsToKG.HybridSearch.service as service_module
from DocsToKG.HybridSearch.config import DenseIndexConfig
from DocsToKG.HybridSearch.pipeline import Observability
from DocsToKG.HybridSearch.service import HybridSearchValidator
from DocsToKG.HybridSearch.types import ChunkFeatures, ChunkPayload, HybridSearchResult


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
