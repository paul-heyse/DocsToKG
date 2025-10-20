"""Tests for dense execution behaviour in :mod:`HybridSearchService`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from DocsToKG.HybridSearch.config import HybridSearchConfig
from DocsToKG.HybridSearch.service import DenseSearchStrategy, HybridSearchService
from DocsToKG.HybridSearch.store import FaissSearchResult
from DocsToKG.HybridSearch.types import (
    ChunkFeatures,
    ChunkPayload,
    HybridSearchRequest,
)


class _StubMetrics:
    def set_gauge(self, *args: object, **kwargs: object) -> None:
        return None

    def observe(self, *args: object, **kwargs: object) -> None:
        return None

    def increment(self, *args: object, **kwargs: object) -> None:
        return None

    def percentile(self, *args: object, **kwargs: object) -> None:
        return None


class _StubObservability:
    def __init__(self) -> None:
        self.metrics = _StubMetrics()


class _StubRegistry:
    def __init__(self, payloads: dict[str, ChunkPayload]) -> None:
        self.payloads = payloads

    def bulk_get(self, vector_ids: list[str]) -> list[ChunkPayload]:
        return [self.payloads[vid] for vid in vector_ids if vid in self.payloads]

    def resolve_embedding(self, vector_id: str, *, cache: dict[str, np.ndarray]) -> np.ndarray:
        return self.payloads[vector_id].features.embedding


class _StubDenseStore:
    def __init__(self, sequences: list[list[FaissSearchResult]]) -> None:
        self._sequences = sequences
        self._call_index = 0
        self.adapter_stats = SimpleNamespace(fp16_enabled=False, nprobe=1)

    def search_batch(self, queries: np.ndarray, depth: int) -> list[list[FaissSearchResult]]:
        index = min(self._call_index, len(self._sequences) - 1)
        self._call_index += 1
        return [self._sequences[index]]


def _make_payload(vector_id: str, value: float) -> ChunkPayload:
    embedding = np.array([value], dtype=np.float32)
    return ChunkPayload(
        doc_id=f"doc-{vector_id}",
        chunk_id=f"chunk-{vector_id}",
        vector_id=vector_id,
        namespace="demo",
        text="dense chunk",
        metadata={},
        features=ChunkFeatures({}, {}, embedding),
        token_count=1,
        source_chunk_idxs=(),
        doc_items_refs=(),
    )


def test_cached_signature_updates_pass_rate_after_second_run() -> None:
    """Dense planner should update cached pass rate on every execution."""

    service = object.__new__(HybridSearchService)
    service._dense_strategy = DenseSearchStrategy()
    service._observability = _StubObservability()

    initial_payloads = {"vec-1": _make_payload("vec-1", 0.5)}
    registry = _StubRegistry(initial_payloads)
    service._registry = registry

    first_hits = [
        FaissSearchResult(vector_id="vec-1", score=0.42),
        FaissSearchResult(vector_id="vec-2", score=0.40),
    ]
    second_hits = [
        FaissSearchResult(vector_id="vec-1", score=0.61),
        FaissSearchResult(vector_id="vec-2", score=0.58),
    ]
    dense_store = _StubDenseStore([first_hits, second_hits])

    request = HybridSearchRequest(
        query="dense planner",
        namespace="demo",
        filters={},
        page_size=1,
    )
    filters: dict[str, object] = {}
    config = HybridSearchConfig()
    query_features = ChunkFeatures({}, {}, np.array([0.33], dtype=np.float32))

    timings: dict[str, float] = {}
    service._execute_dense(request, filters, config, query_features, timings, dense_store)

    signature = service._dense_request_signature(request, filters)
    assert service._dense_strategy.has_cache(signature)
    first_pass = service._dense_strategy._signature_pass[signature]

    registry.payloads["vec-2"] = _make_payload("vec-2", 0.6)

    timings_second: dict[str, float] = {}
    service._execute_dense(request, filters, config, query_features, timings_second, dense_store)

    second_pass = service._dense_strategy._signature_pass[signature]
    assert second_pass > first_pass
