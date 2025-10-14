"""
Hybrid search execution across sparse and dense channels.

This module provides the core hybrid search service for DocsToKG, orchestrating
multiple retrieval methods (BM25, SPLADE, dense vectors) and fusing their results
for optimal document retrieval performance.

The service supports configurable search strategies, real-time observability,
and comprehensive result ranking through advanced fusion techniques.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .config import HybridSearchConfig, HybridSearchConfigManager
from .dense import FaissIndexManager, FaissSearchResult
from .features import FeatureGenerator
from .fusion import ReciprocalRankFusion, apply_mmr_diversification
from .observability import Observability
from .results import ResultShaper
from .storage import ChunkRegistry, OpenSearchSimulator
from .types import ChunkFeatures, FusionCandidate, HybridSearchRequest, HybridSearchResponse


class RequestValidationError(ValueError):
    """Raised when the caller submits an invalid search request.

    This exception is raised when a hybrid search request contains invalid
    parameters, malformed data, or violates system constraints.

    Attributes:
        message: Description of the validation error
        field: Optional field name that caused the error

    Examples:
        >>> try:
        ...     service.search(invalid_request)
        ... except RequestValidationError as e:
        ...     print(f"Invalid request: {e.message}")
    """


@dataclass(slots=True)
class ChannelResults:
    """Results from a single retrieval channel (BM25, SPLADE, or dense).

    This class encapsulates the candidates and scoring information returned
    by a specific retrieval method, preparing them for fusion with results
    from other channels.

    Attributes:
        candidates: List of fusion candidates from this channel
        scores: Performance metrics for this channel's execution

    Examples:
        >>> bm25_results = ChannelResults(
        ...     candidates=[candidate1, candidate2],
        ...     scores={"recall": 0.85, "latency": 45}
        ... )
    """
    candidates: List[FusionCandidate]
    scores: Dict[str, float]


class HybridSearchService:
    """Execute BM25, SPLADE, and dense retrieval with fusion.

    This service orchestrates hybrid search operations by:
    1. Executing parallel retrieval across multiple channels
    2. Fusing results using configurable strategies
    3. Applying diversification and ranking optimizations
    4. Providing comprehensive observability and metrics

    The service is designed for high-performance document retrieval
    combining traditional lexical search with modern semantic methods.

    Attributes:
        _config_manager: Configuration management for search parameters
        _feature_generator: Feature extraction for query processing
        _faiss: Dense vector search using FAISS
        _opensearch: Lexical search using OpenSearch
        _registry: Document and chunk registry management
        _observability: Performance monitoring and metrics collection

    Examples:
        >>> service = HybridSearchService(
        ...     config_manager=config_manager,
        ...     feature_generator=feature_generator,
        ...     faiss_index=faiss_index,
        ...     opensearch=opensearch,
        ...     registry=registry
        ... )
        >>> results = service.search(request)
    """

    def __init__(
        self,
        *,
        config_manager: HybridSearchConfigManager,
        feature_generator: FeatureGenerator,
        faiss_index: FaissIndexManager,
        opensearch: OpenSearchSimulator,
        registry: ChunkRegistry,
        observability: Optional[Observability] = None,
    ) -> None:
        self._config_manager = config_manager
        self._feature_generator = feature_generator
        self._faiss = faiss_index
        self._opensearch = opensearch
        self._registry = registry
        self._observability = observability or Observability()

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        config = self._config_manager.get()
        self._validate_request(request)
        filters = dict(request.filters)
        if request.namespace:
            filters["namespace"] = request.namespace

        with self._observability.trace("hybrid_search", namespace=request.namespace or "*"):
            timings: Dict[str, float] = {}
            total_start = time.perf_counter()
            query_start = time.perf_counter()
            query_features = self._feature_generator.compute_features(request.query)
            timings["feature_ms"] = (time.perf_counter() - query_start) * 1000

            bm25 = self._execute_bm25(request, filters, config, query_features, timings)
            splade = self._execute_splade(request, filters, config, query_features, timings)
            dense = self._execute_dense(request, filters, config, query_features, timings)

            fusion = ReciprocalRankFusion(config.fusion.k0)
            combined_candidates = bm25.candidates + splade.candidates + dense.candidates
            fused_scores = fusion.fuse(combined_candidates)
            unique_candidates = self._dedupe_candidates(combined_candidates, fused_scores)

            if request.diversification and config.fusion.enable_mmr:
                diversified = apply_mmr_diversification(
                    unique_candidates,
                    fused_scores,
                    config.fusion.mmr_lambda,
                    request.page_size * config.dense.oversample,
                )
            else:
                diversified = unique_candidates

            ordered_chunks = [candidate.chunk for candidate in diversified]
            channel_score_map = {
                "bm25": bm25.scores,
                "splade": splade.scores,
                "dense": dense.scores,
            }
            shaper = ResultShaper(self._opensearch, config.fusion)
            shaped_results = shaper.shape(ordered_chunks, fused_scores, request, channel_score_map)

            start = int(request.cursor or "0")
            end = start + request.page_size
            page = shaped_results[start:end]
            next_cursor = str(end) if end < len(shaped_results) else None

            timings["total_ms"] = (time.perf_counter() - total_start) * 1000
            self._observability.metrics.observe("search_candidates", len(combined_candidates))
            self._observability.logger.info(
                "hybrid-search",
                extra={
                    "event": {
                        "query": request.query,
                        "namespace": request.namespace,
                        "returned": len(page),
                        "total_candidates": len(shaped_results),
                    }
                },
            )

        return HybridSearchResponse(
            results=page,
            next_cursor=next_cursor,
            total_candidates=len(shaped_results),
            timings_ms=timings,
        )

    def _execute_bm25(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
        config: HybridSearchConfig,
        query_features: ChunkFeatures,
        timings: Dict[str, float],
    ) -> ChannelResults:
        start = time.perf_counter()
        hits, _ = self._opensearch.search_bm25(
            query_features.bm25_terms,
            filters,
            top_k=config.retrieval.bm25_top_k,
        )
        timings["bm25_ms"] = (time.perf_counter() - start) * 1000
        self._observability.metrics.increment("search_channel_requests", channel="bm25")
        self._observability.metrics.observe("search_channel_candidates", len(hits), channel="bm25")
        candidates = [
            FusionCandidate(source="bm25", score=score, chunk=chunk, rank=idx + 1)
            for idx, (chunk, score) in enumerate(hits)
        ]
        scores = {chunk.vector_id: score for chunk, score in hits}
        return ChannelResults(candidates=candidates, scores=scores)

    def _execute_splade(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
        config: HybridSearchConfig,
        query_features: ChunkFeatures,
        timings: Dict[str, float],
    ) -> ChannelResults:
        start = time.perf_counter()
        hits, _ = self._opensearch.search_splade(
            query_features.splade_weights,
            filters,
            top_k=config.retrieval.splade_top_k,
        )
        timings["splade_ms"] = (time.perf_counter() - start) * 1000
        self._observability.metrics.increment("search_channel_requests", channel="splade")
        self._observability.metrics.observe("search_channel_candidates", len(hits), channel="splade")
        candidates = [
            FusionCandidate(source="splade", score=score, chunk=chunk, rank=idx + 1)
            for idx, (chunk, score) in enumerate(hits)
        ]
        scores = {chunk.vector_id: score for chunk, score in hits}
        return ChannelResults(candidates=candidates, scores=scores)

    def _execute_dense(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
        config: HybridSearchConfig,
        query_features: ChunkFeatures,
        timings: Dict[str, float],
    ) -> ChannelResults:
        start = time.perf_counter()
        oversampled = request.page_size * config.dense.oversample
        hits = self._faiss.search(query_features.embedding, min(config.retrieval.dense_top_k, oversampled))
        timings["dense_ms"] = (time.perf_counter() - start) * 1000
        filtered = self._filter_dense_hits(hits, filters)
        self._observability.metrics.increment("search_channel_requests", channel="dense")
        self._observability.metrics.observe("search_channel_candidates", len(filtered), channel="dense")
        candidates: List[FusionCandidate] = []
        scores: Dict[str, float] = {}
        for idx, hit in enumerate(filtered):
            chunk = self._registry.get(hit.vector_id)
            if chunk is None:
                continue
            candidates.append(
                FusionCandidate(source="dense", score=hit.score, chunk=chunk, rank=idx + 1)
            )
            scores[hit.vector_id] = hit.score
        return ChannelResults(candidates=candidates, scores=scores)

    def _filter_dense_hits(
        self,
        hits: Sequence[FaissSearchResult],
        filters: Mapping[str, object],
    ) -> List[FaissSearchResult]:
        if not hits:
            return []
        chunk_map = {
            chunk.vector_id: chunk for chunk in self._registry.bulk_get([hit.vector_id for hit in hits])
        }
        filtered: List[FaissSearchResult] = []
        for hit in hits:
            chunk = chunk_map.get(hit.vector_id)
            if chunk is None:
                continue
            if self._matches_filters(chunk, filters):
                filtered.append(hit)
        return filtered

    def _matches_filters(self, chunk, filters: Mapping[str, object]) -> bool:
        for key, expected in filters.items():
            if key == "namespace":
                if chunk.namespace != expected:
                    return False
                continue
            value = chunk.metadata.get(key)
            if isinstance(expected, list):
                if isinstance(value, list):
                    if not any(item in value for item in expected):
                        return False
                else:
                    if value not in expected:
                        return False
            else:
                if value != expected:
                    return False
        return True

    def _dedupe_candidates(
        self,
        candidates: Sequence[FusionCandidate],
        fused_scores: Mapping[str, float],
    ) -> List[FusionCandidate]:
        unique: Dict[str, FusionCandidate] = {}
        for candidate in candidates:
            vector_id = candidate.chunk.vector_id
            if vector_id not in fused_scores:
                continue
            if vector_id not in unique:
                unique[vector_id] = candidate
                continue
            existing = unique[vector_id]
            if fused_scores[vector_id] > fused_scores[existing.chunk.vector_id]:
                unique[vector_id] = candidate
        ordered = sorted(
            unique.values(),
            key=lambda candidate: fused_scores[candidate.chunk.vector_id],
            reverse=True,
        )
        return ordered

    def _validate_request(self, request: HybridSearchRequest) -> None:
        if not request.query.strip():
            raise RequestValidationError("query must not be empty")
        if request.page_size <= 0:
            raise RequestValidationError("page_size must be positive")
        if request.page_size > 100:
            raise RequestValidationError("page_size too large")

