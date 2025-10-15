"""Hybrid search service and synchronous HTTP-style API."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from .config import HybridSearchConfig, HybridSearchConfigManager
from .features import FeatureGenerator
from .observability import Observability
from .ranking import ReciprocalRankFusion, ResultShaper, apply_mmr_diversification
from .storage import ChunkRegistry, OpenSearchSimulator, matches_filters
from .types import (
    ChunkFeatures,
    ChunkPayload,
    FusionCandidate,
    HybridSearchRequest,
    HybridSearchResponse,
)
from .vectorstore import FaissIndexManager, FaissSearchResult


class RequestValidationError(ValueError):
    """Raised when the caller submits an invalid search request."""


@dataclass(slots=True)
class ChannelResults:
    """Results from a single retrieval channel (BM25, SPLADE, or dense)."""

    candidates: List[FusionCandidate]
    scores: Dict[str, float]


class HybridSearchService:
    """Execute BM25, SPLADE, and dense retrieval with fusion."""

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
        self._faiss.set_id_resolver(self._registry.resolve_faiss_id)

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

            with ThreadPoolExecutor(max_workers=3) as pool:
                f_bm25 = pool.submit(
                    self._execute_bm25, request, filters, config, query_features, timings
                )
                f_splade = pool.submit(
                    self._execute_splade, request, filters, config, query_features, timings
                )
                f_dense = pool.submit(
                    self._execute_dense, request, filters, config, query_features, timings
                )
                bm25 = f_bm25.result()
                splade = f_splade.result()
                dense = f_dense.result()

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
                    device=self._faiss.device,
                    resources=self._faiss.gpu_resources,
                )
            else:
                diversified = unique_candidates

            ordered_chunks = [candidate.chunk for candidate in diversified]
            channel_score_map = {
                "bm25": bm25.scores,
                "splade": splade.scores,
                "dense": dense.scores,
            }
            shaper = ResultShaper(
                self._opensearch,
                config.fusion,
                device=self._faiss.device,
                resources=self._faiss.gpu_resources,
            )
            shaped = shaper.shape(
                ordered_chunks,
                fused_scores,
                request,
                channel_score_map,
            )
            timings["fusion_ms"] = (time.perf_counter() - total_start) * 1000
            self._observability.metrics.increment("hybrid_search_requests")
            self._observability.metrics.observe("hybrid_search_results", len(shaped))
            self._observability.metrics.observe(
                "hybrid_search_timings_total_ms", timings["fusion_ms"]
            )
            next_cursor = self._build_cursor(shaped, request.page_size)
            self._observability.logger.info(
                "hybrid-search",
                extra={
                    "event": {
                        "query": request.query,
                        "namespace": request.namespace,
                        "results": len(shaped),
                        "timings_ms": {k: round(v, 3) for k, v in timings.items()},
                    }
                },
            )
            return HybridSearchResponse(
                results=shaped[: request.page_size],
                next_cursor=next_cursor,
                total_candidates=len(unique_candidates),
                timings_ms=timings,
            )

    def _validate_request(self, request: HybridSearchRequest) -> None:
        if not request.query.strip():
            raise RequestValidationError("Query must not be empty")
        if request.page_size <= 0:
            raise RequestValidationError("page_size must be positive")
        if request.page_size > 1000:
            raise RequestValidationError("page_size exceeds maximum")

    def _build_cursor(self, results: Sequence[ChunkPayload], page_size: int) -> Optional[str]:
        if len(results) <= page_size:
            return None
        last = results[page_size - 1]
        return f"{last.vector_id}:{last.fused_rank}"

    def _execute_bm25(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
        config: HybridSearchConfig,
        query_features: ChunkFeatures,
        timings: Dict[str, float],
    ) -> ChannelResults:
        start = time.perf_counter()
        hits = self._opensearch.search_bm25(
            query_features.bm25_terms,
            filters,
            top_k=config.retrieval.bm25_top_k,
        )
        timings["bm25_ms"] = (time.perf_counter() - start) * 1000
        self._observability.metrics.increment("search_channel_requests", channel="bm25")
        self._observability.metrics.observe(
            "search_channel_candidates", len(hits), channel="bm25"
        )
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
        self._observability.metrics.observe(
            "search_channel_candidates", len(hits), channel="splade"
        )
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
        oversampled = int(
            request.page_size
            * config.dense.oversample
            * max(1.0, float(getattr(config.retrieval, "dense_overfetch_factor", 1.5)))
        )
        hits = self._faiss.search(
            query_features.embedding, min(config.retrieval.dense_top_k, oversampled)
        )
        timings["dense_ms"] = (time.perf_counter() - start) * 1000
        filtered, payloads = self._filter_dense_hits(hits, filters)
        self._observability.metrics.increment("search_channel_requests", channel="dense")
        self._observability.metrics.observe(
            "search_channel_candidates", len(filtered), channel="dense"
        )
        candidates: List[FusionCandidate] = []
        scores: Dict[str, float] = {}
        for idx, hit in enumerate(filtered):
            chunk = payloads.get(hit.vector_id)
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
    ) -> tuple[List[FaissSearchResult], Dict[str, ChunkPayload]]:
        if not hits:
            return [], {}
        vector_ids = [hit.vector_id for hit in hits]
        payloads = {chunk.vector_id: chunk for chunk in self._registry.bulk_get(vector_ids)}
        filtered = [
            hit
            for hit in hits
            if (chunk := payloads.get(hit.vector_id)) is not None
            and matches_filters(chunk, filters)
        ]
        return filtered, payloads

    def _dedupe_candidates(
        self,
        candidates: Sequence[FusionCandidate],
        fused_scores: Mapping[str, float],
    ) -> List[FusionCandidate]:
        unique: Dict[str, FusionCandidate] = {}
        for candidate in candidates:
            vector_id = candidate.chunk.vector_id
            best = unique.get(vector_id)
            if best is None or fused_scores[vector_id] > fused_scores[best.chunk.vector_id]:
                unique[vector_id] = candidate
        return sorted(
            unique.values(),
            key=lambda candidate: fused_scores.get(candidate.chunk.vector_id, 0.0),
            reverse=True,
        )


class HybridSearchAPI:
    """Minimal synchronous handler for `POST /v1/hybrid-search`."""

    def __init__(self, service: HybridSearchService) -> None:
        if not isinstance(service, HybridSearchService):
            raise TypeError("service must be a HybridSearchService instance")
        self._service = service

    def post_hybrid_search(self, payload: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        try:
            request = self._parse_request(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return HTTPStatus.BAD_REQUEST, {"error": str(exc)}

        try:
            response = self._service.search(request)
        except RequestValidationError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive guard
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}

        body = {
            "results": [
                {
                    "doc_id": result.doc_id,
                    "chunk_id": result.chunk_id,
                    "namespace": result.namespace,
                    "score": result.score,
                    "fused_rank": result.fused_rank,
                    "text": result.text,
                    "highlights": list(result.highlights),
                    "provenance_offsets": [list(offset) for offset in result.provenance_offsets],
                    "metadata": dict(result.metadata),
                    "diagnostics": {
                        "bm25": result.diagnostics.bm25_score,
                        "splade": result.diagnostics.splade_score,
                        "dense": result.diagnostics.dense_score,
                    },
                }
                for result in response.results
            ],
            "next_cursor": response.next_cursor,
            "total_candidates": response.total_candidates,
            "timings_ms": dict(response.timings_ms),
        }
        return HTTPStatus.OK, body

    def _parse_request(self, payload: Mapping[str, Any]) -> HybridSearchRequest:
        query = str(payload["query"])
        namespace = payload.get("namespace")
        filters = self._normalize_filters(payload.get("filters", {}))
        page_size = int(payload.get("page_size", payload.get("limit", 10)))
        cursor = payload.get("cursor")
        diversification = bool(payload.get("diversification", False))
        diagnostics = bool(payload.get("diagnostics", True))
        return HybridSearchRequest(
            query=query,
            namespace=str(namespace) if namespace is not None else None,
            filters=filters,
            page_size=page_size,
            cursor=str(cursor) if cursor is not None else None,
            diversification=diversification,
            diagnostics=diagnostics,
        )

    def _normalize_filters(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        normalized: MutableMapping[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                normalized[str(key)] = [str(item) for item in value]
            else:
                normalized[str(key)] = value
        return normalized


__all__ = [
    "ChannelResults",
    "HybridSearchAPI",
    "HybridSearchService",
    "RequestValidationError",
]

