# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.service",
#   "purpose": "Hybrid search orchestration, pagination guards, and synchronous API surface",
#   "sections": [
#     {
#       "id": "requestvalidationerror",
#       "name": "RequestValidationError",
#       "anchor": "class-requestvalidationerror",
#       "kind": "class"
#     },
#     {
#       "id": "channelresults",
#       "name": "ChannelResults",
#       "anchor": "class-channelresults",
#       "kind": "class"
#     },
#     {
#       "id": "hybridsearchservice",
#       "name": "HybridSearchService",
#       "anchor": "class-hybridsearchservice",
#       "kind": "class"
#     },
#     {
#       "id": "hybridsearchapi",
#       "name": "HybridSearchAPI",
#       "anchor": "class-hybridsearchapi",
#       "kind": "class"
#     },
#     {
#       "id": "paginationcheckresult",
#       "name": "PaginationCheckResult",
#       "anchor": "class-paginationcheckresult",
#       "kind": "class"
#     },
#     {
#       "id": "build-stats-snapshot",
#       "name": "build_stats_snapshot",
#       "anchor": "function-build-stats-snapshot",
#       "kind": "function"
#     },
#     {
#       "id": "verify-pagination",
#       "name": "verify_pagination",
#       "anchor": "function-verify-pagination",
#       "kind": "function"
#     },
#     {
#       "id": "should-rebuild-index",
#       "name": "should_rebuild_index",
#       "anchor": "function-should-rebuild-index",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Hybrid search orchestration, pagination guards, and synchronous HTTP-style API."""

from __future__ import annotations
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from threading import RLock
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .config import HybridSearchConfig, HybridSearchConfigManager
from .features import FeatureGenerator
from .interfaces import DenseVectorStore, LexicalIndex
from .observability import Observability
from .ranking import ReciprocalRankFusion, ResultShaper, apply_mmr_diversification
from .router import FaissRouter
from .storage import ChunkRegistry, matches_filters
from .types import (
    ChunkFeatures,
    ChunkPayload,
    FusionCandidate,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
)
from .vectorstore import FaissSearchResult

# --- Globals ---

__all__ = (
    "ChannelResults",
    "HybridSearchAPI",
    "HybridSearchService",
    "PaginationCheckResult",
    "RequestValidationError",
    "build_stats_snapshot",
    "should_rebuild_index",
    "verify_pagination",
)


# --- Public Classes ---


class RequestValidationError(ValueError):
    """Raised when the caller submits an invalid search request.

    Attributes:
        None

    Examples:
        >>> raise RequestValidationError("page_size must be positive")
        Traceback (most recent call last):
        ...
        RequestValidationError: page_size must be positive
    """


@dataclass(slots=True)
class ChannelResults:
    """Results from a single retrieval channel (BM25, SPLADE, or dense).

    Attributes:
        candidates: Ordered list of channel-specific fusion candidates.
        scores: Mapping of vector identifiers to raw channel scores.

    Examples:
        >>> ChannelResults(candidates=[], scores={})
        ChannelResults(candidates=[], scores={})
    """

    candidates: List[FusionCandidate]
    scores: Dict[str, float]


class HybridSearchService:
    """Execute BM25, SPLADE, and dense retrieval with fusion.

    Attributes:
        _config_manager: Source of runtime hybrid-search configuration.
        _feature_generator: Component producing BM25/SPLADE/dense features.
        _faiss: Default FAISS index used for namespaces routed to the shared store.
        _faiss_router: Namespace-aware router managing FAISS stores.
        _opensearch: Lexical index used for BM25 and SPLADE lookups.
        _registry: Chunk registry providing metadata and FAISS id lookups.
        _observability: Telemetry facade for metrics and traces.

    Examples:
        >>> # File-backed config manager (JSON/YAML on disk)
        >>> from pathlib import Path  # doctest: +SKIP
        >>> manager = HybridSearchConfigManager(Path("config.json"))  # doctest: +SKIP
        >>> from DocsToKG.HybridSearch.storage import OpenSearchSimulator  # doctest: +SKIP
        >>> service = HybridSearchService(  # doctest: +SKIP
        ...     config_manager=manager,
        ...     feature_generator=FeatureGenerator(embedding_dim=16),
        ...     faiss_index=FaissVectorStore(dim=16, config=HybridSearchConfig().dense),
        ...     opensearch=OpenSearchSimulator(),
        ...     registry=ChunkRegistry(),
        ... )
        >>> request = HybridSearchRequest(query="example", namespace="demo", filters={}, page_size=5)
        >>> isinstance(service.search(request), HybridSearchResponse)  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        *,
        config_manager: HybridSearchConfigManager,
        feature_generator: FeatureGenerator,
        faiss_index: DenseVectorStore,
        opensearch: LexicalIndex,
        registry: ChunkRegistry,
        observability: Optional[Observability] = None,
        faiss_router: Optional[FaissRouter] = None,
    ) -> None:
        """Initialise the hybrid search service.

        Args:
            config_manager: Manager providing the latest search configuration.
            feature_generator: Component responsible for feature extraction.
            faiss_index: Dense retrieval index manager (GPU-backed).
            opensearch: Lexical index providing BM25/SPLADE access.
            registry: Chunk registry used for metadata lookups.
            observability: Optional observability facade (defaults to a no-op).

        Returns:
            None
        """
        self._config_manager = config_manager
        self._feature_generator = feature_generator
        self._opensearch = opensearch
        self._registry = registry
        self._observability = observability or Observability()
        if faiss_router is not None:
            self._faiss_router = faiss_router
        else:
            self._faiss_router = FaissRouter(per_namespace=False, default_store=faiss_index)
        self._faiss = self._faiss_router.default_store
        self._faiss_router.set_resolver(self._registry.resolve_faiss_id)
        self._dense_filter_pass_ema: float = 0.75
        self._dense_filter_ema_alpha: float = 0.20
        self._dense_k_cache: OrderedDict[tuple, int] = OrderedDict()
        self._dense_k_cache_limit: int = 64
        self._dense_k_cache_lock = RLock()

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        """Execute a hybrid retrieval round trip for ``request``.

        Args:
            request: Fully validated hybrid search request describing the query,
                namespace, filters, and pagination parameters.

        Returns:
            HybridSearchResponse: Ranked hybrid search results enriched with channel-level
            diagnostics and pagination cursor metadata.

        Raises:
            RequestValidationError: If ``request`` fails validation checks.
        """
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

            dense_store = self._dense_store(request.namespace)

            with ThreadPoolExecutor(max_workers=3) as pool:
                f_bm25 = pool.submit(
                    self._execute_bm25, request, filters, config, query_features, timings
                )
                f_splade = pool.submit(
                    self._execute_splade, request, filters, config, query_features, timings
                )
                f_dense = pool.submit(
                    self._execute_dense,
                    request,
                    filters,
                    config,
                    query_features,
                    timings,
                    dense_store,
                )
                bm25 = f_bm25.result()
                splade = f_splade.result()
                dense = f_dense.result()

            channel_weights = getattr(config.fusion, "channel_weights", None)
            fusion = ReciprocalRankFusion(
                k0=config.fusion.k0,
                channel_weights=dict(channel_weights) if channel_weights is not None else None,
            )
            combined_candidates = bm25.candidates + splade.candidates + dense.candidates
            fused_scores = fusion.fuse(combined_candidates)
            unique_candidates = self._dedupe_candidates(combined_candidates, fused_scores)

            fusion_start = time.perf_counter()
            if request.diversification and config.fusion.enable_mmr:
                pool_size = int(
                    max(
                        1,
                        min(
                            getattr(config.fusion, "mmr_pool_size", len(unique_candidates)),
                            len(unique_candidates),
                        ),
                    )
                )
                pool = list(unique_candidates[:pool_size])
                desired = min(request.page_size, len(pool))
                selected = apply_mmr_diversification(
                    pool,
                    fused_scores,
                    config.fusion.mmr_lambda,
                    desired,
                    device=dense_store.device,
                    resources=dense_store.gpu_resources,
                )
                selected_ids = {candidate.chunk.vector_id for candidate in selected}
                pool_remaining = [
                    candidate for candidate in pool if candidate.chunk.vector_id not in selected_ids
                ]
                tail = list(unique_candidates[pool_size:])
                diversified = [*selected, *pool_remaining, *tail]
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
                device=dense_store.device,
                resources=dense_store.gpu_resources,
            )
            shaped = shaper.shape(
                ordered_chunks,
                fused_scores,
                request,
                channel_score_map,
            )
            timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000
            self._observability.metrics.increment("hybrid_search_requests")
            self._observability.metrics.observe("hybrid_search_results", len(shaped))
            self._observability.metrics.observe(
                "hybrid_search_timings_total_ms", timings["total_ms"]
            )
            paged_results = self._slice_from_cursor(shaped, request.cursor)
            next_cursor = self._build_cursor(paged_results, request.page_size)
            page_results = paged_results[: request.page_size]
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
                results=page_results,
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

    def _dense_store(self, namespace: Optional[str]) -> DenseVectorStore:
        return self._faiss_router.get(namespace)

    def _slice_from_cursor(
        self, results: Sequence[HybridSearchResult], cursor: Optional[str]
    ) -> List[HybridSearchResult]:
        if not cursor:
            return list(results)
        try:
            if cursor.startswith("v2|"):
                _version, score_str, vector_id = cursor.split("|", 2)
                target_score = float(score_str)
                sliced: List[HybridSearchResult] = []
                skipping = True
                for result in results:
                    if skipping:
                        if (
                            result.vector_id == vector_id
                            and abs(result.score - target_score) < 1e-6
                        ):
                            skipping = False
                        continue
                    sliced.append(result)
                return sliced if not skipping else list(results)
            vector_id, rank_str = cursor.rsplit(":", 1)
            target_rank = int(rank_str)
        except (ValueError, TypeError):
            return list(results)

        sliced: List[HybridSearchResult] = []
        skipping = True
        for result in results:
            if skipping:
                if result.vector_id == vector_id and result.fused_rank == target_rank:
                    skipping = False
                continue
            sliced.append(result)
        return sliced if not skipping else list(results)

    def _build_cursor(self, results: Sequence[HybridSearchResult], page_size: int) -> Optional[str]:
        if len(results) <= page_size:
            return None
        last = results[page_size - 1]
        return f"v2|{last.score:.6f}|{last.vector_id}"

    def _execute_bm25(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
        config: HybridSearchConfig,
        query_features: ChunkFeatures,
        timings: Dict[str, float],
    ) -> ChannelResults:
        start = time.perf_counter()
        use_true_bm25 = getattr(config.retrieval, "bm25_scoring", "compat") == "true" and hasattr(
            self._opensearch, "search_bm25_true"
        )
        if use_true_bm25:
            hits, _ = getattr(self._opensearch, "search_bm25_true")(
                query_features.bm25_terms,
                filters,
                top_k=config.retrieval.bm25_top_k,
                k1=getattr(config.retrieval, "bm25_k1", 1.2),
                b=getattr(config.retrieval, "bm25_b", 0.75),
            )
        else:
            hits, _ = self._opensearch.search_bm25(
                query_features.bm25_terms,
                filters,
                top_k=config.retrieval.bm25_top_k,
            )
        timings["bm25_ms"] = (time.perf_counter() - start) * 1000
        self._observability.metrics.observe(
            "search_channel_latency_ms", timings["bm25_ms"], channel="bm25"
        )
        p95_bm25 = self._observability.metrics.percentile(
            "search_channel_latency_ms", 0.95, channel="bm25"
        )
        if p95_bm25 is not None:
            self._observability.metrics.set_gauge(
                "search_channel_latency_p95_ms", p95_bm25, channel="bm25"
            )
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
        self._observability.metrics.observe(
            "search_channel_latency_ms", timings["splade_ms"], channel="splade"
        )
        p95_splade = self._observability.metrics.percentile(
            "search_channel_latency_ms", 0.95, channel="splade"
        )
        if p95_splade is not None:
            self._observability.metrics.set_gauge(
                "search_channel_latency_p95_ms", p95_splade, channel="splade"
            )
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
        store: DenseVectorStore,
    ) -> ChannelResults:
        start = time.perf_counter()
        # Over-fetch relative to the page size (not dense_top_k) to leave headroom for filtering.
        page_size = max(1, request.page_size)
        retrieval_cfg = config.retrieval
        signature = self._dense_request_signature(request, filters)
        oversample = float(
            getattr(retrieval_cfg, "dense_oversample", getattr(config.dense, "oversample", 1.0))
        )
        oversample = max(1.0, oversample)
        overfetch = float(getattr(retrieval_cfg, "dense_overfetch_factor", 1.5))
        overfetch = max(1.0, overfetch)
        pass_rate = max(1e-3, min(1.0, float(self._dense_filter_pass_ema)))
        base = math.ceil(page_size * oversample * overfetch)
        adaptive = math.ceil(page_size / pass_rate * oversample * overfetch)
        k = max(int(retrieval_cfg.dense_top_k), base, adaptive)
        k = min(k, 10_000)
        cached_k = self._dense_lookup_cached_k(signature)
        if cached_k is not None:
            k = cached_k
        queries = np.asarray([query_features.embedding], dtype=np.float32)

        def run_dense_search(current_k: int) -> list[FaissSearchResult]:
            batch_hits_local = store.search_batch(queries, current_k)
            self._observability.metrics.observe(
                "faiss_search_batch_size", float(len(batch_hits_local)), channel="dense"
            )
            return batch_hits_local[0] if batch_hits_local else []

        effective_k = k
        hits = run_dense_search(effective_k)
        filtered, payloads = self._filter_dense_hits(hits, filters)
        observed = (len(filtered) / max(1, len(hits))) if hits else 0.0
        blended_pass = self._blend_pass_rate(observed)

        if len(filtered) < page_size and effective_k < 10_000:
            retry_k = min(
                10_000,
                max(
                    int(retrieval_cfg.dense_top_k),
                    base,
                    math.ceil(page_size / max(1e-3, blended_pass) * oversample * overfetch),
                ),
            )
            if cached_k is not None and cached_k > retry_k:
                retry_k = cached_k
            if retry_k > effective_k:
                effective_k = retry_k
                hits = run_dense_search(effective_k)
                filtered, payloads = self._filter_dense_hits(hits, filters)
                observed = (len(filtered) / max(1, len(hits))) if hits else 0.0
                blended_pass = self._blend_pass_rate(observed)

        self._dense_filter_pass_ema = blended_pass
        self._observability.metrics.set_gauge(
            "dense_filter_pass_rate", self._dense_filter_pass_ema, channel="dense"
        )
        timings["dense_ms"] = (time.perf_counter() - start) * 1000
        self._observability.metrics.observe(
            "search_channel_latency_ms", timings["dense_ms"], channel="dense"
        )
        p95_dense = self._observability.metrics.percentile(
            "search_channel_latency_ms", 0.95, channel="dense"
        )
        if p95_dense is not None:
            self._observability.metrics.set_gauge(
                "search_channel_latency_p95_ms", p95_dense, channel="dense"
            )
        self._observability.metrics.set_gauge("dense_effective_k", float(effective_k), channel="dense")
        self._observability.metrics.increment("search_channel_requests", channel="dense")
        self._observability.metrics.observe(
            "search_channel_candidates", len(filtered), channel="dense"
        )
        self._dense_store_cached_k(signature, effective_k)
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

    def _dense_request_signature(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
    ) -> tuple[object, ...]:
        normalized_filters = self._normalize_signature_value(filters)
        cursor = getattr(request, "cursor", None)
        return (
            request.query,
            request.namespace,
            normalized_filters,
            int(request.page_size),
            cursor,
        )

    def _dense_lookup_cached_k(self, signature: tuple[object, ...]) -> Optional[int]:
        with self._dense_k_cache_lock:
            cached = self._dense_k_cache.get(signature)
            if cached is not None:
                self._dense_k_cache.move_to_end(signature)
            return cached

    def _dense_store_cached_k(self, signature: tuple[object, ...], k: int) -> None:
        with self._dense_k_cache_lock:
            self._dense_k_cache[signature] = k
            self._dense_k_cache.move_to_end(signature)
            if len(self._dense_k_cache) > self._dense_k_cache_limit:
                self._dense_k_cache.popitem(last=False)

    def _normalize_signature_value(self, value: object) -> object:
        if isinstance(value, Mapping):
            return tuple(
                (str(key), self._normalize_signature_value(val))
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(self._normalize_signature_value(val) for val in value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _blend_pass_rate(self, observed: float) -> float:
        clamped_observed = max(0.0, min(1.0, float(observed)))
        blended = (
            self._dense_filter_ema_alpha * clamped_observed
            + (1.0 - self._dense_filter_ema_alpha) * self._dense_filter_pass_ema
        )
        return max(1e-3, min(1.0, blended))

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
            key=lambda candidate: (
                -fused_scores.get(candidate.chunk.vector_id, 0.0),
                candidate.chunk.vector_id,
            ),
        )


class HybridSearchAPI:
    """Minimal synchronous handler for ``POST /v1/hybrid-search``.

    Attributes:
        _service: Underlying :class:`HybridSearchService` instance.

    Examples:
        >>> api = HybridSearchAPI(service)  # doctest: +SKIP
        >>> status, body = api.post_hybrid_search({"query": "example"})  # doctest: +SKIP
        >>> status
        200
    """

    def __init__(self, service: HybridSearchService) -> None:
        """Initialise the API facade.

        Args:
            service: Hybrid search service used to satisfy requests.

        Raises:
            TypeError: If ``service`` does not inherit from
                :class:`HybridSearchService`.

        Returns:
            None
        """
        if not isinstance(service, HybridSearchService):
            raise TypeError("service must be a HybridSearchService instance")
        self._service = service

    def post_hybrid_search(self, payload: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        """Process a synchronous hybrid search HTTP-style request payload.

        Args:
            payload: JSON-like mapping containing the hybrid search request body.

        Returns:
            tuple[int, Mapping[str, Any]]: HTTP status code and serialized response body.
        """
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


@dataclass(slots=True)
class PaginationCheckResult:
    """Result of a pagination verification run.

    Attributes:
        cursor_chain: Sequence of pagination cursors encountered.
        duplicate_detected: True when duplicate results were observed.

    Examples:
        >>> result = PaginationCheckResult(cursor_chain=["cursor1"], duplicate_detected=False)
        >>> result.duplicate_detected
        False
    """

    cursor_chain: Sequence[str]
    duplicate_detected: bool


# --- Public Functions ---


def build_stats_snapshot(
    faiss_index: DenseVectorStore,
    opensearch: LexicalIndex,
    registry: ChunkRegistry,
) -> Mapping[str, object]:
    """Capture a lightweight snapshot of hybrid search storage metrics.

    Args:
        faiss_index: Dense vector index manager.
        opensearch: Lexical index representing sparse storage.
        registry: Chunk registry tracking vector-to-payload mappings.

    Returns:
        Mapping describing FAISS stats, OpenSearch stats, and chunk counts.
    """

    return {
        "faiss": faiss_index.stats(),
        "opensearch": opensearch.stats(),
        "registry": {"chunks": registry.count()},
    }


def verify_pagination(
    service: HybridSearchService, request: HybridSearchRequest
) -> PaginationCheckResult:
    """Ensure pagination cursors produce non-duplicated results.

    Args:
        service: Hybrid search service to execute paginated queries.
        request: Initial hybrid search request payload.

    Returns:
        PaginationCheckResult detailing encountered cursors and duplicates.
    """

    seen: set[tuple[str, str]] = set()
    cursor_chain: list[str] = []
    next_request = request
    duplicate = False
    seen_cursors: set[str] = set()

    while True:
        response = service.search(next_request)
        for result in response.results:
            key = (result.doc_id, result.chunk_id)
            if key in seen:
                duplicate = True
            seen.add(key)

        next_cursor = response.next_cursor
        if not next_cursor or next_cursor in seen_cursors:
            break

        cursor_chain.append(next_cursor)
        seen_cursors.add(next_cursor)
        next_request = HybridSearchRequest(
            query=request.query,
            namespace=request.namespace,
            filters=request.filters,
            page_size=request.page_size,
            cursor=next_cursor,
            diversification=request.diversification,
            diagnostics=request.diagnostics,
        )

    return PaginationCheckResult(cursor_chain=cursor_chain, duplicate_detected=duplicate)


def should_rebuild_index(
    registry: ChunkRegistry, deleted_since_snapshot: int, threshold: float = 0.2
) -> bool:
    """Heuristic to determine when FAISS should be rebuilt after deletions.

    Args:
        registry: Chunk registry reflecting current vector count.
        deleted_since_snapshot: Number of vectors deleted since the last snapshot.
        threshold: Fraction of deletions that triggers a rebuild.

    Returns:
        True when the proportion of deletions exceeds ``threshold``.
    """

    total = registry.count() + deleted_since_snapshot
    if total == 0:
        return False
    return deleted_since_snapshot / total >= threshold
