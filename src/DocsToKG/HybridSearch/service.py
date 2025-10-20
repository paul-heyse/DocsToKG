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
#       "id": "densesearchstrategy",
#       "name": "DenseSearchStrategy",
#       "anchor": "class-densesearchstrategy",
#       "kind": "class"
#     },
#     {
#       "id": "channelresults",
#       "name": "ChannelResults",
#       "anchor": "class-channelresults",
#       "kind": "class"
#     },
#     {
#       "id": "reciprocalrankfusion",
#       "name": "ReciprocalRankFusion",
#       "anchor": "class-reciprocalrankfusion",
#       "kind": "class"
#     },
#     {
#       "id": "resultshaper",
#       "name": "ResultShaper",
#       "anchor": "class-resultshaper",
#       "kind": "class"
#     },
#     {
#       "id": "apply-mmr-diversification",
#       "name": "apply_mmr_diversification",
#       "anchor": "function-apply-mmr-diversification",
#       "kind": "function"
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
#     },
#     {
#       "id": "hybridsearchvalidator",
#       "name": "HybridSearchValidator",
#       "anchor": "class-hybridsearchvalidator",
#       "kind": "class"
#     },
#     {
#       "id": "load-dataset",
#       "name": "load_dataset",
#       "anchor": "function-load-dataset",
#       "kind": "function"
#     },
#     {
#       "id": "infer-embedding-dim",
#       "name": "infer_embedding_dim",
#       "anchor": "function-infer-embedding-dim",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Hybrid-search orchestration, fusion, pagination guards, and API surface.

`service.py` binds the ingestion/storage primitives into the synchronous search
flow described in the README (“Search API contract”). It owns:

- Request validation (`HybridSearchValidator`, `HybridSearchAPI`) including
  namespace awareness, token budgets, and pagination cursors.
- Execution planning for dense + lexical searches, using configurable thread
  pools to issue FAISS GPU queries and BM25/SPLADE lookups concurrently.
- Score fusion (reciprocal rank fusion + optional MMR diversification) and
  response shaping, with per-channel breakdowns that mirror the JSON example in
  the README (including adaptive `fusion_weights` and per-request `stats`).
- Diagnostics: latency sampling, pagination verification, rebuild heuristics,
  and stats snapshots for health checks.

The module assumes dense search runs on the custom FAISS GPU wheel documented in
`faiss-gpu-wheel-reference.md`; when adjusting concurrency or stream behaviour,
ensure changes remain compatible with the `StandardGpuResources` guidance
summarised there.
"""

from __future__ import annotations

import atexit
import base64
import binascii
import hashlib
import json
import math
import random
import statistics
import time

# --- Strategy Helpers ---
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from threading import RLock
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

try:  # pragma: no cover - optional dependency during doc builds
    from .store import _ensure_cuvs_loader_path
except ImportError:  # pragma: no cover - doc builds may skip GPU dependencies
    _ensure_cuvs_loader_path = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised when GPU wheel present
    try:
        if _ensure_cuvs_loader_path is not None:
            _ensure_cuvs_loader_path()
    except Exception:
        # Defer to downstream FAISS import handling so doc builds remain resilient.
        pass

try:  # pragma: no cover - optional dependency during doc builds
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

from .config import (
    DenseIndexConfig,
    FusionConfig,
    HybridSearchConfig,
    HybridSearchConfigManager,
    RetrievalConfig,
)
from .devtools.opensearch_simulator import matches_filters
from .interfaces import DenseVectorStore, LexicalIndex
from .pipeline import ChunkIngestionPipeline, FeatureGenerator, Observability, tokenize
from .router import FaissRouter
from .store import (
    AdapterStats,
    ChunkRegistry,
    FaissSearchResult,
    ManagedFaissAdapter,
    OpenSearchSimulator,
    cosine_batch,
    cosine_topk_blockwise,
    normalize_rows,
    pairwise_inner_products,
    resolve_cuvs_state,
    restore_state,
    serialize_state,
)
from .types import (
    ChunkFeatures,
    ChunkPayload,
    DocumentInput,
    FusionCandidate,
    HybridSearchDiagnostics,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    ValidationReport,
    ValidationSummary,
)

# --- Globals ---

__all__ = (
    "ChannelResults",
    "DenseSearchStrategy",
    "ReciprocalRankFusion",
    "ResultShaper",
    "HybridSearchAPI",
    "HybridSearchService",
    "PaginationCheckResult",
    "RequestValidationError",
    "apply_mmr_diversification",
    "HybridSearchValidator",
    "load_dataset",
    "infer_embedding_dim",
    "build_stats_snapshot",
    "should_rebuild_index",
    "verify_pagination",
    "AdaptiveDensePlanner",
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


class DenseSearchStrategy:
    """Stateful helper encapsulating dense search heuristics."""

    _MAX_K = 2048

    def __init__(
        self,
        *,
        alpha: float = 0.20,
        initial_pass_rate: float = 0.75,
        cache_limit: int = 64,
        cache_path: Optional[Path] = None,
    ) -> None:
        self._alpha = float(alpha)
        self._pass_rate = max(1e-3, min(1.0, float(initial_pass_rate)))
        self._cache_limit = int(cache_limit)
        self._cache: "OrderedDict[Tuple[object, ...], int]" = OrderedDict()
        self._lock = RLock()
        self._signature_pass: Dict[Tuple[object, ...], float] = {}
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._dirty = False
        if self._cache_path is not None:
            self._load_cache()

    def plan(
        self,
        signature: Tuple[object, ...],
        *,
        page_size: int,
        retrieval_cfg: "RetrievalConfig",
        dense_cfg: "DenseIndexConfig",
        min_k: int = 0,
    ) -> tuple[int, float, float]:
        """Return the requested ``k`` and oversampling knobs for a dense search."""

        with self._lock:
            oversample = max(1.0, float(getattr(retrieval_cfg, "dense_oversample", 1.0)))
            legacy_over = getattr(dense_cfg, "oversample", None)
            if legacy_over is not None:
                try:
                    oversample = max(oversample, float(legacy_over))
                except Exception:
                    pass
            overfetch = max(1.0, float(getattr(retrieval_cfg, "dense_overfetch_factor", 1.5)))
            basis = max(1, int(page_size))
            base = math.ceil(basis * oversample * overfetch)
            adaptive = math.ceil(basis / max(1e-3, self._pass_rate) * oversample * overfetch)
            target = max(
                int(getattr(retrieval_cfg, "dense_top_k", basis)), base, adaptive, int(min_k)
            )
            cached = self._cache.get(signature)
            if cached is not None:
                target = max(target, int(cached))
                self._cache.move_to_end(signature)
            return (min(target, self._MAX_K), oversample, overfetch)

    def observe_pass_rate(
        self,
        signature: Tuple[object, ...],
        observed: float,
        *,
        update_global: bool = True,
    ) -> float:
        """Blend ``observed`` into the running EMA and return the updated value."""

        with self._lock:
            bounded = max(0.0, min(1.0, float(observed)))
            local_prev = self._signature_pass.get(signature, self._pass_rate)
            local_rate = max(
                1e-3, min(1.0, self._alpha * bounded + (1.0 - self._alpha) * local_prev)
            )
            self._signature_pass[signature] = local_rate
            if update_global:
                self._pass_rate = max(
                    1e-3,
                    min(1.0, self._alpha * bounded + (1.0 - self._alpha) * self._pass_rate),
                )
            self._dirty = True
            return local_rate

    def remember(self, signature: Tuple[object, ...], k: int) -> None:
        """Cache ``k`` for ``signature`` to seed future requests."""

        with self._lock:
            self._cache[signature] = int(k)
            self._cache.move_to_end(signature)
            if len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
            self._dirty = True

    def has_cache(self, signature: Tuple[object, ...]) -> bool:
        """Return ``True`` when ``signature`` has a cached ``k`` value."""

        with self._lock:
            return signature in self._cache

    def current_pass_rate(self) -> float:
        """Expose the current blended pass rate."""

        with self._lock:
            return self._pass_rate

    def persist(self) -> None:
        """Persist the adaptive cache to disk when a cache path is configured."""

        if self._cache_path is None:
            return
        with self._lock:
            if not self._dirty:
                return
            try:
                self._persist_cache()
            finally:
                self._dirty = False

    # --- Persistence helpers ---

    def _encode_signature(self, value: object) -> object:
        if isinstance(value, tuple):
            return {"__tuple__": [self._encode_signature(v) for v in value]}
        if isinstance(value, list):
            return [self._encode_signature(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._encode_signature(v) for k, v in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"__str__": str(value)}

    def _decode_signature(self, value: object) -> object:
        if isinstance(value, dict):
            if "__tuple__" in value:
                return tuple(self._decode_signature(v) for v in value["__tuple__"])
            if "__str__" in value:
                return value["__str__"]
            return {k: self._decode_signature(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._decode_signature(v) for v in value]
        return value

    def _persist_cache(self) -> None:
        if self._cache_path is None:
            return
        payload = {
            "cache": [
                {"signature": self._encode_signature(signature), "k": value}
                for signature, value in self._cache.items()
            ],
            "signature_pass": [
                {"signature": self._encode_signature(signature), "rate": value}
                for signature, value in self._signature_pass.items()
            ],
            "global_pass_rate": self._pass_rate,
        }
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:
            pass

    def _load_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            data = json.loads(self._cache_path.read_text())
        except FileNotFoundError:
            return
        except Exception:
            return
        cache_entries = data.get("cache", [])
        signature_pass_entries = data.get("signature_pass", [])
        pass_rate = data.get("global_pass_rate")
        with self._lock:
            for entry in cache_entries:
                try:
                    signature = self._decode_signature(entry["signature"])
                    if not isinstance(signature, tuple):
                        if isinstance(signature, list):
                            signature = tuple(signature)
                        else:
                            signature = (signature,)
                    value = int(entry["k"])
                except Exception:
                    continue
                self._cache[signature] = value
            for entry in signature_pass_entries:
                try:
                    signature = self._decode_signature(entry["signature"])
                    if not isinstance(signature, tuple):
                        if isinstance(signature, list):
                            signature = tuple(signature)
                        else:
                            signature = (signature,)
                    rate = float(entry["rate"])
                except Exception:
                    continue
                self._signature_pass[signature] = rate
            if isinstance(pass_rate, (int, float)):
                self._pass_rate = max(1e-3, min(1.0, float(pass_rate)))
            self._dirty = False


class AdaptiveDensePlanner(DenseSearchStrategy):
    """Backward-compatible alias that preserves the old planner name."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


@dataclass(slots=True)
class ChannelResults:
    """Results from a single retrieval channel (BM25, SPLADE, or dense).

    ``embeddings`` stores an optional matrix aligned with ``candidates`` for
    downstream GPU deduplication and diversification reuse.
    """

    candidates: List[FusionCandidate]
    scores: Dict[str, float]
    embeddings: Optional[np.ndarray] = None


class ReciprocalRankFusion:
    """Combine ranked lists using Reciprocal Rank Fusion."""

    def __init__(
        self, k0: float = 60.0, *, channel_weights: Mapping[str, float] | None = None
    ) -> None:
        if k0 <= 0:
            raise ValueError("k0 must be positive")
        self._k0 = k0
        self._weights = dict(channel_weights or {})

    def fuse(self, candidates: Sequence[FusionCandidate]) -> Dict[str, float]:
        """Combine channel rankings into fused scores keyed by vector id.

        Args:
            candidates: Ranked fusion candidates from individual retrieval channels.

        Returns:
            Mapping from vector identifiers to fused RRF scores.
        """
        scores: Dict[str, float] = defaultdict(float)
        for candidate in candidates:
            weight = float(self._weights.get(candidate.source, 1.0))
            contribution = weight * (1.0 / (self._k0 + candidate.rank))
            scores[candidate.chunk.vector_id] += contribution
        return dict(scores)


class ResultShaper:
    """Collapse duplicates, enforce quotas, and generate highlights."""

    def __init__(
        self,
        opensearch: LexicalIndex,
        fusion_config: FusionConfig,
        *,
        device: int = 0,
        resources: Optional["faiss.StandardGpuResources"] = None,
        channel_weights: Optional[Mapping[str, float]] = None,
        fp16_enabled: bool = False,
        cuvs_requested: Optional[bool] = None,
        registry: Optional[ChunkRegistry] = None,
    ) -> None:
        self._opensearch = opensearch
        self._fusion_config = fusion_config
        self._gpu_device = int(device)
        self._gpu_resources = resources
        self._channel_weights = dict(channel_weights or {})
        self._fp16_enabled = bool(fp16_enabled)
        self._use_cuvs = cuvs_requested
        self._registry = registry

    def shape(
        self,
        ordered_chunks: Sequence[ChunkPayload],
        fused_scores: Mapping[str, float],
        request: HybridSearchRequest,
        channel_scores: Optional[Mapping[str, Dict[str, float]]] = None,
        *,
        precomputed_embeddings: Optional[np.ndarray] = None,
    ) -> List[HybridSearchResult]:
        """Shape ranked chunks into API responses with highlights and diagnostics.

        Args:
            ordered_chunks: Chunks ordered by fused score.
            fused_scores: Combined score per vector identifier.
            request: Incoming hybrid search request.
            channel_scores: Optional per-channel score maps for diagnostics
                emission. Ignored when diagnostics are disabled on the request.
            precomputed_embeddings: Optional dense embeddings aligned with chunks.

        Returns:
            List of `HybridSearchResult` instances ready for serialization.
        """
        if not ordered_chunks:
            return []

        include_diagnostics = bool(request.diagnostics)
        score_lookup: Mapping[str, Dict[str, float]] = channel_scores or {}

        if precomputed_embeddings is not None:
            embeddings = np.ascontiguousarray(precomputed_embeddings, dtype=np.float32)
        else:
            if self._registry is None:
                raise RuntimeError(
                    "ResultShaper requires a registry to reconstruct embeddings when "
                    "precomputed embeddings are not supplied"
                )
            embeddings = self._registry.resolve_embeddings(
                [chunk.vector_id for chunk in ordered_chunks]
            )

        doc_buckets: Dict[str, int] = defaultdict(int)
        emitted_indices: List[int] = []
        results: List[HybridSearchResult] = []
        query_tokens = tokenize(request.query)
        token_budget = int(self._fusion_config.token_budget)
        byte_budget = int(self._fusion_config.byte_budget)
        tokens_used = 0
        bytes_used = 0

        for current_idx, chunk in enumerate(ordered_chunks):
            rank = current_idx + 1
            if not self._within_doc_limit(chunk.doc_id, doc_buckets):
                continue
            if emitted_indices and self._is_near_duplicate(
                embeddings, current_idx, emitted_indices
            ):
                continue
            highlights = self._build_highlights(chunk, query_tokens)
            chunk_tokens = int(getattr(chunk, "token_count", 0))
            chunk_bytes = len(chunk.text.encode("utf-8"))
            if token_budget and tokens_used + chunk_tokens > token_budget:
                break
            if byte_budget and bytes_used + chunk_bytes > byte_budget:
                break
            diagnostics: Optional[HybridSearchDiagnostics]
            if include_diagnostics:
                diagnostics = HybridSearchDiagnostics(
                    bm25_score=score_lookup.get("bm25", {}).get(chunk.vector_id),
                    splade_score=score_lookup.get("splade", {}).get(chunk.vector_id),
                    dense_score=score_lookup.get("dense", {}).get(chunk.vector_id),
                    fusion_weights=(dict(self._channel_weights) if self._channel_weights else None),
                )
            else:
                diagnostics = None
            doc_buckets[chunk.doc_id] += 1
            results.append(
                HybridSearchResult(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector_id=chunk.vector_id,
                    namespace=chunk.namespace,
                    score=fused_scores[chunk.vector_id],
                    fused_rank=rank,
                    text=chunk.text,
                    highlights=highlights,
                    diagnostics=diagnostics,
                    provenance_offsets=[chunk.char_offset] if chunk.char_offset else [],
                    metadata=dict(chunk.metadata),
                )
            )
            emitted_indices.append(current_idx)
            tokens_used += chunk_tokens
            bytes_used += chunk_bytes
        return results

    def _within_doc_limit(self, doc_id: str, doc_buckets: Mapping[str, int]) -> bool:
        return doc_buckets.get(doc_id, 0) < self._fusion_config.max_chunks_per_doc

    def _is_near_duplicate(
        self,
        embeddings: np.ndarray,
        current_idx: int,
        emitted_indices: Sequence[int],
    ) -> bool:
        if not emitted_indices:
            return False
        resources = self._gpu_resources
        query = embeddings[current_idx]
        corpus = embeddings[list(emitted_indices)]
        if resources is not None:
            top1, _ = cosine_topk_blockwise(
                np.asarray(query, dtype=np.float32).reshape(1, -1),
                corpus.astype(np.float32, copy=False),
                k=1,
                device=self._gpu_device,
                resources=resources,
                use_fp16=self._fp16_enabled,
                use_cuvs=self._use_cuvs,
            )
            return float(top1[0, 0]) >= self._fusion_config.cosine_dedupe_threshold
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return False
        other_norms = np.linalg.norm(corpus, axis=1)
        other_norms[other_norms == 0.0] = 1.0
        sims = (corpus @ query) / (other_norms * query_norm)
        return float(sims.max()) >= self._fusion_config.cosine_dedupe_threshold

    def _build_highlights(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        highlights = self._opensearch.highlight(chunk, query_tokens)
        if highlights:
            return list(highlights)
        if getattr(self._fusion_config, "strict_highlights", True):
            return []
        text = chunk.text[: min(len(chunk.text), 200)]
        return [text] if text else []


def apply_mmr_diversification(
    fused_candidates: Sequence[FusionCandidate],
    fused_scores: Mapping[str, float],
    lambda_param: float,
    top_k: int,
    *,
    embeddings: Optional[np.ndarray] = None,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
    use_fp16: bool = False,
    block_rows: int = 4096,
    use_cuvs: Optional[bool] = None,
    registry: Optional[ChunkRegistry] = None,
) -> List[FusionCandidate]:
    """Diversify fused candidates using Maximum Marginal Relevance.

    Args:
        fused_candidates: Ranked candidates in fused order.
        fused_scores: Combined scores keyed by vector identifier.
        lambda_param: Trade-off between relevance and diversity (0.0-1.0).
        top_k: Maximum number of diversified candidates to retain.
        embeddings: Optional dense embedding matrix aligned with ``fused_candidates``.
        device: GPU device id when leveraging FAISS GPU routines.
        resources: Optional FAISS GPU resources handle reused across calls.
        block_rows: Corpus rows processed per block when estimating diversity on GPU.
        use_cuvs: Optional override that controls cuVS usage during GPU similarity
            lookups. ``None`` defers to runtime detection; ``True`` forces cuVS when
            supported and ``False`` disables it.

    Returns:
        List of diversified `FusionCandidate` objects.
    """
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be within [0, 1]")
    if not fused_candidates:
        return []

    if embeddings is None:
        if registry is None:
            raise ValueError(
                "apply_mmr_diversification requires either precomputed embeddings or a registry"
            )
        embeddings = registry.resolve_embeddings(
            [candidate.chunk.vector_id for candidate in fused_candidates]
        )
    else:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    total = embeddings.shape[0]
    if total <= 0:
        return []
    if total <= top_k:
        return list(fused_candidates[:total])

    scores = np.array(
        [fused_scores.get(candidate.chunk.vector_id, 0.0) for candidate in fused_candidates],
        dtype=np.float32,
    )
    selected: List[int] = []
    remaining = np.arange(total, dtype=np.int64)
    max_sim = np.zeros(total, dtype=np.float32)
    neighbor_lookup: Optional[List[Dict[int, float]]] = None

    def _cosine_cpu(query: np.ndarray, pool: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(pool, axis=1)
        norms[norms == 0.0] = 1.0
        query_norm = np.linalg.norm(query) or 1.0
        return (pool @ query) / (norms * query_norm)

    if resources is not None:
        try:
            top_k_neighbors = min(total, 128)
            scores_block, indices_block = cosine_topk_blockwise(
                embeddings,
                embeddings,
                k=top_k_neighbors,
                device=device,
                resources=resources,
                block_rows=block_rows,
                use_fp16=use_fp16,
                use_cuvs=use_cuvs,
            )
        except Exception:
            scores_block = indices_block = None
        if scores_block is not None and indices_block is not None:
            neighbor_lookup = []
            for row in range(total):
                row_scores = scores_block[row]
                row_indices = indices_block[row]
                row_map: Dict[int, float] = {}
                for score, idx in zip(row_scores, row_indices):
                    candidate_idx = int(idx)
                    if candidate_idx < 0 or candidate_idx == row:
                        continue
                    row_map[candidate_idx] = float(score)
                neighbor_lookup.append(row_map)

    while len(selected) < top_k and remaining.size:
        if not selected:
            idx = int(np.argmax(scores[remaining]))
            best_idx = int(remaining[idx])
        else:
            relevance = scores[remaining]
            diversity = max_sim[remaining]
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            best_idx = int(remaining[int(np.argmax(mmr))])

        selected.append(best_idx)
        remaining = remaining[remaining != best_idx]
        if not remaining.size:
            break

        query_vec = np.ascontiguousarray(embeddings[best_idx], dtype=np.float32)
        pool = np.ascontiguousarray(embeddings[remaining], dtype=np.float32)
        if resources is not None:
            if neighbor_lookup is not None:
                sims = np.full(pool.shape[0], -np.inf, dtype=np.float32)
                neighbor_scores = neighbor_lookup[best_idx]
                for pos, idx in enumerate(remaining):
                    score = neighbor_scores.get(int(idx))
                    if score is not None:
                        sims[pos] = score
                missing_mask = sims == -np.inf
                if np.any(missing_mask):
                    gpu_scores = cosine_batch(
                        query_vec.reshape(1, -1),
                        pool,
                        device=device,
                        resources=resources,
                    )[0]
                    sims[missing_mask] = gpu_scores[missing_mask]
            else:
                sims = cosine_batch(
                    query_vec.reshape(1, -1),
                    pool,
                    device=device,
                    resources=resources,
                )[0]
        else:
            sims = _cosine_cpu(query_vec, pool)
        max_sim[remaining] = np.maximum(max_sim[remaining], sims)

    return [fused_candidates[idx] for idx in selected]


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
        >>> from DocsToKG.HybridSearch.interfaces import DenseVectorStore  # doctest: +SKIP
        >>> from DocsToKG.HybridSearch.router import FaissRouter  # doctest: +SKIP
        >>> from DocsToKG.HybridSearch.store import FaissVectorStore, ManagedFaissAdapter, OpenSearchSimulator  # doctest: +SKIP
        >>> dense_store: DenseVectorStore = ManagedFaissAdapter(  # doctest: +SKIP
        ...     FaissVectorStore(dim=16, config=HybridSearchConfig().dense)
        ... )
        >>> router = FaissRouter(per_namespace=False, default_store=dense_store)  # doctest: +SKIP
        >>> service = HybridSearchService(  # doctest: +SKIP
        ...     config_manager=manager,
        ...     feature_generator=FeatureGenerator(embedding_dim=16),
        ...     faiss_index=dense_store,
        ...     opensearch=OpenSearchSimulator(),
        ...     registry=ChunkRegistry(),
        ...     faiss_router=router,
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
        dense_strategy_cache_path: Optional[Path] = None,
    ) -> None:
        """Initialise the hybrid search service.

        Args:
            config_manager: Manager providing the latest search configuration.
            feature_generator: Component responsible for feature extraction.
            faiss_index: Dense retrieval index manager (GPU-backed).
            opensearch: Lexical index providing BM25/SPLADE access.
            registry: Chunk registry used for metadata lookups.
            observability: Optional observability facade (defaults to a no-op).
            dense_strategy_cache_path: Optional path used to persist dense planner heuristics.

        Returns:
            None
        """
        self._config_manager = config_manager
        config = self._config_manager.get()
        self._feature_generator = feature_generator
        self._opensearch = opensearch
        self._registry = registry
        self._observability = observability or Observability()
        cache_path = (
            Path(dense_strategy_cache_path).expanduser()
            if dense_strategy_cache_path is not None
            else None
        )
        self._assert_managed_store(faiss_index)
        if faiss_router is not None:
            self._faiss_router = faiss_router
        else:
            self._faiss_router = FaissRouter(per_namespace=False, default_store=faiss_index)
        self._faiss = self._faiss_router.default_store
        self._assert_managed_store(self._faiss)
        self._faiss_router.set_resolver(self._registry.resolve_faiss_id)
        self._dense_strategy = DenseSearchStrategy(cache_path=cache_path)
        max_workers = config.retrieval.executor_max_workers or 3
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        schema_manager = None
        for attr in ("schema_manager", "schema", "_schema"):
            candidate = getattr(self._opensearch, attr, None)
            if candidate is not None:
                schema_manager = candidate
                break
        if schema_manager is None and hasattr(self._opensearch, "validate_namespace_schema"):
            schema_manager = self._opensearch
        if schema_manager is not None and hasattr(schema_manager, "validate_namespace_schema"):
            try:
                namespaces: Sequence[str] = ()
                if hasattr(schema_manager, "list_templates"):
                    templates = schema_manager.list_templates()  # type: ignore[call-arg]
                    if isinstance(templates, Mapping):
                        namespaces = list(templates.keys())
                if not namespaces and hasattr(schema_manager, "get_template"):
                    default_ns = "default"
                    template = schema_manager.get_template(default_ns)  # type: ignore[call-arg]
                    if template is not None:
                        namespaces = [default_ns]
                for namespace in namespaces:
                    schema_manager.validate_namespace_schema(namespace)  # type: ignore[attr-defined]
            except Exception:
                self._observability.logger.exception("lexical-schema-validation-warning")
        self._closed = False
        atexit.register(self.close)

    def close(self) -> None:
        """Release pooled resources held by the service."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._flush_dense_snapshots()
        except Exception:
            self._observability.logger.exception("dense-snapshot-flush-failed")
        try:
            self._dense_strategy.persist()
        except Exception:
            self._observability.logger.exception("dense-strategy-persist-failed")
        self._executor.shutdown(wait=False)

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
        cursor_fingerprint = self._cursor_fingerprint(request, filters)
        recall_first = bool(getattr(request, "recall_first", False))

        with self._observability.trace("hybrid_search", namespace=request.namespace or "*"):
            timings: Dict[str, float] = {}
            total_start = time.perf_counter()
            query_start = time.perf_counter()
            query_features = self._feature_generator.compute_features(request.query)
            timings["feature_ms"] = (time.perf_counter() - query_start) * 1000

            dense_store = self._dense_store(request.namespace)
            try:
                adapter_stats = dense_store.adapter_stats  # type: ignore[attr-defined]
            except AttributeError:
                adapter_stats = None
            store_resources = getattr(dense_store, "get_gpu_resources", lambda: None)()
            resources_hint = (
                adapter_stats.resources if adapter_stats is not None else store_resources
            )
            device_hint = (
                adapter_stats.device
                if adapter_stats is not None
                else getattr(dense_store, "device", 0)
            )
            fp16_hint = (
                bool(adapter_stats.fp16_enabled)
                if adapter_stats is not None
                else bool(getattr(getattr(dense_store, "config", object()), "flat_use_fp16", False))
            )
            cuvs_requested = getattr(config.dense, "use_cuvs", None)
            if adapter_stats is not None:
                cuvs_enabled = bool(adapter_stats.cuvs_enabled)
                cuvs_available = bool(adapter_stats.cuvs_available)
                cuvs_reported = adapter_stats.cuvs_reported
            else:
                cuvs_enabled, cuvs_available, cuvs_reported = resolve_cuvs_state(cuvs_requested)
            self._observability.metrics.set_gauge(
                "faiss_cuvs_enabled", 1.0 if cuvs_enabled else 0.0, channel="dense"
            )
            self._observability.metrics.set_gauge(
                "faiss_cuvs_available", 1.0 if cuvs_available else 0.0, channel="dense"
            )
            self._observability.logger.debug(
                "faiss-cuvs-state",
                extra={
                    "event": {
                        "requested": cuvs_requested,
                        "enabled": cuvs_enabled,
                        "available": cuvs_available,
                        "reported_available": cuvs_reported,
                    }
                },
            )

            f_bm25 = self._executor.submit(
                self._execute_bm25, request, filters, config, query_features, timings
            )
            f_splade = self._executor.submit(
                self._execute_splade, request, filters, config, query_features, timings
            )
            f_dense = self._executor.submit(
                self._execute_dense,
                request,
                filters,
                config,
                query_features,
                timings,
                dense_store,
            )
            futures = {"bm25": f_bm25, "splade": f_splade, "dense": f_dense}
            channel_results: Dict[str, ChannelResults] = {}
            failed_channel: Optional[str] = None
            try:
                for channel_name in ("bm25", "splade", "dense"):
                    failed_channel = channel_name
                    channel_results[channel_name] = futures[channel_name].result()
            except Exception as exc:
                for name, future in futures.items():
                    if name != failed_channel and not future.done():
                        future.cancel()
                namespace = request.namespace or "*"
                error_message = str(exc)
                error_details = (
                    f"{type(exc).__name__}: {error_message}"
                    if error_message
                    else type(exc).__name__
                )
                self._observability.logger.exception(
                    "hybrid-search-channel-error",
                    extra={
                        "event": {
                            "channel": failed_channel,
                            "namespace": namespace,
                            "filters": filters,
                            "query": request.query,
                            "error": error_details,
                        }
                    },
                )
                if failed_channel:
                    message = (
                        "Hybrid search channel '"
                        f"{failed_channel}' failed for namespace '{namespace}': "
                        f"{error_details}"
                    )
                else:
                    message = (
                        "Hybrid search channel failed for namespace '"
                        f"{namespace}': {error_details}"
                    )
                raise RequestValidationError(message) from exc
            bm25 = channel_results["bm25"]
            splade = channel_results["splade"]
            dense = channel_results["dense"]

            embedding_cache: Dict[str, np.ndarray] = {}
            if dense.embeddings is not None:
                for candidate, row in zip(dense.candidates, dense.embeddings):
                    embedding_cache[candidate.chunk.vector_id] = row

            def _resolve_embedding(candidate: FusionCandidate) -> np.ndarray:
                vector_id = candidate.chunk.vector_id
                cached = embedding_cache.get(vector_id)
                if cached is not None:
                    return cached
                embedding = self._registry.resolve_embedding(vector_id, cache=embedding_cache)
                return embedding

            raw_weights = getattr(config.fusion, "channel_weights", None)
            adaptive_weights = dict(raw_weights) if raw_weights is not None else {}
            lexical_ids = {candidate.chunk.vector_id for candidate in bm25.candidates}
            lexical_ids.update(candidate.chunk.vector_id for candidate in splade.candidates)
            lexical_rate = min(1.0, len(lexical_ids) / max(1, request.page_size))
            dense_weight = adaptive_weights.get("dense", 1.0)
            if lexical_rate >= 0.8:
                dense_weight = max(0.5, dense_weight * 0.7)
            elif lexical_rate <= 0.3:
                dense_weight = min(2.0, dense_weight * 1.25)
            adaptive_weights["dense"] = dense_weight
            self._observability.metrics.observe(
                "hybrid_lexical_hit_rate",
                lexical_rate,
                namespace=request.namespace or "*",
            )
            fusion = ReciprocalRankFusion(k0=config.fusion.k0, channel_weights=adaptive_weights)
            combined_candidates = bm25.candidates + splade.candidates + dense.candidates
            fused_scores = fusion.fuse(combined_candidates)
            unique_candidates = self._dedupe_candidates(combined_candidates, fused_scores)

            if unique_candidates:
                unique_embeddings = np.ascontiguousarray(
                    np.stack([_resolve_embedding(candidate) for candidate in unique_candidates]),
                    dtype=np.float32,
                )
                normalize_rows(unique_embeddings)
                embedding_index = {
                    candidate.chunk.vector_id: idx
                    for idx, candidate in enumerate(unique_candidates)
                }
            else:
                unique_embeddings = None
                embedding_index = {}

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
                pool_embeddings = (
                    unique_embeddings[:pool_size] if unique_embeddings is not None else None
                )
                device_id = device_hint
                resources = resources_hint
                fp16_enabled = fp16_hint
                selected = apply_mmr_diversification(
                    pool,
                    fused_scores,
                    config.fusion.mmr_lambda,
                    desired,
                    embeddings=pool_embeddings,
                    device=device_id,
                    resources=resources,
                    use_fp16=fp16_enabled,
                    use_cuvs=cuvs_requested,
                    registry=self._registry,
                )
                selected_ids = {candidate.chunk.vector_id for candidate in selected}
                pool_remaining = [
                    candidate for candidate in pool if candidate.chunk.vector_id not in selected_ids
                ]
                tail = list(unique_candidates[pool_size:])
                diversified = [*selected, *pool_remaining, *tail]
            else:
                diversified = unique_candidates

            diversified = sorted(
                diversified,
                key=lambda cand: (
                    -fused_scores.get(cand.chunk.vector_id, 0.0),
                    cand.chunk.vector_id,
                ),
            )

            if diversified:
                if unique_embeddings is not None and embedding_index:
                    try:
                        diversified_embeddings = np.ascontiguousarray(
                            unique_embeddings[
                                [
                                    embedding_index[candidate.chunk.vector_id]
                                    for candidate in diversified
                                ]
                            ],
                            dtype=np.float32,
                        )
                    except KeyError:
                        diversified_embeddings = np.ascontiguousarray(
                            np.stack([_resolve_embedding(candidate) for candidate in diversified]),
                            dtype=np.float32,
                        )
                else:
                    diversified_embeddings = np.ascontiguousarray(
                        np.stack([_resolve_embedding(candidate) for candidate in diversified]),
                        dtype=np.float32,
                    )
            else:
                diversified_embeddings = None

            ordered_chunks = [candidate.chunk for candidate in diversified]
            channel_score_map: Optional[Mapping[str, Dict[str, float]]]
            if request.diagnostics:
                channel_score_map = {
                    "bm25": bm25.scores,
                    "splade": splade.scores,
                    "dense": dense.scores,
                }
            else:
                channel_score_map = None
            shaper = ResultShaper(
                self._opensearch,
                config.fusion,
                device=device_hint,
                resources=resources_hint,
                channel_weights=adaptive_weights,
                fp16_enabled=fp16_hint,
                cuvs_requested=cuvs_requested,
                registry=self._registry,
            )
            shaped = shaper.shape(
                ordered_chunks,
                fused_scores,
                request,
                channel_score_map,
                precomputed_embeddings=diversified_embeddings,
            )
            timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000
            timings["total_ms"] = (time.perf_counter() - total_start) * 1000
            self._observability.metrics.increment("hybrid_search_requests")
            self._observability.metrics.observe("hybrid_search_results", len(shaped))
            self._observability.metrics.observe(
                "hybrid_search_timings_total_ms", timings["total_ms"]
            )
            paged_results = self._slice_from_cursor(
                shaped,
                request.cursor,
                request.page_size,
                cursor_fingerprint,
                recall_first,
            )
            next_cursor = self._build_cursor(
                paged_results,
                request.page_size,
                cursor_fingerprint,
                recall_first,
            )
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
            stats_snapshot = build_stats_snapshot(dense_store, self._opensearch, self._registry)
            if adapter_stats is not None:
                faiss_stats = stats_snapshot.get("faiss")
                if isinstance(faiss_stats, Mapping):
                    faiss_stats = dict(faiss_stats)
                else:
                    faiss_stats = {}
                faiss_stats["nprobe_effective"] = adapter_stats.nprobe
                stats_snapshot["faiss"] = faiss_stats
            return HybridSearchResponse(
                results=page_results,
                next_cursor=next_cursor,
                total_candidates=len(unique_candidates),
                timings_ms=timings,
                fusion_weights=dict(adaptive_weights),
                stats=stats_snapshot,
            )

    def _validate_request(self, request: HybridSearchRequest) -> None:
        if not request.query.strip():
            raise RequestValidationError("Query must not be empty")
        if request.page_size <= 0:
            raise RequestValidationError("page_size must be positive")
        if request.page_size > 1000:
            raise RequestValidationError("page_size exceeds maximum")

    @staticmethod
    def _assert_managed_store(store: DenseVectorStore) -> None:
        unsafe_attrs = ("index", "_index", "gpu_resources")
        if isinstance(store, ManagedFaissAdapter):
            return
        if any(hasattr(store, attr) for attr in unsafe_attrs):
            raise TypeError("HybridSearchService requires a managed dense vector store")
        if not hasattr(store, "get_gpu_resources"):
            raise TypeError("Dense vector store must expose get_gpu_resources()")

    def _flush_dense_snapshots(self) -> None:
        router = getattr(self, "_faiss_router", None)
        if router is None:
            return
        iter_fn = getattr(router, "iter_stores", None)
        if callable(iter_fn):
            stores: Iterable[Tuple[str, DenseVectorStore]] = iter_fn()
        else:
            default_store = getattr(router, "default_store", None)
            if default_store is None:
                return
            stores = [("__default__", default_store)]
        for namespace, store in stores:
            flush_fn = getattr(store, "flush_snapshot", None)
            try:
                if callable(flush_fn):
                    flush_fn(reason="shutdown")
                else:
                    serializer = getattr(store, "serialize", None)
                    if callable(serializer):
                        serializer()
            except Exception:
                self._observability.logger.exception(
                    "dense-snapshot-flush-error",
                    extra={"event": {"namespace": namespace}},
                )

    def _dense_store(self, namespace: Optional[str]) -> DenseVectorStore:
        store = self._faiss_router.get(namespace)
        self._assert_managed_store(store)
        return store

    def _slice_from_cursor(
        self,
        results: Sequence[HybridSearchResult],
        cursor: Optional[str],
        page_size: int,
        fingerprint: str,
        recall_first: bool,
    ) -> List[HybridSearchResult]:
        if not cursor:
            return list(results)
        try:
            if cursor.startswith("v3|"):
                _, encoded = cursor.split("|", 1)
                padding = "=" * (-len(encoded) % 4)
                raw = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
                payload = json.loads(raw.decode("utf-8"))
                page_hint = int(payload.get("p"))
                if page_hint != page_size:
                    raise RequestValidationError("Cursor page size mismatch")
                if payload.get("f") != fingerprint:
                    raise RequestValidationError("Cursor fingerprint mismatch")
                stored_recall = payload.get("r")
                if stored_recall is not None and bool(stored_recall) != bool(recall_first):
                    raise RequestValidationError("Cursor recall-first mismatch")
                anchor = payload.get("a") or {}
                vector_id = anchor.get("id") or anchor.get("vector_id")
                score = anchor.get("score")
                rank = anchor.get("rank")
                if vector_id is None or score is None:
                    raise RequestValidationError("Cursor anchor missing data")
                return self._slice_after_anchor(
                    results,
                    str(vector_id),
                    float(score),
                    int(rank) if rank is not None else None,
                )
            if cursor.startswith("v2|"):
                _version, score_str, vector_id = cursor.split("|", 2)
                return self._slice_after_anchor(results, vector_id, float(score_str), None)
            vector_id, rank_str = cursor.rsplit(":", 1)
            return self._slice_after_anchor(results, vector_id, None, int(rank_str))
        except RequestValidationError:
            raise
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
            return list(results)

    def _build_cursor(
        self,
        results: Sequence[HybridSearchResult],
        page_size: int,
        fingerprint: str,
        recall_first: bool,
    ) -> Optional[str]:
        if len(results) <= page_size:
            return None
        last = results[page_size - 1]
        anchor = {
            "id": last.vector_id,
            "score": float(last.score),
            "rank": int(getattr(last, "fused_rank", page_size)),
        }
        payload = {
            "p": int(page_size),
            "f": fingerprint,
            "a": anchor,
            "r": bool(recall_first),
        }
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        encoded = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")
        return f"v3|{encoded}"

    def _slice_after_anchor(
        self,
        results: Sequence[HybridSearchResult],
        vector_id: str,
        score: Optional[float],
        rank: Optional[int],
    ) -> List[HybridSearchResult]:
        sliced: List[HybridSearchResult] = []
        skipping = True
        for result in results:
            if skipping:
                if result.vector_id != vector_id:
                    continue
                score_match = (
                    True if score is None or math.isnan(score) else abs(result.score - score) < 1e-6
                )
                rank_match = True if rank is None else result.fused_rank == rank
                if score_match and rank_match:
                    skipping = False
                continue
            sliced.append(result)
        return sliced if not skipping else list(results)

    def run_compaction_cycle(self) -> Dict[str, Dict[str, object]]:
        """Trigger FAISS maintenance (rebuild/compact) and emit diagnostics."""

        start = time.perf_counter()
        before_snapshot = self._faiss_router.stats()
        actions = self._faiss_router.run_maintenance()
        after_snapshot = self._faiss_router.stats()
        elapsed_ms = (time.perf_counter() - start) * 1000

        before = before_snapshot.get("namespaces", {})
        after = after_snapshot.get("namespaces", {})

        def _dirty(snapshot: Mapping[str, Mapping[str, object]], namespace: str) -> float:
            bucket = snapshot.get(namespace, {})
            if not isinstance(bucket, Mapping):
                return 0.0
            raw = bucket.get("dirty_deletes", 0.0)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        summary: Dict[str, Dict[str, object]] = {}
        for namespace, action in actions.items():
            before_dirty = _dirty(before, namespace)
            after_dirty = _dirty(after, namespace)
            rebuilt = bool(action.get("rebuilt"))
            trained = bool(action.get("trained"))
            summary[namespace] = {
                "rebuilt": rebuilt,
                "trained": trained,
                "dirty_deletes_before": before_dirty,
                "dirty_deletes_after": after_dirty,
            }
            self._observability.metrics.observe(
                "faiss_dirty_deletes", after_dirty, namespace=namespace
            )
            if rebuilt:
                self._observability.metrics.increment(
                    "faiss_compactions_rebuilt", namespace=namespace
                )

        self._observability.metrics.increment("faiss_compactions_attempted")
        self._observability.metrics.observe("faiss_compaction_ms", elapsed_ms)
        self._observability.logger.info(
            "hybrid-search-compaction",
            extra={
                "event": {
                    "elapsed_ms": round(elapsed_ms, 3),
                    "namespaces": summary,
                }
            },
        )
        return {
            "namespaces": summary,
            "elapsed_ms": elapsed_ms,
            "aggregate": after_snapshot.get("aggregate", {}),
        }

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
        strategy = self._dense_strategy
        cached_signature = strategy.has_cache(signature)
        initial_k, oversample, overfetch = strategy.plan(
            signature,
            page_size=page_size,
            retrieval_cfg=retrieval_cfg,
            dense_cfg=config.dense,
        )
        queries = np.asarray([query_features.embedding], dtype=np.float32)
        adapter_stats: Optional[AdapterStats]
        try:
            adapter_stats = store.adapter_stats  # type: ignore[attr-defined]
        except AttributeError:
            adapter_stats = None

        score_floor = float(getattr(config.retrieval, "dense_score_floor", 0.0))
        use_score_floor = score_floor > 0.0
        recall_first = bool(getattr(request, "recall_first", False))
        use_range = recall_first or use_score_floor
        if use_range:
            budget = max(1, int(initial_k))
            hits = list(store.range_search(queries[0], score_floor, limit=budget))
            self._observability.metrics.observe("faiss_search_batch_size", 1.0, channel="dense")
            filtered, payloads = self._filter_dense_hits(hits, filters, score_floor)
            observed = (len(filtered) / max(1, len(hits))) if hits else 0.0
            blended_pass = strategy.observe_pass_rate(
                signature,
                observed,
                update_global=not recall_first,
            )
            strategy.remember(signature, max(len(filtered), len(hits)))
            filtered.sort(key=lambda hit: (-hit.score, hit.vector_id))
            bounded_filtered = filtered[:budget]
            effective_k = len(hits)
            self._observability.metrics.set_gauge(
                "dense_pass_through_rate",
                float(observed),
                namespace=request.namespace or "*",
            )
            self._observability.metrics.set_gauge(
                "dense_filter_pass_rate", blended_pass, channel="dense"
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
            self._observability.metrics.set_gauge(
                "dense_effective_k", float(effective_k), channel="dense"
            )
            self._observability.metrics.increment("search_channel_requests", channel="dense")
            self._observability.metrics.observe(
                "search_channel_candidates", len(bounded_filtered), channel="dense"
            )
            if adapter_stats is not None:
                fp16_metric = 1.0 if bool(getattr(adapter_stats, "fp16_enabled", False)) else 0.0
                self._observability.metrics.set_gauge(
                    "faiss_fp16_enabled", fp16_metric, channel="dense"
                )
                self._observability.metrics.set_gauge(
                    "nprobe_in_effect", float(adapter_stats.nprobe), channel="dense"
                )
            candidates: List[FusionCandidate] = []
            scores: Dict[str, float] = {}
            embedding_cache_local: Dict[str, np.ndarray] = {}
            resolved_hits: List[tuple[FaissSearchResult, ChunkPayload]] = []
            for hit in bounded_filtered:
                chunk = payloads.get(hit.vector_id)
                if chunk is None:
                    continue
                resolved_hits.append((hit, chunk))

            vector_ids = [chunk.vector_id for _, chunk in resolved_hits]
            if vector_ids:
                embedding_matrix = self._registry.resolve_embeddings(
                    vector_ids, cache=embedding_cache_local
                )
            else:
                embedding_matrix = None

            for idx, (hit, chunk) in enumerate(resolved_hits):
                candidates.append(
                    FusionCandidate(source="dense", score=hit.score, chunk=chunk, rank=idx + 1)
                )
                scores[hit.vector_id] = hit.score
            return ChannelResults(candidates=candidates, scores=scores, embeddings=embedding_matrix)

        def run_dense_search(current_k: int) -> list[FaissSearchResult]:
            """Query FAISS for dense document candidates at the requested depth.

            Args:
                current_k: Number of vector matches to request from the dense index.

            Returns:
                Dense similarity matches ordered by score for the current document search.
            """
            depth = max(1, int(current_k))
            batch_hits_local = store.search_batch(queries, depth)
            self._observability.metrics.observe(
                "faiss_search_batch_size", float(len(batch_hits_local)), channel="dense"
            )
            hits_local = batch_hits_local[0] if batch_hits_local else []
            if use_score_floor:
                hits_local = [hit for hit in hits_local if hit.score >= score_floor]
            return hits_local

        effective_k = initial_k
        hits = run_dense_search(effective_k)
        filtered, payloads = self._filter_dense_hits(hits, filters, score_floor)
        observed = (len(filtered) / max(1, len(hits))) if hits else 0.0
        self._observability.metrics.set_gauge(
            "dense_pass_through_rate",
            float(observed),
            namespace=request.namespace or "*",
        )
        update_global = not bool(getattr(request, "recall_first", False))
        blended_pass = (
            strategy.observe_pass_rate(signature, observed, update_global=update_global)
            if not cached_signature
            else strategy.current_pass_rate()
        )

        while True:
            if len(filtered) >= page_size or effective_k >= 10_000:
                next_k = strategy.plan(
                    signature,
                    page_size=page_size,
                    retrieval_cfg=retrieval_cfg,
                    dense_cfg=config.dense,
                    min_k=effective_k,
                )[0]
            else:
                next_k, _, _ = strategy.plan(
                    signature,
                    page_size=page_size,
                    retrieval_cfg=retrieval_cfg,
                    dense_cfg=config.dense,
                    min_k=effective_k + 1,
                )
            if next_k <= effective_k:
                break
            effective_k = next_k
            hits = run_dense_search(effective_k)
            filtered, payloads = self._filter_dense_hits(hits, filters, score_floor)
            observed = (len(filtered) / max(1, len(hits))) if hits else 0.0
            blended_pass = strategy.observe_pass_rate(
                signature, observed, update_global=update_global
            )

        filtered.sort(key=lambda hit: (-hit.score, hit.vector_id))
        strategy.remember(signature, effective_k)
        self._observability.metrics.set_gauge(
            "dense_filter_pass_rate", blended_pass, channel="dense"
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
        self._observability.metrics.set_gauge(
            "dense_effective_k", float(effective_k), channel="dense"
        )
        self._observability.metrics.increment("search_channel_requests", channel="dense")
        self._observability.metrics.observe(
            "search_channel_candidates", len(filtered), channel="dense"
        )
        if adapter_stats is not None:
            fp16_metric = 1.0 if bool(getattr(adapter_stats, "fp16_enabled", False)) else 0.0
            self._observability.metrics.set_gauge(
                "faiss_fp16_enabled", fp16_metric, channel="dense"
            )
            self._observability.metrics.set_gauge(
                "nprobe_in_effect", float(adapter_stats.nprobe), channel="dense"
            )
        candidates: List[FusionCandidate] = []
        scores: Dict[str, float] = {}
        embedding_rows: List[np.ndarray] = []
        embedding_cache_local: Dict[str, np.ndarray] = {}
        for idx, hit in enumerate(filtered):
            chunk = payloads.get(hit.vector_id)
            if chunk is None:
                continue
            candidates.append(
                FusionCandidate(source="dense", score=hit.score, chunk=chunk, rank=idx + 1)
            )
            scores[hit.vector_id] = hit.score
            embedding_rows.append(
                self._registry.resolve_embedding(chunk.vector_id, cache=embedding_cache_local)
            )
        embedding_matrix: Optional[np.ndarray]
        if embedding_rows:
            embedding_matrix = np.ascontiguousarray(np.stack(embedding_rows), dtype=np.float32)
        else:
            embedding_matrix = None
        return ChannelResults(candidates=candidates, scores=scores, embeddings=embedding_matrix)

    def _filter_dense_hits(
        self,
        hits: Sequence[FaissSearchResult],
        filters: Mapping[str, object],
        score_floor: float,
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
            and float(hit.score) >= float(score_floor)
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
            bool(request.recall_first),
        )

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

    def _cursor_fingerprint(
        self,
        request: HybridSearchRequest,
        filters: Mapping[str, object],
    ) -> str:
        payload = {
            "q": request.query.strip(),
            "ns": request.namespace or "",
            "filters": self._normalize_signature_value(filters),
            "div": bool(getattr(request, "diversification", False)),
            "recall_first": bool(getattr(request, "recall_first", False)),
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.blake2s(raw, digest_size=12).hexdigest()

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

        results: List[Mapping[str, Any]] = []
        for result in response.results:
            item: Dict[str, Any] = {
                "doc_id": result.doc_id,
                "chunk_id": result.chunk_id,
                "namespace": result.namespace,
                "score": result.score,
                "fused_rank": result.fused_rank,
                "text": result.text,
                "highlights": list(result.highlights),
                "provenance_offsets": [list(offset) for offset in result.provenance_offsets],
                "metadata": dict(result.metadata),
            }
            diagnostics = getattr(result, "diagnostics", None)
            if request.diagnostics and diagnostics is not None:
                item["diagnostics"] = {
                    "bm25": getattr(diagnostics, "bm25_score", None),
                    "splade": getattr(diagnostics, "splade_score", None),
                    "dense": getattr(diagnostics, "dense_score", None),
                    "fusion_weights": (
                        dict(diagnostics.fusion_weights)
                        if getattr(diagnostics, "fusion_weights", None) is not None
                        else None
                    ),
                }
            results.append(item)

        body = {
            "results": results,
            "next_cursor": response.next_cursor,
            "total_candidates": response.total_candidates,
            "timings_ms": dict(response.timings_ms),
            "fusion_weights": dict(response.fusion_weights),
            "stats": response.stats,
        }
        return HTTPStatus.OK, body

    def _parse_request(self, payload: Mapping[str, Any]) -> HybridSearchRequest:
        query = str(payload["query"])
        namespace = payload.get("namespace")
        filters = self._normalize_filters(payload.get("filters"))
        page_size = int(payload.get("page_size", payload.get("limit", 10)))
        cursor = payload.get("cursor")
        diversification = bool(payload.get("diversification", False))
        diagnostics = bool(payload.get("diagnostics", True))
        recall_first = bool(payload.get("recall_first", False))
        return HybridSearchRequest(
            query=query,
            namespace=str(namespace) if namespace is not None else None,
            filters=filters,
            page_size=page_size,
            cursor=str(cursor) if cursor is not None else None,
            diversification=diversification,
            diagnostics=diagnostics,
            recall_first=recall_first,
        )

    def _normalize_filters(self, payload: Optional[Mapping[str, Any]]) -> MutableMapping[str, Any]:
        normalized: MutableMapping[str, Any] = {}
        if not payload:
            return normalized

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
        termination_reason: Description explaining why pagination inspection
            stopped.

    Examples:
        >>> result = PaginationCheckResult(cursor_chain=["cursor1"], duplicate_detected=False)
        >>> result.duplicate_detected
        False
    """

    cursor_chain: Sequence[str]
    duplicate_detected: bool
    termination_reason: str = "cursor_exhausted"


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


_DEFAULT_PAGINATION_PAGE_LIMIT = 32


def verify_pagination(
    service: HybridSearchService,
    request: HybridSearchRequest,
    *,
    max_pages: Optional[int] = None,
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
    inspected_pages = 0
    termination_reason = "cursor_exhausted"

    page_limit = max_pages
    if page_limit is None:
        try:
            config = service._config_manager.get()  # type: ignore[attr-defined]
        except Exception:
            config = None
        if config is not None:
            try:
                mmr_pool = getattr(config.retrieval, "mmr_pool_size", None)
                page_size = max(1, int(getattr(request, "page_size", 1)))
            except Exception:
                mmr_pool = None
                page_size = 1
            else:
                if isinstance(mmr_pool, (int, float)) and mmr_pool > 0:
                    page_limit = max(1, math.ceil(mmr_pool / page_size))
    if page_limit is None:
        page_limit = _DEFAULT_PAGINATION_PAGE_LIMIT

    while True:
        response = service.search(next_request)
        inspected_pages += 1
        for result in response.results:
            key = (result.doc_id, result.chunk_id)
            if key in seen:
                duplicate = True
            seen.add(key)

        next_cursor = response.next_cursor
        if not next_cursor:
            termination_reason = "cursor_exhausted"
            break
        if next_cursor in seen_cursors:
            termination_reason = "cursor_cycle"
            break
        if inspected_pages >= page_limit:
            termination_reason = "max_pages_reached"
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
            recall_first=request.recall_first,
        )

    return PaginationCheckResult(
        cursor_chain=cursor_chain,
        duplicate_detected=duplicate,
        termination_reason=termination_reason,
    )


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


# --- Validation Utilities ---

DEFAULT_SCALE_THRESHOLDS: Dict[str, float] = {
    "dense_self_hit": 0.99,
    "dense_recall_at_10": 0.95,
    "dense_perturb_top3": 0.95,
    "bm25_hit_rate@10": 0.8,
    "splade_hit_rate@10": 0.8,
    "dense_hit_rate@10": 0.8,
    "rrf_hit_rate@10": 0.8,
    "mmr_redundancy_reduction": 0.1,
    "mmr_hit_rate_delta": 0.2,
    "latency_p95_ms": 300.0,
    "gpu_headroom_fraction": 0.2,
}

BASIC_DENSE_SELF_HIT_THRESHOLD = 0.99
BASIC_SPARSE_RELEVANCE_THRESHOLD = 0.90


# --- Public Classes ---


class HybridSearchValidator:
    """Execute validation sweeps and persist reports.

    Attributes:
        _ingestion: Chunk ingestion pipeline used for preparing test data.
        _service: Hybrid search service under validation.
        _registry: Registry exposing ingested chunk metadata.
        _opensearch: OpenSearch simulator for lexical storage inspection.

    Examples:
        >>> validator = HybridSearchValidator(
        ...     ingestion=ChunkIngestionPipeline(...),  # doctest: +SKIP
        ...     service=HybridSearchService(...),      # doctest: +SKIP
        ...     registry=ChunkRegistry(),
        ...     opensearch=OpenSearchSimulator(),
        ... )
        >>> isinstance(validator, HybridSearchValidator)
        True
    """

    def __init__(
        self,
        *,
        ingestion: ChunkIngestionPipeline,
        service: HybridSearchService,
        registry: ChunkRegistry,
        opensearch: OpenSearchSimulator,
    ) -> None:
        """Bind ingestion, retrieval, and storage components for validation sweeps.

        Args:
            ingestion: Ingestion pipeline used to materialise chunk data.
            service: Hybrid search service under test.
            registry: Chunk registry exposing ingested metadata.
            opensearch: OpenSearch simulator for lexical validation.

        Returns:
            None
        """

        self._ingestion = ingestion
        self._service = service
        self._registry = registry
        self._opensearch = opensearch
        self._validation_resources: Optional["faiss.StandardGpuResources"] = None

    def run(
        self, dataset: Sequence[Mapping[str, object]], output_root: Optional[Path] = None
    ) -> ValidationSummary:
        """Execute standard validation against a dataset.

        Args:
            dataset: Loaded dataset entries containing document and query payloads.
            output_root: Optional directory where reports should be written.

        Returns:
            ValidationSummary capturing per-check reports.
        """
        started = datetime.now(UTC)
        documents = [self._to_document(entry["document"]) for entry in dataset]
        self._ingestion.upsert_documents(documents)
        reports: List[ValidationReport] = []
        reports.append(self._check_ingest_integrity())
        reports.append(self._check_dense_self_hit())
        reports.append(self._check_sparse_relevance(dataset))
        reports.append(self._check_namespace_filters(dataset))
        reports.append(self._check_pagination(dataset))
        reports.append(self._check_highlights(dataset))
        reports.append(self._check_backup_restore(dataset))
        calibration = self._run_calibration(dataset)
        reports.append(calibration)
        completed = datetime.now(UTC)
        summary = ValidationSummary(reports=reports, started_at=started, completed_at=completed)
        self._persist_reports(
            summary, output_root, calibration.details if calibration.details else None
        )
        return summary

    def run_scale(
        self,
        dataset: Sequence[Mapping[str, object]],
        *,
        output_root: Optional[Path] = None,
        thresholds: Optional[Mapping[str, float]] = None,
        query_sample_size: int = 120,
    ) -> ValidationSummary:
        """Execute a comprehensive scale validation suite.

        Args:
            dataset: Dataset entries containing documents and queries.
            output_root: Optional directory for detailed metrics output.
            thresholds: Optional overrides for scale validation thresholds.
            query_sample_size: Number of queries sampled for specific checks.

        Returns:
            ValidationSummary with detailed per-check reports.
        """
        merged_thresholds: Dict[str, float] = dict(DEFAULT_SCALE_THRESHOLDS)
        if thresholds:
            merged_thresholds.update(thresholds)

        started = datetime.now(UTC)
        documents = [self._to_document(entry["document"]) for entry in dataset]
        inputs_by_doc = {doc.doc_id: doc for doc in documents}
        self._ingestion.upsert_documents(documents)
        rng = random.Random(1337)

        reports: List[ValidationReport] = []
        extras: Dict[str, Mapping[str, object]] = {}

        data_report = self._scale_data_sanity(documents, dataset)
        reports.append(data_report)
        extras[data_report.name] = data_report.details

        crud_report = self._scale_crud_namespace(documents, dataset, inputs_by_doc, rng)
        reports.append(crud_report)
        extras[crud_report.name] = crud_report.details

        dense_report = self._scale_dense_metrics(merged_thresholds, rng)
        reports.append(dense_report)
        extras[dense_report.name] = dense_report.details

        relevance_report = self._scale_channel_relevance(
            dataset, merged_thresholds, rng, query_sample_size
        )
        reports.append(relevance_report)
        extras[relevance_report.name] = relevance_report.details

        fusion_report = self._scale_fusion_mmr(dataset, merged_thresholds, rng, query_sample_size)
        reports.append(fusion_report)
        extras[fusion_report.name] = fusion_report.details

        pagination_report = self._scale_pagination(dataset, rng)
        reports.append(pagination_report)
        extras[pagination_report.name] = pagination_report.details

        shaping_report = self._scale_result_shaping(dataset, rng)
        reports.append(shaping_report)
        extras[shaping_report.name] = shaping_report.details

        backup_report = self._scale_backup_restore(dataset, rng, query_sample_size)
        reports.append(backup_report)
        extras[backup_report.name] = backup_report.details

        acl_report = self._scale_acl(dataset)
        reports.append(acl_report)
        extras[acl_report.name] = acl_report.details

        performance_report = self._scale_performance(
            dataset, merged_thresholds, rng, query_sample_size
        )
        reports.append(performance_report)
        extras[performance_report.name] = performance_report.details

        stability_report = self._scale_stability(dataset, inputs_by_doc, rng, query_sample_size)
        reports.append(stability_report)
        extras[stability_report.name] = stability_report.details

        calibration = self._run_calibration(dataset)
        reports.append(calibration)
        extras[calibration.name] = calibration.details if calibration.details else {}

        completed = datetime.now(UTC)
        summary = ValidationSummary(reports=reports, started_at=started, completed_at=completed)
        self._persist_reports(
            summary,
            output_root,
            calibration.details if calibration.details else None,
            extras=extras,
        )
        return summary

    def _to_document(self, payload: Mapping[str, object]) -> DocumentInput:
        """Transform dataset document payload into a `DocumentInput`.

        Args:
            payload: Dataset document dictionary containing paths and metadata.

        Returns:
            Corresponding `DocumentInput` instance for ingestion.
        """
        return DocumentInput(
            doc_id=str(payload["doc_id"]),
            namespace=str(payload["namespace"]),
            chunk_path=Path(str(payload["chunk_file"])).resolve(),
            vector_path=Path(str(payload["vector_file"])).resolve(),
            metadata=dict(payload.get("metadata", {})),
        )

    def _check_ingest_integrity(self) -> ValidationReport:
        """Verify that ingested chunks contain valid embeddings.

        Args:
            None

        Returns:
            ValidationReport describing integrity checks for ingested chunks.
        """
        all_chunks = self._registry.all()
        vector_ids = [chunk.vector_id for chunk in all_chunks]
        ok = True
        dim = 0
        if vector_ids:
            try:
                reconstructed = self._registry.resolve_embeddings(vector_ids)
            except Exception:
                ok = False
            else:
                ok = reconstructed.ndim == 2 and reconstructed.shape[0] == len(vector_ids)
                dim = int(reconstructed.shape[1]) if reconstructed.ndim == 2 else 0
        details = {"total_chunks": len(all_chunks), "embedding_dim": dim}
        return ValidationReport(name="ingest_integrity", passed=ok, details=details)

    def _check_dense_self_hit(self) -> ValidationReport:
        """Ensure dense search returns each chunk as its own nearest neighbor.

        Args:
            None

        Returns:
            ValidationReport summarizing dense self-hit accuracy.
        """
        total = 0
        hits_met = 0
        embedding_cache: Dict[str, np.ndarray] = {}
        for chunk in self._registry.all():
            total += 1
            query_vector = self._registry.resolve_embedding(chunk.vector_id, cache=embedding_cache)
            hits = self._ingestion.faiss_index.search(query_vector, 1)
            if hits and hits[0].vector_id == chunk.vector_id:
                hits_met += 1
        rate = hits_met / total if total else 0.0
        passed = rate >= BASIC_DENSE_SELF_HIT_THRESHOLD
        return ValidationReport(
            name="dense_self_hit",
            passed=passed,
            details={
                "total_chunks": total,
                "correct_hits": hits_met,
                "self_hit_rate": rate,
                "threshold": BASIC_DENSE_SELF_HIT_THRESHOLD,
            },
        )

    def _check_sparse_relevance(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Check that sparse channels retrieve expected documents.

        Args:
            dataset: Dataset entries with expected relevance metadata.

        Returns:
            ValidationReport covering sparse channel relevance.
        """
        total = 0
        hits_met = 0
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                total += 1
                expected = query.get("expected_doc_id")
                request = self._request_for_query(query)
                response = self._service.search(request)
                if response.results and (not expected or response.results[0].doc_id == expected):
                    hits_met += 1
        rate = hits_met / total if total else 0.0
        passed = rate >= BASIC_SPARSE_RELEVANCE_THRESHOLD
        return ValidationReport(
            name="sparse_relevance",
            passed=passed,
            details={
                "total_queries": total,
                "top1_matches": hits_met,
                "hit_rate": rate,
                "threshold": BASIC_SPARSE_RELEVANCE_THRESHOLD,
            },
        )

    def _check_namespace_filters(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Ensure namespace-constrained queries do not leak results.

        Args:
            dataset: Dataset entries containing namespace-constrained queries.

        Returns:
            ValidationReport indicating whether namespace isolation holds.
        """
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                namespace = query.get("namespace")
                if not namespace:
                    continue
                request = self._request_for_query(query)
                response = self._service.search(request)
                for result in response.results:
                    if result.namespace != namespace:
                        ok = False
                        break
            if not ok:
                break
        return ValidationReport(name="namespace_filter", passed=ok, details={})

    def _check_pagination(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Verify pagination cursors do not create duplicate results.

        Args:
            dataset: Dataset entries providing queries for pagination tests.

        Returns:
            ValidationReport detailing pagination stability.
        """
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                request = self._request_for_query(query, page_size=2)
                first = self._service.search(request)
                if not first.next_cursor:
                    continue
                request.cursor = first.next_cursor
                second = self._service.search(request)
                seen = {(result.doc_id, result.chunk_id) for result in first.results}
                overlap = any((result.doc_id, result.chunk_id) in seen for result in second.results)
                if overlap:
                    ok = False
                    break
            if not ok:
                break
        return ValidationReport(name="pagination_stability", passed=ok, details={})

    def _check_highlights(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Assert that each result includes highlight snippets.

        Args:
            dataset: Dataset entries with queries to validate highlight generation.

        Returns:
            ValidationReport indicating highlight completeness.
        """
        ok = True
        for entry in dataset:
            queries = entry.get("queries", [])
            for query in queries:
                request = self._request_for_query(query)
                response = self._service.search(request)
                if any(not result.highlights for result in response.results):
                    ok = False
                    break
            if not ok:
                break
        return ValidationReport(name="highlight_presence", passed=ok, details={})

    def _check_backup_restore(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Validate backup/restore by round-tripping the FAISS index.

        Args:
            dataset: Dataset entries used to verify consistency post-restore.

        Returns:
            ValidationReport capturing backup/restore status.
        """
        if not dataset:
            return ValidationReport(
                name="backup_restore", passed=False, details={"error": "dataset empty"}
            )
        queries = dataset[0].get("queries", [])
        if not queries:
            return ValidationReport(
                name="backup_restore", passed=False, details={"error": "missing queries"}
            )
        faiss_bytes = self._ingestion.faiss_index.serialize()
        first_query = self._request_for_query(queries[0])
        before = self._service.search(first_query)
        self._ingestion.faiss_index.restore(faiss_bytes)
        after = self._service.search(first_query)
        ok = bool(
            before.results and after.results and before.results[0].doc_id == after.results[0].doc_id
        )
        return ValidationReport(name="backup_restore", passed=ok, details={})

    def _request_for_query(
        self, query: Mapping[str, object], page_size: int = 5
    ) -> HybridSearchRequest:
        """Construct a `HybridSearchRequest` for a dataset query payload.

        Args:
            query: Query payload from the dataset.
            page_size: Desired page size for generated requests.

        Returns:
            HybridSearchRequest ready for execution against the service.
        """
        return HybridSearchRequest(
            query=str(query["query"]),
            namespace=query.get("namespace"),
            filters=dict(query.get("filters", {})),
            page_size=page_size,
            cursor=None,
            diversification=bool(query.get("diversification", False)),
            diagnostics=bool(query.get("diagnostics", True)),
            recall_first=bool(query.get("recall_first", False)),
        )

    def _persist_reports(
        self,
        summary: ValidationSummary,
        output_root: Optional[Path],
        calibration_details: Optional[Mapping[str, object]],
        *,
        extras: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        """Persist validation summaries and detailed metrics to disk.

        Args:
            summary: Validation summary containing reports.
            output_root: Optional directory for storing artifacts.
            calibration_details: Optional calibration metrics to persist.
            extras: Optional mapping of additional metrics categories.

        Returns:
            None
        """
        root = output_root or Path("reports/validation")
        directory = root / summary.started_at.strftime("%Y%m%d%H%M%S")
        directory.mkdir(parents=True, exist_ok=True)
        reports_json = [
            {
                "name": report.name,
                "passed": report.passed,
                "details": report.details,
            }
            for report in summary.reports
        ]
        (directory / "summary.json").write_text(
            json.dumps(reports_json, indent=2), encoding="utf-8"
        )
        human_lines = [
            f"Validation started at: {summary.started_at.isoformat()}",
            f"Validation completed at: {summary.completed_at.isoformat()}",
            f"Overall status: {'PASS' if summary.passed else 'FAIL'}",
        ]
        for report in summary.reports:
            human_lines.append(f"- {report.name}: {'PASS' if report.passed else 'FAIL'}")
        (directory / "summary.txt").write_text("\n".join(human_lines), encoding="utf-8")
        if calibration_details is not None:
            (directory / "calibration.json").write_text(
                json.dumps(calibration_details, indent=2), encoding="utf-8"
            )
        if extras:
            (directory / "metrics.json").write_text(json.dumps(extras, indent=2), encoding="utf-8")

    def _collect_queries(
        self, dataset: Sequence[Mapping[str, object]]
    ) -> List[tuple[Mapping[str, object], Mapping[str, object]]]:
        """Collect (document, query) pairs from the dataset.

        Args:
            dataset: Loaded dataset entries containing documents and queries.

        Returns:
            List of tuples pairing document payloads with query payloads.
        """
        pairs: List[tuple[Mapping[str, object], Mapping[str, object]]] = []
        for entry in dataset:
            document_payload = entry.get("document", {})
            for query in entry.get("queries", []):
                pairs.append((document_payload, query))
        return pairs

    def _sample_queries(
        self,
        dataset: Sequence[Mapping[str, object]],
        sample_size: int,
        rng: random.Random,
    ) -> List[tuple[Mapping[str, object], Mapping[str, object]]]:
        """Sample a subset of (document, query) pairs for randomized checks.

        Args:
            dataset: Dataset entries containing documents and queries.
            sample_size: Maximum number of pairs to return.
            rng: Random generator used for sampling.

        Returns:
            List of sampled (document, query) pairs.
        """
        pairs = self._collect_queries(dataset)
        if not pairs:
            return []
        if sample_size >= len(pairs):
            return pairs
        return rng.sample(pairs, sample_size)

    def _scale_data_sanity(
        self,
        documents: Sequence[DocumentInput],
        dataset: Sequence[Mapping[str, object]],
    ) -> ValidationReport:
        """Validate that ingested corpus statistics look healthy.

        Args:
            documents: Document inputs ingested during the scale run.
            dataset: Dataset entries used for validation.

        Returns:
            ValidationReport summarizing data sanity findings.
        """
        total_chunks = self._registry.count()
        namespaces = sorted({doc.namespace for doc in documents})
        dims: set[int] = set()
        invalid_vectors = 0
        embedding_cache: Dict[str, np.ndarray] = {}
        for chunk in self._registry.all():
            vector = self._registry.resolve_embedding(chunk.vector_id, cache=embedding_cache)
            dims.add(vector.shape[0])
            if not np.isfinite(vector).all():
                invalid_vectors += 1
        acl_missing = sum(1 for doc in documents if not doc.metadata.get("acl"))
        query_pairs = self._collect_queries(dataset)
        passed = len(dims) == 1 and invalid_vectors == 0 and acl_missing == 0
        details: Dict[str, object] = {
            "total_documents": len(documents),
            "total_queries": len(query_pairs),
            "total_chunks": total_chunks,
            "namespaces": namespaces,
            "vector_dimensions": sorted(dims),
            "invalid_vector_count": invalid_vectors,
            "documents_missing_acl": acl_missing,
        }
        return ValidationReport(name="scale_data_sanity", passed=passed, details=details)

    def _scale_crud_namespace(
        self,
        documents: Sequence[DocumentInput],
        dataset: Sequence[Mapping[str, object]],
        inputs_by_doc: Mapping[str, DocumentInput],
        rng: random.Random,
    ) -> ValidationReport:
        """Exercise CRUD operations to ensure namespace isolation stays intact.

        Args:
            documents: Documents ingested for the validation run.
            dataset: Dataset entries providing queries for verification.
            inputs_by_doc: Mapping back to original document inputs.
            rng: Random generator used to select documents.

        Returns:
            ValidationReport describing CRUD and namespace violations if any.
        """
        initial_registry = self._registry.count()
        initial_faiss = self._ingestion.faiss_index.ntotal

        doc_ids = [doc.doc_id for doc in documents]
        update_count = min(max(10, len(doc_ids) // 10), len(doc_ids)) or 1
        update_doc_ids = rng.sample(doc_ids, update_count)
        self._ingestion.upsert_documents([inputs_by_doc[doc_id] for doc_id in update_doc_ids])

        update_ok = (
            self._registry.count() == initial_registry
            and self._ingestion.faiss_index.ntotal == initial_faiss
        )

        vector_ids = [chunk.vector_id for chunk in self._registry.all()]
        delete_count = min(max(10, len(vector_ids) // 20), len(vector_ids) // 2 or 1)
        delete_ids = rng.sample(vector_ids, delete_count)
        deleted_doc_ids = set()
        for vector_id in delete_ids:
            chunk = self._registry.get(vector_id)
            if chunk is not None:
                deleted_doc_ids.add(chunk.doc_id)
        self._ingestion.delete_chunks(delete_ids)

        delete_ok = (
            self._registry.count() == initial_registry - delete_count
            and self._ingestion.faiss_index.ntotal == initial_faiss - delete_count
        )

        if deleted_doc_ids:
            self._ingestion.upsert_documents([inputs_by_doc[doc_id] for doc_id in deleted_doc_ids])

        restore_ok = (
            self._registry.count() == initial_registry
            and self._ingestion.faiss_index.ntotal == initial_faiss
        )

        namespace_pairs: Dict[str, List[Mapping[str, object]]] = {}
        for document_payload, query_payload in self._collect_queries(dataset):
            namespace = query_payload.get("namespace") or document_payload.get("namespace")
            if namespace:
                namespace_pairs.setdefault(str(namespace), []).append(query_payload)

        namespace_violations: List[str] = []
        for namespace, queries in namespace_pairs.items():
            sample = queries[: min(5, len(queries))]
            for query_payload in sample:
                request = self._request_for_query(query_payload)
                response = self._service.search(request)
                for result in response.results:
                    if result.namespace != namespace:
                        namespace_violations.append(namespace)
                        break
                if namespace_violations:
                    break

        details: Dict[str, object] = {
            "updates_tested": update_count,
            "deletes_tested": delete_count,
            "namespaces_checked": sorted(namespace_pairs.keys()),
            "namespace_violations": namespace_violations,
            "registry_count": self._registry.count(),
            "faiss_ntotal": self._ingestion.faiss_index.ntotal,
        }

        passed = update_ok and delete_ok and restore_ok and not namespace_violations
        return ValidationReport(name="scale_crud_namespace", passed=passed, details=details)

    def _scale_dense_metrics(
        self,
        thresholds: Mapping[str, float],
        rng: random.Random,
    ) -> ValidationReport:
        """Assess dense retrieval self-hit, perturbation, and recall metrics.

        Args:
            thresholds: Threshold values for dense metric success.
            rng: Random generator for selecting sample chunks.

        Returns:
            ValidationReport containing dense metric measurements.
        """
        all_chunks = self._registry.all()
        if not all_chunks:
            return ValidationReport(
                name="scale_dense_metrics",
                passed=False,
                details={"error": "registry empty"},
            )

        sample_size = min(max(200, len(all_chunks) // 4), len(all_chunks))
        sampled_chunks = rng.sample(all_chunks, sample_size)

        top_k = min(10, len(all_chunks))
        self_hits = 0
        perturb_hits = 0
        recalls: List[float] = []

        # Precompute matrix for brute-force recall estimates.
        embedding_cache: Dict[str, np.ndarray] = {}
        vector_ids = [chunk.vector_id for chunk in all_chunks]
        vector_matrix = self._registry.resolve_embeddings(
            vector_ids, cache=embedding_cache
        )
        vector_matrix = np.asarray(vector_matrix, dtype=np.float32)
        if vector_matrix.ndim == 1:
            vector_matrix = vector_matrix.reshape(1, -1)
        vector_lookup = {
            vector_id: embedding_cache.get(vector_id, vector_matrix[idx])
            for idx, vector_id in enumerate(vector_ids)
        }
        try:
            adapter_stats = self._ingestion.faiss_index.adapter_stats  # type: ignore[attr-defined]
        except AttributeError:
            adapter_stats = None

        resources = adapter_stats.resources if adapter_stats is not None else None
        try:
            ingestion_device = getattr(self._ingestion.faiss_index, "device", 0)
        except Exception:
            ingestion_device = 0
        try:
            device = (
                int(adapter_stats.device) if adapter_stats is not None else int(ingestion_device)
            )
        except Exception:
            device = 0

        gpu_resources = resources
        scratch_index = None
        using_gpu_ground_truth = False
        ordered_vector_ids: List[str] = list(vector_ids)
        normalized_cpu_vectors: Optional[List[np.ndarray]] = None
        embedding_dim = int(vector_matrix.shape[1]) if vector_matrix.size else 0
        if embedding_dim <= 0:
            embedding_dim = int(
                getattr(getattr(self._ingestion.faiss_index, "config", object()), "dim", 0)
                or getattr(self._ingestion.faiss_index, "dim", 0)
            )

        if gpu_resources is None and faiss is not None and hasattr(faiss, "StandardGpuResources"):
            try:
                gpu_resources = self._ensure_validation_resources()
            except Exception:
                gpu_resources = None

        if faiss is not None and gpu_resources is not None and hasattr(faiss, "GpuIndexFlatIP"):
            try:
                gpu_config_cls = getattr(faiss, "GpuIndexFlatConfig", None)
                if gpu_config_cls is not None:
                    gpu_config = gpu_config_cls()
                    gpu_config.device = int(device)
                    if adapter_stats is not None and hasattr(gpu_config, "useFloat16"):
                        gpu_config.useFloat16 = bool(getattr(adapter_stats, "fp16_enabled", False))
                    scratch_index = faiss.GpuIndexFlatIP(gpu_resources, embedding_dim, gpu_config)
                else:
                    scratch_index = faiss.index_cpu_to_gpu(
                        gpu_resources,
                        int(device),
                        faiss.IndexFlatIP(embedding_dim),
                    )
            except Exception:
                scratch_index = None

        if scratch_index is not None:
            using_gpu_ground_truth = True
            try:
                for vector_id in ordered_vector_ids:
                    embedding = np.array(vector_lookup[vector_id], dtype=np.float32, copy=True)
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)
                    faiss.normalize_L2(embedding)
                    scratch_index.add(embedding)
            except Exception:
                scratch_index.reset()
                scratch_index = None
                using_gpu_ground_truth = False

        if not using_gpu_ground_truth:
            normalized_cpu_vectors = []
            for vector_id in ordered_vector_ids:
                embedding = np.array(vector_lookup[vector_id], dtype=np.float32, copy=True)
                flat_embedding = embedding.reshape(-1)
                norm = float(np.linalg.norm(flat_embedding)) or 1.0
                normalized = (flat_embedding / norm).astype(np.float32, copy=False)
                normalized_cpu_vectors.append(normalized)

        noise_rng = np.random.default_rng(2024)

        for chunk in sampled_chunks:
            query_vec = np.array(vector_lookup[chunk.vector_id], dtype=np.float32, copy=True)
            hits = self._ingestion.faiss_index.search(query_vec, top_k)
            retrieved_ids = [hit.vector_id for hit in hits]
            if retrieved_ids and retrieved_ids[0] == chunk.vector_id:
                self_hits += 1

            noise = noise_rng.normal(scale=0.01, size=query_vec.shape).astype(np.float32)
            perturbed = query_vec + noise
            perturbed_hits = self._ingestion.faiss_index.search(perturbed, top_k)
            if any(
                hit.vector_id == chunk.vector_id
                for hit in perturbed_hits[: min(3, len(perturbed_hits))]
            ):
                perturb_hits += 1

            if using_gpu_ground_truth and scratch_index is not None and faiss is not None:
                q = np.array(query_vec, dtype=np.float32, copy=True)
                if q.ndim == 1:
                    q = q.reshape(1, -1)
                faiss.normalize_L2(q)
                _scores, indices_block = scratch_index.search(q, top_k)
                top_indices = indices_block[0]
            else:
                q = np.array(query_vec, dtype=np.float32, copy=True)
                norm = float(np.linalg.norm(q)) or 1.0
                q = (q / norm).astype(np.float32, copy=False)
                assert normalized_cpu_vectors is not None
                scores = np.fromiter(
                    (float(np.dot(q, corpus_vec)) for corpus_vec in normalized_cpu_vectors),
                    dtype=np.float32,
                    count=len(normalized_cpu_vectors),
                )
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            valid_indices = [idx for idx in top_indices if 0 <= idx < len(ordered_vector_ids)]
            ground_truth_ids = [ordered_vector_ids[idx] for idx in valid_indices]
            overlap = len(set(retrieved_ids) & set(ground_truth_ids))
            recalls.append(overlap / min(top_k, len(ground_truth_ids)) if ground_truth_ids else 0.0)

        self_hit_rate = self_hits / sample_size if sample_size else 0.0
        perturb_rate = perturb_hits / sample_size if sample_size else 0.0
        avg_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0

        details = {
            "sampled_chunks": sample_size,
            "self_hit_rate": self_hit_rate,
            "perturb_top3_rate": perturb_rate,
            "recall_at_10": avg_recall,
        }

        passed = (
            self_hit_rate >= thresholds.get("dense_self_hit", 0.0)
            and perturb_rate >= thresholds.get("dense_perturb_top3", 0.0)
            and avg_recall >= thresholds.get("dense_recall_at_10", 0.0)
        )
        if scratch_index is not None:
            scratch_index.reset()

        return ValidationReport(name="scale_dense_metrics", passed=passed, details=details)

    def _scale_channel_relevance(
        self,
        dataset: Sequence[Mapping[str, object]],
        thresholds: Mapping[str, float],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Measure top-10 relevance for each retrieval channel across sampled queries.

        Args:
            dataset: Dataset entries containing queries for evaluation.
            thresholds: Threshold mapping for hit-rate expectations.
            rng: Random generator used for sampling query pairs.
            query_sample_size: Number of query pairs sampled.

        Returns:
            ValidationReport capturing per-channel relevance metrics.
        """
        sampled_pairs = self._sample_queries(dataset, query_sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_channel_relevance",
                passed=False,
                details={"error": "no queries available"},
            )

        feature_generator = self._service._feature_generator
        bm25_hits = 0
        splade_hits = 0
        dense_hits = 0
        rrf_hits = 0

        bm25_ranks: List[int] = []
        splade_ranks: List[int] = []
        dense_ranks: List[int] = []
        rrf_ranks: List[int] = []

        doc_to_embedding: Dict[Tuple[str, str], np.ndarray] = {}
        registry_chunks = self._registry.all()
        if registry_chunks:
            vectors = self._registry.resolve_embeddings(
                [chunk.vector_id for chunk in registry_chunks]
            )
            for chunk, vector in zip(registry_chunks, vectors):
                key = (chunk.namespace, chunk.doc_id)
                doc_to_embedding.setdefault(key, vector)

        for document_payload, query_payload in sampled_pairs:
            expected_doc_id = str(
                query_payload.get("expected_doc_id") or document_payload.get("doc_id")
            )
            request = self._request_for_query(query_payload, page_size=10)
            filters = dict(request.filters)
            if request.namespace:
                filters["namespace"] = request.namespace

            features = feature_generator.compute_features(request.query)
            namespace = request.namespace
            if namespace is None:
                document_namespace = document_payload.get("namespace")
                namespace = (
                    ""
                    if document_namespace is None
                    else str(document_namespace)
                )
            else:
                namespace = str(namespace)
            dense_query_vector = doc_to_embedding.get(
                (namespace, expected_doc_id),
                features.embedding,
            )

            bm25_results, _ = self._opensearch.search_bm25(features.bm25_terms, filters, top_k=10)
            bm25_doc_ids = [chunk.doc_id for chunk, _ in bm25_results]
            if expected_doc_id in bm25_doc_ids:
                bm25_hits += 1
                bm25_ranks.append(bm25_doc_ids.index(expected_doc_id) + 1)

            splade_results, _ = self._opensearch.search_splade(
                features.splade_weights, filters, top_k=10
            )
            splade_doc_ids = [chunk.doc_id for chunk, _ in splade_results]
            if expected_doc_id in splade_doc_ids:
                splade_hits += 1
                splade_ranks.append(splade_doc_ids.index(expected_doc_id) + 1)

            dense_results = self._ingestion.faiss_index.search(dense_query_vector, 10)
            dense_doc_ids: List[str] = []
            for hit in dense_results:
                payload = self._registry.get(hit.vector_id)
                if payload is not None:
                    dense_doc_ids.append(payload.doc_id)
            if expected_doc_id in dense_doc_ids:
                dense_hits += 1
                dense_ranks.append(dense_doc_ids.index(expected_doc_id) + 1)

            fused_response = self._service.search(request)
            fused_doc_ids = [result.doc_id for result in fused_response.results[:10]]
            if expected_doc_id in fused_doc_ids:
                rrf_hits += 1
                rrf_ranks.append(fused_doc_ids.index(expected_doc_id) + 1)

        total_queries = len(sampled_pairs)
        bm25_rate = bm25_hits / total_queries
        splade_rate = splade_hits / total_queries
        dense_rate = dense_hits / total_queries
        rrf_rate = rrf_hits / total_queries

        details: Dict[str, object] = {
            "query_count": total_queries,
            "bm25_hit_rate@10": bm25_rate,
            "splade_hit_rate@10": splade_rate,
            "dense_hit_rate@10": dense_rate,
            "rrf_hit_rate@10": rrf_rate,
            "bm25_avg_rank": statistics.mean(bm25_ranks) if bm25_ranks else None,
            "splade_avg_rank": statistics.mean(splade_ranks) if splade_ranks else None,
            "dense_avg_rank": statistics.mean(dense_ranks) if dense_ranks else None,
            "rrf_avg_rank": statistics.mean(rrf_ranks) if rrf_ranks else None,
        }

        passed = (
            bm25_rate >= thresholds.get("bm25_hit_rate@10", 0.0)
            and splade_rate >= thresholds.get("splade_hit_rate@10", 0.0)
            and dense_rate >= thresholds.get("dense_hit_rate@10", 0.0)
            and rrf_rate >= thresholds.get("rrf_hit_rate@10", 0.0)
        )
        return ValidationReport(name="scale_channel_relevance", passed=passed, details=details)

    def _scale_fusion_mmr(
        self,
        dataset: Sequence[Mapping[str, object]],
        thresholds: Mapping[str, float],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Evaluate diversification impact of MMR versus RRF on sampled queries.

        Args:
            dataset: Dataset entries containing documents and queries.
            thresholds: Diversification thresholds for redundancy and hit-rate deltas.
            rng: Random generator for sampling queries.
            query_sample_size: Number of query pairs to evaluate.

        Returns:
            ValidationReport detailing redundancy reduction and hit-rate deltas.
        """
        sampled_pairs = self._sample_queries(dataset, query_sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_fusion_mmr",
                passed=False,
                details={"error": "no queries available"},
            )

        chunk_lookup = {
            (chunk.namespace, chunk.doc_id, chunk.chunk_id): chunk
            for chunk in self._registry.all()
        }

        redundancy_reductions: List[float] = []
        rrf_cosines: List[float] = []
        mmr_cosines: List[float] = []
        rrf_hits = 0
        mmr_hits = 0

        for document_payload, query_payload in sampled_pairs:
            expected_doc_id = str(
                query_payload.get("expected_doc_id") or document_payload.get("doc_id")
            )

            baseline_request = self._request_for_query(query_payload, page_size=10)
            baseline_response = self._service.search(baseline_request)
            baseline_doc_ids = [result.doc_id for result in baseline_response.results[:10]]
            baseline_vectors = self._embeddings_for_results(
                baseline_response.results, chunk_lookup
            )
            baseline_cos = self._average_pairwise_cos(baseline_vectors)
            rrf_cosines.append(baseline_cos)
            if expected_doc_id in baseline_doc_ids:
                rrf_hits += 1

            mmr_request = self._request_for_query(query_payload, page_size=10)
            mmr_request.diversification = True
            mmr_response = self._service.search(mmr_request)
            mmr_doc_ids = [result.doc_id for result in mmr_response.results[:10]]
            mmr_vectors = self._embeddings_for_results(mmr_response.results, chunk_lookup)
            mmr_cos = self._average_pairwise_cos(mmr_vectors)
            mmr_cosines.append(mmr_cos)
            if expected_doc_id in mmr_doc_ids:
                mmr_hits += 1

            redundancy_reductions.append(baseline_cos - mmr_cos)

        total_queries = len(sampled_pairs)
        mean_reduction = (
            float(sum(redundancy_reductions) / len(redundancy_reductions))
            if redundancy_reductions
            else 0.0
        )
        mean_rrf_cos = float(sum(rrf_cosines) / len(rrf_cosines)) if rrf_cosines else 0.0
        mean_mmr_cos = float(sum(mmr_cosines) / len(mmr_cosines)) if mmr_cosines else 0.0
        rrf_rate = rrf_hits / total_queries
        mmr_rate = mmr_hits / total_queries

        details = {
            "query_count": total_queries,
            "rrf_hit_rate@10": rrf_rate,
            "mmr_hit_rate@10": mmr_rate,
            "avg_rrf_pairwise_cosine": mean_rrf_cos,
            "avg_mmr_pairwise_cosine": mean_mmr_cos,
            "redundancy_reduction": mean_reduction,
            "hit_rate_delta": abs(rrf_rate - mmr_rate),
        }

        passed = mean_reduction >= thresholds.get("mmr_redundancy_reduction", 0.0) and abs(
            rrf_rate - mmr_rate
        ) <= thresholds.get("mmr_hit_rate_delta", float("inf"))
        return ValidationReport(name="scale_fusion_mmr", passed=passed, details=details)

    def _scale_pagination(
        self,
        dataset: Sequence[Mapping[str, object]],
        rng: random.Random,
        sample_size: int = 40,
    ) -> ValidationReport:
        """Ensure page cursors generate disjoint result sets across pages.

        Args:
            dataset: Dataset entries containing documents and queries.
            rng: Random generator for sampling query pairs.
            sample_size: Number of queries to evaluate for pagination.

        Returns:
            ValidationReport indicating pagination overlap issues.
        """
        sampled_pairs = self._sample_queries(dataset, sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_pagination",
                passed=False,
                details={"error": "no queries available"},
            )

        overlap_failures = 0
        order_mismatches = 0
        checked = 0

        for _, query_payload in sampled_pairs:
            base_request = self._request_for_query(query_payload, page_size=20)
            first_response = self._service.search(base_request)
            first_keys = [(res.doc_id, res.chunk_id) for res in first_response.results]
            if first_response.next_cursor:
                second_request = self._request_for_query(query_payload, page_size=20)
                second_request.cursor = first_response.next_cursor
                second_response = self._service.search(second_request)
                second_keys = [(res.doc_id, res.chunk_id) for res in second_response.results]
                if set(first_keys) & set(second_keys):
                    overlap_failures += 1
            repeat_request = self._request_for_query(query_payload, page_size=20)
            repeat_response = self._service.search(repeat_request)
            repeat_keys = [(res.doc_id, res.chunk_id) for res in repeat_response.results]
            if first_keys != repeat_keys:
                order_mismatches += 1
            checked += 1

        details = {
            "queries_checked": checked,
            "overlap_failures": overlap_failures,
            "order_mismatches": order_mismatches,
        }
        passed = overlap_failures == 0 and order_mismatches == 0
        return ValidationReport(name="scale_pagination", passed=passed, details=details)

    def _scale_result_shaping(
        self,
        dataset: Sequence[Mapping[str, object]],
        rng: random.Random,
        sample_size: int = 40,
    ) -> ValidationReport:
        """Confirm per-document limits, deduping, and highlights behave as expected.

        Args:
            dataset: Dataset entries containing documents and queries.
            rng: Random generator used for sampling query pairs.
            sample_size: Number of sampled queries to evaluate.

        Returns:
            ValidationReport summarizing result shaping concerns.
        """
        sampled_pairs = self._sample_queries(dataset, sample_size, rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_result_shaping",
                passed=False,
                details={"error": "no queries available"},
            )

        config = self._service._config_manager.get()
        max_per_doc = config.fusion.max_chunks_per_doc
        dedupe_threshold = config.fusion.cosine_dedupe_threshold
        chunk_lookup = {
            (chunk.namespace, chunk.doc_id, chunk.chunk_id): chunk
            for chunk in self._registry.all()
        }
        device = int(config.dense.device)

        doc_limit_violations = 0
        dedupe_violations = 0
        highlight_missing = 0
        embedding_cache: Dict[str, np.ndarray] = {}

        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=20)
            response = self._service.search(request)
            doc_counts: Dict[str, int] = {}
            embeddings: List[np.ndarray] = []

            for result in response.results:
                doc_counts[result.doc_id] = doc_counts.get(result.doc_id, 0) + 1
                chunk = chunk_lookup.get((result.namespace, result.doc_id, result.chunk_id))
                if chunk is not None:
                    embeddings.append(
                        self._registry.resolve_embedding(chunk.vector_id, cache=embedding_cache)
                    )
                if not result.highlights:
                    highlight_missing += 1

            if any(count > max_per_doc for count in doc_counts.values()):
                doc_limit_violations += 1

            # Dedupe check based on cosine similarity
            if len(embeddings) > 1:
                matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
                sims = pairwise_inner_products(
                    matrix,
                    device=device,
                    resources=self._ensure_validation_resources(),
                )
                if np.any(np.triu(sims, k=1) >= dedupe_threshold):
                    dedupe_violations += 1

        details = {
            "queries_checked": len(sampled_pairs),
            "doc_limit_violations": doc_limit_violations,
            "dedupe_violations": dedupe_violations,
            "missing_highlights": highlight_missing,
        }
        passed = doc_limit_violations == 0 and dedupe_violations == 0 and highlight_missing == 0
        return ValidationReport(name="scale_result_shaping", passed=passed, details=details)

    def _scale_backup_restore(
        self,
        dataset: Sequence[Mapping[str, object]],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Validate snapshot/restore under randomized workloads.

        Args:
            dataset: Dataset entries with documents and queries.
            rng: Random generator for sampling queries.
            query_sample_size: Maximum number of query pairs to evaluate.

        Returns:
            ValidationReport highlighting snapshot mismatches, if any.
        """
        sampled_pairs = self._sample_queries(dataset, min(30, query_sample_size), rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_backup_restore",
                passed=False,
                details={"error": "no queries available"},
            )

        snapshot = serialize_state(self._ingestion.faiss_index, self._registry)

        baseline_results: List[List[tuple[str, float]]] = []
        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=15)
            response = self._service.search(request)
            baseline_results.append(
                [(result.doc_id, round(result.score, 6)) for result in response.results[:15]]
            )

        restore_state(self._ingestion.faiss_index, snapshot, registry=self._registry)

        mismatches = 0
        for (_, query_payload), expected in zip(sampled_pairs, baseline_results):
            request = self._request_for_query(query_payload, page_size=15)
            response = self._service.search(request)
            observed = [(result.doc_id, round(result.score, 6)) for result in response.results[:15]]
            if observed != expected:
                mismatches += 1

        details = {
            "queries_checked": len(sampled_pairs),
            "mismatches": mismatches,
        }
        passed = mismatches == 0
        return ValidationReport(name="scale_backup_restore", passed=passed, details=details)

    def _scale_acl(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Ensure per-namespace ACL tags are enforced across queries.

        Args:
            dataset: Dataset entries providing documents and queries.

        Returns:
            ValidationReport listing ACL violations, if discovered.
        """
        namespace_to_acl: Dict[str, str] = {}
        namespace_queries: Dict[str, List[Mapping[str, object]]] = {}
        for entry in dataset:
            document = entry.get("document", {})
            namespace = str(document.get("namespace", ""))
            metadata = document.get("metadata", {})
            acl_entries = metadata.get("acl", [])
            if acl_entries:
                namespace_to_acl.setdefault(namespace, acl_entries[0])
            for query in entry.get("queries", []):
                namespace_queries.setdefault(namespace, []).append(query)

        violations: List[str] = []
        checked = 0
        for namespace, queries in namespace_queries.items():
            acl_tag = namespace_to_acl.get(namespace)
            if not acl_tag:
                continue
            sample = queries[: min(5, len(queries))]
            for query_payload in sample:
                request = self._request_for_query(query_payload, page_size=10)
                filters = dict(request.filters)
                filters["acl"] = [acl_tag]
                request.filters = filters
                response = self._service.search(request)
                checked += 1
                for result in response.results:
                    acl_values = result.metadata.get("acl")
                    if isinstance(acl_values, list):
                        allowed = acl_tag in acl_values
                    else:
                        allowed = acl_values == acl_tag
                    if not allowed:
                        violations.append(f"namespace={namespace}")
                        break
                # Cross-namespace negative check
                for other_namespace, other_acl in namespace_to_acl.items():
                    if other_namespace == namespace:
                        continue
                    negative_filters = dict(request.filters)
                    negative_filters["acl"] = [other_acl]
                    negative_request = self._request_for_query(query_payload, page_size=10)
                    negative_request.filters = negative_filters
                    negative_response = self._service.search(negative_request)
                    if negative_response.results:
                        violations.append(f"cross-namespace:{namespace}->{other_namespace}")
                        break

        details = {
            "queries_checked": checked,
            "violations": violations,
            "namespaces": sorted(namespace_to_acl.keys()),
        }
        passed = not violations
        return ValidationReport(name="scale_acl", passed=passed, details=details)

    def _scale_performance(
        self,
        dataset: Sequence[Mapping[str, object]],
        thresholds: Mapping[str, float],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Benchmark latency and throughput against configured thresholds.

        Args:
            dataset: Dataset entries containing documents and queries.
            thresholds: Threshold values for latency and headroom metrics.
            rng: Random generator used for sampling query pairs.
            query_sample_size: Number of queries to sample for benchmarking.

        Returns:
            ValidationReport detailing performance metrics.
        """
        sampled_pairs = self._sample_queries(dataset, min(120, query_sample_size * 2), rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_performance",
                passed=False,
                details={"error": "no queries available"},
            )

        total_timings: List[float] = []
        bm25_timings: List[float] = []
        splade_timings: List[float] = []
        dense_timings: List[float] = []
        wall_start = time.perf_counter()

        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=10)
            iter_start = time.perf_counter()
            response = self._service.search(request)
            total_timings.append(
                response.timings_ms.get("total_ms", (time.perf_counter() - iter_start) * 1000)
            )
            bm25_timings.append(response.timings_ms.get("bm25_ms", 0.0))
            splade_timings.append(response.timings_ms.get("splade_ms", 0.0))
            dense_timings.append(response.timings_ms.get("dense_ms", 0.0))

        wall_elapsed = time.perf_counter() - wall_start
        qps = len(sampled_pairs) / wall_elapsed if wall_elapsed > 0 else float("inf")

        p50 = self._percentile(total_timings, 50)
        p95 = self._percentile(total_timings, 95)
        p99 = self._percentile(total_timings, 99)

        estimated_usage_mb = (
            self._ingestion.faiss_index.ntotal * self._ingestion.faiss_index._dim * 4
        ) / (1024 * 1024)
        assumed_capacity_mb = 24000.0
        headroom_fraction = (
            1.0
            if assumed_capacity_mb <= 0
            else max(0.0, 1.0 - (estimated_usage_mb / assumed_capacity_mb))
        )

        details = {
            "query_count": len(sampled_pairs),
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "bm25_avg_ms": statistics.mean(bm25_timings) if bm25_timings else 0.0,
            "splade_avg_ms": statistics.mean(splade_timings) if splade_timings else 0.0,
            "dense_avg_ms": statistics.mean(dense_timings) if dense_timings else 0.0,
            "observed_qps": qps,
            "estimated_vector_memory_mb": estimated_usage_mb,
            "headroom_fraction": headroom_fraction,
        }

        passed = p95 <= thresholds.get(
            "latency_p95_ms", float("inf")
        ) and headroom_fraction >= thresholds.get("gpu_headroom_fraction", 0.0)
        return ValidationReport(name="scale_performance", passed=passed, details=details)

    def _run_calibration(self, dataset: Sequence[Mapping[str, object]]) -> ValidationReport:
        """Record calibration metrics (self-hit accuracy) across oversampling factors.

        Args:
            dataset: Dataset entries (unused, provided for symmetry).

        Returns:
            ValidationReport summarizing calibration accuracy per oversample setting.
        """
        oversamples = [1, 2, 3]
        results: List[Mapping[str, object]] = []
        chunks = list(self._registry.all())
        total_chunks = max(1, len(chunks))

        # Derive a defensible clamp for calibration batch sizing. The FAISS GPU
        # wheel comfortably serves ~8 MiB of query vectors per sweep, so clamp
        # the batch size according to embedding dimensionality and registry
        # cardinality. This prevents misconfiguration from issuing oversized
        # batches that could exhaust device memory.
        embedding_dim = getattr(self._ingestion.faiss_index, "_dim", 0)
        safe_min_batch = 1
        default_ceiling = 512
        if isinstance(embedding_dim, int) and embedding_dim > 0:
            bytes_per_vector = max(1, embedding_dim * 4)
            approx_limit = (8 * 1024 * 1024) // bytes_per_vector
            default_ceiling = max(safe_min_batch, min(512, approx_limit))
        available_chunks = len(chunks)
        safe_max_batch = min(default_ceiling, available_chunks) if available_chunks else default_ceiling
        safe_max_batch = max(safe_min_batch, safe_max_batch)

        config_manager = getattr(self._service, "_config_manager", None)
        retrieval_cfg = None
        if config_manager is not None:
            config = config_manager.get()
            retrieval_cfg = getattr(config, "retrieval", None)
        batch_size_raw = (
            getattr(retrieval_cfg, "dense_calibration_batch_size", None)
            if retrieval_cfg is not None
            else None
        )
        try:
            candidate_batch = int(batch_size_raw) if batch_size_raw is not None else None
        except (TypeError, ValueError):
            batch_size = None
        if batch_size is None or batch_size <= 0:
            batch_size = max(1, len(chunks))
        embedding_cache: Dict[str, np.ndarray] = {}
        missing_vectors: set[str] = set()
        for oversample in oversamples:
            hits = 0
            top_k = max(1, oversample * 3)
            if chunks:
                for start in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[start : start + batch_size]
                    candidate_pairs: List[tuple[ChunkPayload, str]] = [
                        (chunk, chunk.vector_id)
                        for chunk in batch_chunks
                        if chunk.vector_id not in missing_vectors
                    ]
                    if not candidate_pairs:
                        continue
                    vector_ids = [vector_id for _, vector_id in candidate_pairs]
                    try:
                        embedding_matrix = self._registry.resolve_embeddings(
                            vector_ids, cache=embedding_cache
                        )
                        valid_chunks = [chunk for chunk, _ in candidate_pairs]
                    except KeyError:
                        embedding_rows: list[np.ndarray] = []
                        valid_chunks = []
                        for chunk, vector_id in candidate_pairs:
                            try:
                                row = self._registry.resolve_embedding(
                                    vector_id, cache=embedding_cache
                                )
                            except KeyError:
                                missing_vectors.add(vector_id)
                                continue
                            embedding_rows.append(np.asarray(row, dtype=np.float32))
                            valid_chunks.append(chunk)
                        if not embedding_rows:
                            continue
                        embedding_matrix = np.ascontiguousarray(
                            np.stack(embedding_rows), dtype=np.float32
                        )
                    if embedding_matrix.size == 0:
                        continue
                    if hasattr(self._ingestion.faiss_index, "last_vector_ids"):
                        self._ingestion.faiss_index.last_vector_ids = [
                            chunk.vector_id for chunk in valid_chunks
                        ]
                    queries = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
                    batch_hits = self._ingestion.faiss_index.search_batch(queries, top_k)
                    if not batch_hits:
                        continue
                    for chunk, hits_list in zip(valid_chunks, batch_hits):
                        if hits_list and hits_list[0].vector_id == chunk.vector_id:
                            hits += 1
            accuracy = hits / total_chunks
            results.append({"oversample": oversample, "self_hit_accuracy": accuracy})
        passed = all(
            entry["self_hit_accuracy"] >= 0.95 for entry in results if entry["oversample"] >= 2
        )
        return ValidationReport(name="calibration_sweep", passed=passed, details={"dense": results})

    def _embeddings_for_results(
        self,
        results: Sequence[HybridSearchResult],
        chunk_lookup: Mapping[tuple[str, str, str], ChunkPayload],
        limit: int = 10,
    ) -> List[np.ndarray]:
        """Retrieve embeddings for the top-N results using a chunk lookup.

        Args:
            results: Search results from which embeddings are needed.
            chunk_lookup: Mapping from (namespace, doc_id, chunk_id) to stored payloads.
            limit: Maximum number of results to consider.

        Returns:
            List of embedding vectors associated with the results.
        """
        embeddings: List[np.ndarray] = []
        cache: Dict[str, np.ndarray] = {}
        for result in results[:limit]:
            chunk = chunk_lookup.get((result.namespace, result.doc_id, result.chunk_id))
            if chunk is None:
                continue
            embeddings.append(self._registry.resolve_embedding(chunk.vector_id, cache=cache))
        return embeddings

    def _average_pairwise_cos(self, embeddings: Sequence[np.ndarray]) -> float:
        """Compute average pairwise cosine similarity for a set of embeddings.

        Args:
            embeddings: Sequence of embedding vectors.

        Returns:
            Mean pairwise cosine similarity, or 0.0 when insufficient points exist.
        """
        if len(embeddings) < 2:
            return 0.0
        matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
        config = self._service._config_manager.get()
        sims = pairwise_inner_products(
            matrix,
            device=int(config.dense.device),
            resources=self._ensure_validation_resources(),
        )
        upper = np.triu(sims, k=1)
        values = upper[np.triu_indices_from(upper, k=1)]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    def _ensure_validation_resources(self) -> "faiss.StandardGpuResources":
        """Lazy-create and cache GPU resources for validation-only cosine checks.

        Args:
            None

        Returns:
            FAISS GPU resources reused across validation runs.
        """

        if faiss is None:
            raise RuntimeError("faiss is required for validation checks")
        if self._validation_resources is None:
            service = self._service
            observability = getattr(service, "_observability", None)
            logger = getattr(observability, "logger", None)

            candidate_stores: Sequence[DenseVectorStore] = ()
            router = getattr(service, "_faiss_router", None)
            if router is not None:
                iter_fn = getattr(router, "iter_stores", None)
                if callable(iter_fn):
                    try:
                        candidate_stores = [store for _, store in iter_fn()]
                    except Exception:
                        candidate_stores = ()
                else:
                    default_store = getattr(router, "default_store", None)
                    if default_store is not None:
                        candidate_stores = (default_store,)
            else:
                default_store = getattr(service, "_faiss", None)
                if default_store is not None:
                    candidate_stores = (default_store,)

            for store in candidate_stores:
                getter = getattr(store, "get_gpu_resources", None)
                if not callable(getter):
                    continue
                try:
                    reused = getter()
                except Exception:
                    continue
                if reused is not None:
                    self._validation_resources = reused
                    break

            if self._validation_resources is None:
                config = service._config_manager.get()
                dense_cfg = getattr(config, "dense", DenseIndexConfig())
                resource = faiss.StandardGpuResources()

                temp_memory_raw = getattr(dense_cfg, "gpu_temp_memory_bytes", None)
                try:
                    temp_memory: Optional[int] = (
                        int(temp_memory_raw) if temp_memory_raw is not None else None
                    )
                except (TypeError, ValueError):
                    temp_memory = None
                if temp_memory is not None and hasattr(resource, "setTempMemory"):
                    try:
                        resource.setTempMemory(temp_memory)
                    except Exception:
                        if logger is not None:
                            logger.debug(
                                "validation-gpu-temp-memory-config-failed",
                                exc_info=True,
                            )

                pinned_raw = getattr(dense_cfg, "gpu_pinned_memory_bytes", None)
                try:
                    pinned_memory: Optional[int] = (
                        int(pinned_raw) if pinned_raw is not None else None
                    )
                except (TypeError, ValueError):
                    pinned_memory = None
                if pinned_memory is not None and hasattr(resource, "setPinnedMemory"):
                    try:
                        resource.setPinnedMemory(pinned_memory)
                    except Exception:
                        if logger is not None:
                            logger.debug(
                                "validation-gpu-pinned-memory-config-failed",
                                exc_info=True,
                            )

                use_null_all = bool(
                    getattr(dense_cfg, "gpu_use_default_null_stream_all_devices", False)
                    or getattr(dense_cfg, "gpu_default_null_stream_all_devices", False)
                )
                use_null = bool(
                    getattr(dense_cfg, "gpu_use_default_null_stream", False)
                    or getattr(dense_cfg, "gpu_default_null_stream", False)
                )
                try:
                    device = int(getattr(dense_cfg, "device", 0))
                except (TypeError, ValueError):
                    device = 0

                if use_null_all:
                    method = getattr(resource, "setDefaultNullStreamAllDevices", None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            if logger is not None:
                                logger.debug(
                                    "validation-gpu-null-stream-all-devices-failed",
                                    exc_info=True,
                                )
                elif use_null:
                    method = getattr(resource, "setDefaultNullStream", None)
                    if callable(method):
                        try:
                            method(device)
                        except TypeError:
                            try:
                                method()
                            except Exception:
                                if logger is not None:
                                    logger.debug(
                                        "validation-gpu-null-stream-config-failed",
                                        exc_info=True,
                                    )
                        except Exception:
                            if logger is not None:
                                logger.debug(
                                    "validation-gpu-null-stream-config-failed",
                                    exc_info=True,
                                )

                if observability is not None:
                    metrics = getattr(observability, "metrics", None)
                    if metrics is not None and temp_memory is not None:
                        metrics.set_gauge(
                            "faiss_gpu_temp_memory_bytes", float(temp_memory), scope="validation"
                        )
                    if metrics is not None:
                        metrics.set_gauge(
                            "faiss_gpu_default_null_stream",
                            1.0 if use_null else 0.0,
                            scope="validation",
                        )
                        metrics.set_gauge(
                            "faiss_gpu_default_null_stream_all_devices",
                            1.0 if use_null_all else 0.0,
                            scope="validation",
                        )
                    if logger is not None:
                        logger.info(
                            "faiss-gpu-resource-configured",
                            extra={
                                "event": {
                                    "device": device,
                                    "temp_memory_bytes": temp_memory,
                                    "default_null_stream": use_null,
                                    "default_null_stream_all_devices": use_null_all,
                                    "scope": "validation",
                                }
                            },
                        )

                self._validation_resources = resource
        return self._validation_resources

    def _percentile(self, values: Sequence[float], percentile: float) -> float:
        """Return percentile value for a sequence; defaults to 0.0 when empty.

        Args:
            values: Sequence of numeric values.
            percentile: Desired percentile expressed as a value between 0 and 100.

        Returns:
            Percentile value cast to float, or 0.0 when the sequence is empty.
        """
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))

    def _scale_stability(
        self,
        dataset: Sequence[Mapping[str, object]],
        inputs_by_doc: Mapping[str, DocumentInput],
        rng: random.Random,
        query_sample_size: int,
    ) -> ValidationReport:
        """Stress-test search stability under repeated queries and churn.

        Args:
            dataset: Dataset entries containing documents and queries.
            inputs_by_doc: Mapping of document IDs to ingestion inputs.
            rng: Random generator used for sampling queries and churn operations.
            query_sample_size: Number of queries to sample for stability checks.

        Returns:
            ValidationReport detailing stability mismatches.
        """
        sampled_pairs = self._sample_queries(dataset, min(40, query_sample_size), rng)
        if not sampled_pairs:
            return ValidationReport(
                name="scale_stability",
                passed=False,
                details={"error": "no queries available"},
            )

        repeat_mismatches = 0
        for _, query_payload in sampled_pairs:
            request = self._request_for_query(query_payload, page_size=15)
            baseline = [res.doc_id for res in self._service.search(request).results]
            for _ in range(4):
                repeat_request = self._request_for_query(query_payload, page_size=15)
                repeat_results = [
                    res.doc_id for res in self._service.search(repeat_request).results
                ]
                if repeat_results != baseline:
                    repeat_mismatches += 1
                    break

        churn_doc_ids = list(inputs_by_doc.keys())
        churn_count = min(30, len(churn_doc_ids))
        if churn_count:
            churn_samples = rng.sample(churn_doc_ids, churn_count)
            self._ingestion.upsert_documents([inputs_by_doc[doc_id] for doc_id in churn_samples])

        churn_failures = 0
        for document_payload, query_payload in sampled_pairs[: min(15, len(sampled_pairs))]:
            expected_doc_id = str(
                query_payload.get("expected_doc_id") or document_payload.get("doc_id")
            )
            request = self._request_for_query(query_payload, page_size=15)
            response = self._service.search(request)
            if expected_doc_id and not any(
                res.doc_id == expected_doc_id for res in response.results
            ):
                churn_failures += 1

        details = {
            "queries_checked": len(sampled_pairs),
            "repeat_mismatches": repeat_mismatches,
            "churn_failures": churn_failures,
        }
        passed = repeat_mismatches == 0 and churn_failures == 0
        return ValidationReport(name="scale_stability", passed=passed, details=details)


# --- Public Functions ---


class JsonlDataset(Sequence[Mapping[str, object]]):
    """Memory-efficient JSONL dataset that streams entries on demand."""

    __slots__ = ("_path", "_encoding", "_offsets")

    def __init__(self, path: Path, encoding: str = "utf-8") -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self._path = path
        self._encoding = encoding
        offsets: List[int] = []
        offset = 0
        with path.open("rb") as handle:
            for raw in handle:
                length = len(raw)
                if raw.strip():
                    offsets.append(offset)
                offset += length
        self._offsets = offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int | slice) -> Mapping[str, object]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        offset = self._offsets[index]
        with self._path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.readline()
        return json.loads(raw.decode(self._encoding))

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        with self._path.open("r", encoding=self._encoding) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_dataset(path: Path) -> JsonlDataset:
    """Load a JSONL dataset describing documents and queries."""

    return JsonlDataset(path)


def infer_embedding_dim(dataset: Sequence[Mapping[str, object]]) -> int:
    """Infer dense embedding dimensionality from dataset vector artifacts.

    Args:
        dataset: Sequence of dataset entries containing vector metadata.

    Returns:
        Inferred embedding dimensionality, defaulting to 2560 when unknown.
    """

    for entry in dataset:
        document = entry.get("document", {})
        vector_file = document.get("vector_file")
        if not vector_file:
            continue
        path = Path(str(vector_file))
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "Parquet vector ingestion requires the optional dependency 'pyarrow'. "
                    "Install DocsToKG[docparse-parquet] or add pyarrow to the environment."
                ) from exc
            try:
                parquet_file = pq.ParquetFile(path)
            except Exception:  # pragma: no cover - IO errors
                continue
            for record_batch in parquet_file.iter_batches(batch_size=1):
                for payload in record_batch.to_pylist():
                    metadata = payload.get("model_metadata")
                    if isinstance(metadata, str) and metadata:
                        try:
                            payload["model_metadata"] = json.loads(metadata)
                        except json.JSONDecodeError:
                            payload["model_metadata"] = {}
                    vector = (payload.get("Qwen3-4B") or payload.get("Qwen3_4B") or {}).get(
                        "vector"
                    )
                    if isinstance(vector, list) and vector:
                        return len(vector)
        else:
            with path.open(encoding="utf-8") as stream:
                for raw_line in stream:
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    vector_payload = payload.get("Qwen3-4B") or payload.get("Qwen3_4B") or {}
                    vector = vector_payload.get("vector")
                    if isinstance(vector, list) and vector:
                        return len(vector)
    return 2560
