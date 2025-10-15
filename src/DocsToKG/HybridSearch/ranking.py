"""Ranking, fusion, and result shaping utilities for DocsToKG hybrid search."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .config import FusionConfig
from .features import tokenize
from .storage import OpenSearchSimulator
from .types import (
    ChunkPayload,
    FusionCandidate,
    HybridSearchDiagnostics,
    HybridSearchRequest,
    HybridSearchResult,
)
from .vectorstore import cosine_batch, pairwise_inner_products

if TYPE_CHECKING:  # pragma: no cover - typing only
    import faiss  # type: ignore


class ReciprocalRankFusion:
    """Combine ranked lists using Reciprocal Rank Fusion.

    Attributes:
        _k0: Smoothing constant used in the reciprocal rank formula.

    Examples:
        >>> fusion = ReciprocalRankFusion(k0=60.0)
        >>> scores = fusion.fuse([])
        >>> scores
        {}
    """

    def __init__(self, k0: float = 60.0) -> None:
        """Create a new fusion helper.

        Args:
            k0: Smoothing constant added to ranks before inversion.

        Raises:
            ValueError: If ``k0`` is not strictly positive.

        Returns:
            None
        """
        if k0 <= 0:
            raise ValueError("k0 must be positive")
        self._k0 = k0

    def fuse(self, candidates: Sequence[FusionCandidate]) -> Dict[str, float]:
        """Fuse ranked candidates using Reciprocal Rank Fusion.

        Args:
            candidates: Ranked candidates from individual retrieval channels.

        Returns:
            Dict[str, float]: Mapping of vector IDs to fused reciprocal-rank scores.
        """
        scores: Dict[str, float] = defaultdict(float)
        for candidate in candidates:
            contribution = 1.0 / (self._k0 + candidate.rank)
            scores[candidate.chunk.vector_id] += contribution
        return dict(scores)


def apply_mmr_diversification(
    fused_candidates: Sequence[FusionCandidate],
    fused_scores: Mapping[str, float],
    lambda_param: float,
    top_k: int,
    *,
    device: int = 0,
    resources: Optional["faiss.StandardGpuResources"] = None,
) -> List[FusionCandidate]:
    """Apply maximal marginal relevance to promote diversity.

    Args:
        fused_candidates: Ordered fusion candidates before diversification.
        fused_scores: Precomputed fused scores for each candidate vector ID.
        lambda_param: Balancing factor between relevance and diversity (0-1).
        top_k: Maximum number of diversified candidates to retain.
        device: GPU device identifier used for similarity computations.
        resources: FAISS GPU resources used for pairwise similarity.

    Returns:
        List[FusionCandidate]: Diversified candidate list ordered by MMR score.

    Raises:
        ValueError: If ``lambda_param`` falls outside ``[0, 1]``.
        RuntimeError: When GPU resources are not available for diversification.
    """

    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be within [0, 1]")
    if not fused_candidates:
        return []
    if resources is None:
        raise RuntimeError("GPU resources are required for MMR diversification")

    embeddings = np.stack(
        [
            candidate.chunk.features.embedding.astype(np.float32, copy=False)
            for candidate in fused_candidates
        ]
    )

    candidate_indices = list(range(len(fused_candidates)))
    selected_indices: List[int] = []
    sims_all = pairwise_inner_products(embeddings, device=int(device), resources=resources)

    while candidate_indices and len(selected_indices) < top_k:
        best_idx: Optional[int] = None
        best_score = float("-inf")
        for idx in candidate_indices:
            vector_id = fused_candidates[idx].chunk.vector_id
            relevance = fused_scores.get(vector_id, 0.0)
            diversity_penalty = (
                float(sims_all[idx, selected_indices].max()) if selected_indices else 0.0
            )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        candidate_indices = [idx for idx in candidate_indices if idx != best_idx]

    return [fused_candidates[idx] for idx in selected_indices]


class ResultShaper:
    """Collapse duplicates, enforce quotas, and generate highlights.

    Attributes:
        _opensearch: Simulator providing metadata and highlights.
        _fusion_config: Fusion configuration controlling result shaping.
        _gpu_device: CUDA device used for optional similarity checks.
        _gpu_resources: Optional FAISS resources for GPU pairwise similarity.

    Examples:
        >>> shaper = ResultShaper(OpenSearchSimulator(), FusionConfig())  # doctest: +SKIP
        >>> shaper.shape([], {}, HybridSearchRequest(query="", namespace=None, filters={}, page_size=1), {})  # doctest: +SKIP
        []
    """

    def __init__(
        self,
        opensearch: OpenSearchSimulator,
        fusion_config: FusionConfig,
        *,
        device: int = 0,
        resources: Optional["faiss.StandardGpuResources"] = None,
    ) -> None:
        """Initialise the shaper with supporting components.

        Args:
            opensearch: Simulator used to fetch highlights and metadata.
            fusion_config: Fusion configuration influencing dedupe limits.
            device: CUDA device index for GPU-assisted operations.
            resources: Optional FAISS GPU resources passed to similarity helpers.

        Returns:
            None
        """
        self._opensearch = opensearch
        self._fusion_config = fusion_config
        self._gpu_device = int(device)
        self._gpu_resources = resources

    def shape(
        self,
        ordered_chunks: Sequence[ChunkPayload],
        fused_scores: Mapping[str, float],
        request: HybridSearchRequest,
        channel_scores: Mapping[str, Dict[str, float]],
    ) -> List[HybridSearchResult]:
        """Shape ranked candidates into final results with highlights and diagnostics.

        Args:
            ordered_chunks: Ranked chunks produced by hybrid fusion.
            fused_scores: Mapping of vector IDs to fused scores.
            request: Original hybrid search request, used for query context.
            channel_scores: Channel-specific score lookups keyed by resolver name.

        Returns:
            List[HybridSearchResult]: Finalised results respecting per-document quotas
            and duplicate suppression.
        """
        if not ordered_chunks:
            return []

        embeddings = np.ascontiguousarray(
            np.stack(
                [
                    chunk.features.embedding.astype(np.float32, copy=False)
                    for chunk in ordered_chunks
                ]
            ),
            dtype=np.float32,
        )
        pairwise: Optional[np.ndarray] = None
        if self._gpu_resources is not None and len(ordered_chunks) <= 256:
            pairwise = pairwise_inner_products(
                embeddings, device=self._gpu_device, resources=self._gpu_resources
            )

        doc_buckets: Dict[str, int] = defaultdict(int)
        emitted_indices: List[int] = []
        results: List[HybridSearchResult] = []
        query_tokens = tokenize(request.query)
        for current_idx, chunk in enumerate(ordered_chunks):
            rank = current_idx + 1
            if not self._within_doc_limit(chunk.doc_id, doc_buckets):
                continue
            if emitted_indices and self._is_near_duplicate(
                embeddings, current_idx, emitted_indices, pairwise
            ):
                continue
            highlights = self._build_highlights(chunk, query_tokens)
            diagnostics = HybridSearchDiagnostics(
                bm25_score=channel_scores.get("bm25", {}).get(chunk.vector_id),
                splade_score=channel_scores.get("splade", {}).get(chunk.vector_id),
                dense_score=channel_scores.get("dense", {}).get(chunk.vector_id),
            )
            provenance = [chunk.char_offset] if chunk.char_offset else []
            results.append(
                HybridSearchResult(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    namespace=chunk.namespace,
                    score=fused_scores[chunk.vector_id],
                    fused_rank=rank,
                    text=chunk.text,
                    highlights=highlights,
                    provenance_offsets=provenance,
                    diagnostics=diagnostics,
                    metadata=dict(chunk.metadata),
                )
            )
            emitted_indices.append(current_idx)
        return results

    def _within_doc_limit(self, doc_id: str, doc_buckets: Dict[str, int]) -> bool:
        doc_buckets[doc_id] += 1
        return doc_buckets[doc_id] <= self._fusion_config.max_chunks_per_doc

    def _is_near_duplicate(
        self,
        embeddings: np.ndarray,
        current_idx: int,
        emitted_indices: Sequence[int],
        pairwise: Optional[np.ndarray] = None,
    ) -> bool:
        if not emitted_indices:
            return False
        if pairwise is not None:
            return (
                float(pairwise[current_idx, emitted_indices].max())
                >= self._fusion_config.cosine_dedupe_threshold
            )
        resources = self._gpu_resources
        if resources is None:
            query = embeddings[current_idx]
            others = embeddings[list(emitted_indices)]
            query_norm = np.linalg.norm(query)
            if query_norm == 0.0:
                return False
            other_norms = np.linalg.norm(others, axis=1)
            other_norms[other_norms == 0.0] = 1.0
            sims = (others @ query) / (other_norms * query_norm)
            return float(sims.max()) >= self._fusion_config.cosine_dedupe_threshold
        query = embeddings[current_idx]
        corpus = embeddings[list(emitted_indices)]
        sims = cosine_batch(query, corpus, device=self._gpu_device, resources=resources)
        return float(sims[0].max()) >= self._fusion_config.cosine_dedupe_threshold

    def _build_highlights(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        highlights = self._opensearch.highlight(chunk, query_tokens)
        if highlights:
            return highlights
        tokens = tokenize(chunk.text)
        matches = [token for token in tokens if token in set(query_tokens)]
        if matches:
            return [" ".join(tokens[: min(len(tokens), 40)])]
        return [chunk.text[: min(len(chunk.text), 200)]]


__all__ = ["ReciprocalRankFusion", "apply_mmr_diversification", "ResultShaper"]
