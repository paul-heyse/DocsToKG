"""Reciprocal Rank Fusion and diversification utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .similarity import cosine_against_corpus_gpu
from .types import FusionCandidate


class ReciprocalRankFusion:
    """Combine ranked lists using Reciprocal Rank Fusion.

    Attributes:
        _k0: Fusion parameter controlling the influence of item rank.

    Examples:
        >>> rrf = ReciprocalRankFusion(k0=50.0)
        >>> rrf.fuse([])
        {}
    """

    def __init__(self, k0: float = 60.0) -> None:
        if k0 <= 0:
            raise ValueError("k0 must be positive")
        self._k0 = k0

    def fuse(self, candidates: Sequence[FusionCandidate]) -> Dict[str, float]:
        """Score candidates using reciprocal rank fusion.

        Args:
            candidates: Ordered sequence of fusion candidates.

        Returns:
            Mapping of vector IDs to aggregated RRF scores.
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
) -> List[FusionCandidate]:
    """Apply maximal marginal relevance to promote diversity.

    Args:
        fused_candidates: Candidates ranked by initial fusion stage.
        fused_scores: Pre-computed relevance scores keyed by vector ID.
        lambda_param: Balance factor between relevance and diversity [0,1].
        top_k: Number of candidates to retain after diversification.

    Returns:
        List of diversified candidates ordered by selection.
    """
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be within [0, 1]")
    if not fused_candidates:
        return []

    embeddings = np.stack(
        [
            candidate.chunk.features.embedding.astype(np.float32, copy=False)
            for candidate in fused_candidates
        ]
    )

    candidate_indices = list(range(len(fused_candidates)))
    selected_indices: List[int] = []

    while candidate_indices and len(selected_indices) < top_k:
        best_idx: int | None = None
        best_score = float("-inf")
        for idx in candidate_indices:
            vector_id = fused_candidates[idx].chunk.vector_id
            relevance = fused_scores.get(vector_id, 0.0)
            if selected_indices:
                selected_embeddings = embeddings[selected_indices]
                sims = cosine_against_corpus_gpu(embeddings[idx], selected_embeddings)
                diversity_penalty = float(sims[0].max())
            else:
                diversity_penalty = 0.0
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        candidate_indices = [idx for idx in candidate_indices if idx != best_idx]

    return [fused_candidates[idx] for idx in selected_indices]
