"""Reciprocal Rank Fusion and diversification utilities."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .types import FusionCandidate


class ReciprocalRankFusion:
    """Combine ranked lists using Reciprocal Rank Fusion."""

    def __init__(self, k0: float = 60.0) -> None:
        if k0 <= 0:
            raise ValueError("k0 must be positive")
        self._k0 = k0

    def fuse(self, candidates: Sequence[FusionCandidate]) -> Dict[str, float]:
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
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be within [0, 1]")
    remaining = list(fused_candidates)
    selected: List[FusionCandidate] = []
    while remaining and len(selected) < top_k:
        best_candidate = None
        best_score = float("-inf")
        for candidate in remaining:
            relevance = fused_scores[candidate.chunk.vector_id]
            diversity_penalty = 0.0
            if selected:
                similarities = [
                    _cosine_similarity(candidate.chunk.features.embedding, other.chunk.features.embedding)
                    for other in selected
                ]
                diversity_penalty = max(similarities)
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_candidate = candidate
                best_score = score
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining = [candidate for candidate in remaining if candidate.chunk.vector_id != best_candidate.chunk.vector_id]
    return selected


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom == 0.0:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)

