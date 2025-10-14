"""Result shaping utilities for hybrid search responses."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .config import FusionConfig
from .similarity import max_inner_product, normalize_rows
from .storage import OpenSearchSimulator
from .tokenization import tokenize
from .types import (
    ChunkPayload,
    HybridSearchDiagnostics,
    HybridSearchRequest,
    HybridSearchResult,
)


class ResultShaper:
    """Collapse duplicates, enforce quotas, and generate highlights."""

    def __init__(self, opensearch: OpenSearchSimulator, fusion_config: FusionConfig) -> None:
        self._opensearch = opensearch
        self._fusion_config = fusion_config

    def shape(
        self,
        ordered_chunks: Sequence[ChunkPayload],
        fused_scores: Mapping[str, float],
        request: HybridSearchRequest,
        channel_scores: Mapping[str, Dict[str, float]],
    ) -> List[HybridSearchResult]:
        doc_buckets: Dict[str, int] = defaultdict(int)
        emitted_vectors: List[str] = []
        emitted_embeddings: List[np.ndarray] = []
        results: List[HybridSearchResult] = []
        query_tokens = tokenize(request.query)
        for rank, chunk in enumerate(ordered_chunks, start=1):
            if not self._within_doc_limit(chunk.doc_id, doc_buckets):
                continue
            embedding = self._normalize_embedding(chunk.features.embedding)
            if self._is_near_duplicate(embedding, emitted_embeddings):
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
            emitted_vectors.append(chunk.vector_id)
            emitted_embeddings.append(embedding)
        return results

    def _within_doc_limit(self, doc_id: str, doc_buckets: Dict[str, int]) -> bool:
        doc_buckets[doc_id] += 1
        return doc_buckets[doc_id] <= self._fusion_config.max_chunks_per_doc

    def _is_near_duplicate(
        self,
        embedding: np.ndarray,
        emitted_embeddings: Sequence[np.ndarray],
    ) -> bool:
        if not emitted_embeddings:
            return False
        corpus = np.stack(emitted_embeddings)
        similarity = max_inner_product(embedding, corpus)
        return similarity >= self._fusion_config.cosine_dedupe_threshold

    def _build_highlights(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        highlights = self._opensearch.highlight(chunk, query_tokens)
        if highlights:
            return highlights
        snippet = chunk.text[:200]
        return [snippet] if snippet else []

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        normalized = normalize_rows(embedding.reshape(1, -1).astype(np.float32, copy=False))
        return normalized[0]

