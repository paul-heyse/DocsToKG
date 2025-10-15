"""Result shaping utilities for hybrid search responses."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .config import FusionConfig
from .similarity import cosine_against_corpus_gpu
from .storage import OpenSearchSimulator
from .tokenization import tokenize
from .types import (
    ChunkPayload,
    HybridSearchDiagnostics,
    HybridSearchRequest,
    HybridSearchResult,
)


class ResultShaper:
    """Collapse duplicates, enforce quotas, and generate highlights.

    Attributes:
        _opensearch: OpenSearch simulator providing highlighting hooks.
        _fusion_config: Fusion configuration controlling dedupe thresholds and quotas.

    Examples:
        >>> shaper = ResultShaper(OpenSearchSimulator(), FusionConfig())
        >>> shaper._fusion_config.max_chunks_per_doc
        3
    """

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
        """Transform ordered chunks into shaped search results.

        Args:
            ordered_chunks: Ranked chunk payloads emitted by fusion.
            fused_scores: Final fused score per vector ID.
            request: Original hybrid search request used for context.
            channel_scores: Per-channel scoring maps keyed by vector ID.

        Returns:
            List of `HybridSearchResult` objects ready for response serialization.
        """
        if not ordered_chunks:
            return []

        embeddings = np.stack(
            [chunk.features.embedding.astype(np.float32, copy=False) for chunk in ordered_chunks]
        )

        doc_buckets: Dict[str, int] = defaultdict(int)
        emitted_indices: List[int] = []
        results: List[HybridSearchResult] = []
        query_tokens = tokenize(request.query)
        for current_idx, chunk in enumerate(ordered_chunks):
            rank = current_idx + 1
            if not self._within_doc_limit(chunk.doc_id, doc_buckets):
                continue
            if emitted_indices and self._is_near_duplicate(embeddings, current_idx, emitted_indices):
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
        """Check and update per-document emission counts.

        Args:
            doc_id: Document identifier being considered for emission.
            doc_buckets: Mutable counter of chunks emitted per document.

        Returns:
            True if the document is still below the configured limit.
        """
        doc_buckets[doc_id] += 1
        return doc_buckets[doc_id] <= self._fusion_config.max_chunks_per_doc

    def _is_near_duplicate(
        self,
        embeddings: np.ndarray,
        current_idx: int,
        emitted_indices: Sequence[int],
    ) -> bool:
        """Determine whether the current chunk is too similar to emitted ones."""

        if not emitted_indices:
            return False
        query = embeddings[current_idx]
        corpus = embeddings[list(emitted_indices)]
        sims = cosine_against_corpus_gpu(query, corpus)
        return float(sims[0].max()) >= self._fusion_config.cosine_dedupe_threshold

    def _build_highlights(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        """Generate highlight snippets for a chunk.

        Args:
            chunk: Chunk payload being rendered.
            query_tokens: Tokens derived from the user's query.

        Returns:
            List of highlight strings; falls back to a snippet when necessary.
        """
        highlights = self._opensearch.highlight(chunk, query_tokens)
        if highlights:
            return highlights
        snippet = chunk.text[:200]
        return [snippet] if snippet else []
