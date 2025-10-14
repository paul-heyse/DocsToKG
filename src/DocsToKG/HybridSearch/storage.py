"""In-memory storage simulators for OpenSearch and chunk registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .ids import vector_uuid_to_faiss_int
from .tokenization import tokenize
from .types import ChunkPayload


def matches_filters(chunk: ChunkPayload, filters: Mapping[str, object]) -> bool:
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

if TYPE_CHECKING:  # pragma: no cover - import guard for type checking only
    from .schema import OpenSearchIndexTemplate


@dataclass(slots=True)
class StoredChunk:
    """Internal representation of a chunk stored in the OpenSearch simulator."""

    payload: ChunkPayload


class ChunkRegistry:
    """Durable mapping of `vector_id` → `ChunkPayload` for joins and reconciliation."""

    def __init__(self) -> None:
        self._chunks: Dict[str, ChunkPayload] = {}
        self._bridge: Dict[int, str] = {}

    def upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        for chunk in chunks:
            self._chunks[chunk.vector_id] = chunk
            self._bridge[vector_uuid_to_faiss_int(chunk.vector_id)] = chunk.vector_id

    def delete(self, vector_ids: Sequence[str]) -> None:
        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
            self._bridge.pop(vector_uuid_to_faiss_int(vector_id), None)

    def get(self, vector_id: str) -> Optional[ChunkPayload]:
        return self._chunks.get(vector_id)

    def bulk_get(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        return [self._chunks[vid] for vid in vector_ids if vid in self._chunks]

    def resolve_faiss_id(self, internal_id: int) -> Optional[str]:
        return self._bridge.get(internal_id)

    def all(self) -> List[ChunkPayload]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)

    def vector_ids(self) -> List[str]:
        return list(self._chunks.keys())


class OpenSearchSimulator:
    """Subset of OpenSearch capabilities required for hybrid retrieval tests."""

    def __init__(self) -> None:
        self._chunks: Dict[str, StoredChunk] = {}
        self._avg_length: float = 0.0
        self._templates: Dict[str, "OpenSearchIndexTemplate"] = {}

    def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        for chunk in chunks:
            self._chunks[chunk.vector_id] = StoredChunk(chunk)
        self._recompute_avg_length()

    def bulk_delete(self, vector_ids: Sequence[str]) -> None:
        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
        self._recompute_avg_length()

    def fetch(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        return [self._chunks[vid].payload for vid in vector_ids if vid in self._chunks]

    def vector_ids(self) -> List[str]:
        return list(self._chunks.keys())

    def register_template(self, template: "OpenSearchIndexTemplate") -> None:
        self._templates[template.namespace] = template

    def template_for(self, namespace: str) -> Optional["OpenSearchIndexTemplate"]:
        return self._templates.get(namespace)

    def search_bm25(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> tuple[List[tuple[ChunkPayload, float]], Optional[int]]:
        return self._search_sparse(
            lambda stored: self._bm25_score(stored, query_weights),
            filters,
            top_k,
            cursor,
        )

    def search_splade(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> tuple[List[tuple[ChunkPayload, float]], Optional[int]]:
        def score_chunk(stored: StoredChunk) -> float:
            score = 0.0
            for token, weight in query_weights.items():
                if token in stored.payload.features.splade_weights:
                    score += weight * stored.payload.features.splade_weights[token]
            return float(score)

        return self._search_sparse(score_chunk, filters, top_k, cursor)

    def highlight(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        highlights: List[str] = []
        lower_text = chunk.text.lower()
        for token in query_tokens:
            token_lower = token.lower()
            if token_lower in lower_text:
                highlights.append(token)
        return highlights

    def _filtered_chunks(self, filters: Mapping[str, object]) -> List[StoredChunk]:
        results: List[StoredChunk] = []
        for stored in self._chunks.values():
            if matches_filters(stored.payload, filters):
                results.append(stored)
        return results

    def _bm25_score(self, stored: StoredChunk, query_weights: Mapping[str, float]) -> float:
        score = 0.0
        for token, weight in query_weights.items():
            chunk_weight = stored.payload.features.bm25_terms.get(token)
            if chunk_weight is None:
                continue
            score += weight * chunk_weight
        return float(score)

    def _paginate(
        self,
        scores: List[tuple[ChunkPayload, float]],
        top_k: int,
        cursor: Optional[int],
    ) -> tuple[List[tuple[ChunkPayload, float]], Optional[int]]:
        offset = cursor or 0
        end = offset + top_k
        page = scores[offset:end]
        next_cursor = end if end < len(scores) else None
        return page, next_cursor

    def _search_sparse(
        self,
        scoring_fn: Callable[[StoredChunk], float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int],
    ) -> tuple[List[tuple[ChunkPayload, float]], Optional[int]]:
        candidates = self._filtered_chunks(filters)
        scored: List[tuple[ChunkPayload, float]] = []
        for stored in candidates:
            score = scoring_fn(stored)
            if score > 0.0:
                scored.append((stored.payload, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return self._paginate(scored, top_k, cursor)

    def _recompute_avg_length(self) -> None:
        if not self._chunks:
            self._avg_length = 0.0
            return
        total_length = sum(chunk.payload.token_count for chunk in self._chunks.values())
        self._avg_length = total_length / len(self._chunks)

    def stats(self) -> Mapping[str, float]:
        return {
            "document_count": float(len(self._chunks)),
            "avg_token_length": float(self._avg_length),
        }
