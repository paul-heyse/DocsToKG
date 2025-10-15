"""In-memory storage simulators for OpenSearch and chunk registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence

from .ids import vector_uuid_to_faiss_int
from .types import ChunkPayload


def matches_filters(chunk: ChunkPayload, filters: Mapping[str, object]) -> bool:
    """Check whether a chunk satisfies OpenSearch-style filter conditions.

    Args:
        chunk: Chunk payload under evaluation.
        filters: Mapping of filter names to expected values or lists.

    Returns:
        True if the chunk matches all supplied filters, False otherwise.
    """
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
        """Insert or update chunk metadata for the provided vector IDs.

        Args:
            chunks: Sequence of chunks to cache by vector identifier.

        Returns:
            None
        """
        for chunk in chunks:
            self._chunks[chunk.vector_id] = chunk
            self._bridge[vector_uuid_to_faiss_int(chunk.vector_id)] = chunk.vector_id

    def delete(self, vector_ids: Sequence[str]) -> None:
        """Remove chunk entries associated with the supplied vector IDs.

        Args:
            vector_ids: Vector identifiers whose payloads should be removed.

        Returns:
            None
        """
        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
            self._bridge.pop(vector_uuid_to_faiss_int(vector_id), None)

    def get(self, vector_id: str) -> Optional[ChunkPayload]:
        """Retrieve a chunk payload by its vector identifier.

        Args:
            vector_id: Vector identifier to look up.

        Returns:
            Matching `ChunkPayload` or None if not present.
        """
        return self._chunks.get(vector_id)

    def bulk_get(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        """Return existing chunks for the provided vector identifiers.

        Args:
            vector_ids: Collection of vector identifiers to fetch.

        Returns:
            List of chunk payloads found in the registry.
        """
        return [self._chunks[vid] for vid in vector_ids if vid in self._chunks]

    def resolve_faiss_id(self, internal_id: int) -> Optional[str]:
        """Translate a FAISS internal ID back to the original vector UUID.

        Args:
            internal_id: Integer identifier assigned by FAISS.

        Returns:
            Vector UUID if the mapping exists, else None.
        """
        return self._bridge.get(internal_id)

    def all(self) -> List[ChunkPayload]:
        """Return a list of all registered chunk payloads.

        Args:
            None

        Returns:
            List of all cached chunk payloads.
        """
        return list(self._chunks.values())

    def count(self) -> int:
        """Return the total number of tracked chunks.

        Args:
            None

        Returns:
            Number of chunks currently cached.
        """
        return len(self._chunks)

    def vector_ids(self) -> List[str]:
        """List all vector identifiers in insertion order.

        Args:
            None

        Returns:
            List of vector identifiers tracked by the registry.
        """
        return list(self._chunks.keys())


class OpenSearchSimulator:
    """Subset of OpenSearch capabilities required for hybrid retrieval tests."""

    def __init__(self) -> None:
        self._chunks: Dict[str, StoredChunk] = {}
        self._avg_length: float = 0.0
        self._templates: Dict[str, "OpenSearchIndexTemplate"] = {}

    def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        """Insert or replace OpenSearch documents for the provided chunks.

        Args:
            chunks: Sequence of chunk payloads to index.

        Returns:
            None
        """
        for chunk in chunks:
            self._chunks[chunk.vector_id] = StoredChunk(chunk)
        self._recompute_avg_length()

    def bulk_delete(self, vector_ids: Sequence[str]) -> None:
        """Delete OpenSearch documents associated with the vector identifiers.

        Args:
            vector_ids: Vector identifiers whose documents should be removed.

        Returns:
            None
        """
        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
        self._recompute_avg_length()

    def fetch(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        """Fetch stored chunk payloads for a list of vector IDs.

        Args:
            vector_ids: Vector identifiers to retrieve.

        Returns:
            List of chunk payloads matching the provided identifiers.
        """
        return [self._chunks[vid].payload for vid in vector_ids if vid in self._chunks]

    def vector_ids(self) -> List[str]:
        """Return all vector identifiers currently stored in OpenSearch.

        Args:
            None

        Returns:
            List of vector identifiers present in the simulator.
        """
        return list(self._chunks.keys())

    def register_template(self, template: "OpenSearchIndexTemplate") -> None:
        """Register an index template used for namespace-specific behavior.

        Args:
            template: Template to associate with its namespace.

        Returns:
            None
        """
        self._templates[template.namespace] = template

    def template_for(self, namespace: str) -> Optional["OpenSearchIndexTemplate"]:
        """Look up an index template for the given namespace.

        Args:
            namespace: Namespace identifier whose template is needed.

        Returns:
            Matching `OpenSearchIndexTemplate`, or None when missing.
        """
        return self._templates.get(namespace)

    def search_bm25(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> tuple[List[tuple[ChunkPayload, float]], Optional[int]]:
        """Execute a BM25-style search over stored chunks.

        Args:
            query_weights: Weighted token map for the query.
            filters: Filter constraints to apply before scoring.
            top_k: Maximum number of results to return.
            cursor: Optional pagination offset.

        Returns:
            Tuple of scored results and optional next cursor.
        """
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
        """Execute a SPLADE-style sparse search over stored chunks.

        Args:
            query_weights: Weighted token map for SPLADE features.
            filters: Filter constraints applied before scoring.
            top_k: Maximum results to return.
            cursor: Optional pagination offset.

        Returns:
            Tuple of scored results and optional next cursor.
        """
        return self._search_sparse(
            lambda stored: sum(
                weight * stored.payload.features.splade_weights.get(token, 0.0)
                for token, weight in query_weights.items()
            ),
            filters,
            top_k,
            cursor,
        )

    def highlight(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        """Return basic term highlights for the given chunk.

        Args:
            chunk: Chunk payload whose text requires highlighting.
            query_tokens: Tokens derived from the query.

        Returns:
            List of highlight strings.
        """
        highlights: List[str] = []
        lower_text = chunk.text.lower()
        for token in query_tokens:
            token_lower = token.lower()
            if token_lower in lower_text:
                highlights.append(token)
        return highlights

    def _filtered_chunks(self, filters: Mapping[str, object]) -> List[StoredChunk]:
        """Return chunks that match the provided filter constraints.

        Args:
            filters: Filter mapping applied to chunk metadata.

        Returns:
            List of stored chunks satisfying the filters.
        """
        results: List[StoredChunk] = []
        for stored in self._chunks.values():
            if matches_filters(stored.payload, filters):
                results.append(stored)
        return results

    def _bm25_score(self, stored: StoredChunk, query_weights: Mapping[str, float]) -> float:
        """Compute BM25 similarity between a stored chunk and query weights.

        Args:
            stored: Chunk under evaluation.
            query_weights: Weighted token map for the query.

        Returns:
            BM25 score for the chunk.
        """
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
        """Slice scored results according to the pagination cursor.

        Args:
            scores: Pre-sorted list of chunk-score tuples.
            top_k: Maximum number of items to emit.
            cursor: Optional offset into the ranked list.

        Returns:
            Tuple of the current page and optional next cursor.
        """
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
        """Search sparse features using the supplied scoring function.

        Args:
            scoring_fn: Callable to compute a score for each stored chunk.
            filters: Filter mapping applied prior to scoring.
            top_k: Maximum number of results to return.
            cursor: Optional pagination offset.

        Returns:
            Tuple of scored results and optional next cursor.
        """
        candidates = self._filtered_chunks(filters)
        scored: List[tuple[ChunkPayload, float]] = []
        for stored in candidates:
            score = scoring_fn(stored)
            if score > 0.0:
                scored.append((stored.payload, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return self._paginate(scored, top_k, cursor)

    def _recompute_avg_length(self) -> None:
        """Maintain the running average token length for stored chunks.

        Args:
            None

        Returns:
            None
        """
        if not self._chunks:
            self._avg_length = 0.0
            return
        total_length = sum(chunk.payload.token_count for chunk in self._chunks.values())
        self._avg_length = total_length / len(self._chunks)

    def stats(self) -> Mapping[str, float]:
        """Return summary statistics for stored documents.

        Args:
            None

        Returns:
            Mapping containing document count and average token length.
        """
        return {
            "document_count": float(len(self._chunks)),
            "avg_token_length": float(self._avg_length),
        }
