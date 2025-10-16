"""Chunk registry helpers and shared filter utilities.

Args:
    None

Returns:
    None

Raises:
    None
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Mapping, Optional, Sequence

from .types import ChunkPayload, vector_uuid_to_faiss_int

from .devtools.opensearch_simulator import OpenSearchSimulator

__all__ = ("ChunkRegistry", "OpenSearchSimulator", "matches_filters")


class ChunkRegistry:
    """Durable mapping of vector identifiers to chunk payloads.

    Attributes:
        _chunks: Mapping from vector identifier to chunk payload.
        _bridge: Mapping from FAISS integer identifier to vector identifier.

    Examples:
        >>> registry = ChunkRegistry()
        >>> registry.upsert([])  # no-op
        >>> registry.count()
        0
    """

    def __init__(self) -> None:
        self._chunks: Dict[str, ChunkPayload] = {}
        self._bridge: Dict[int, str] = {}

    def upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        """Insert or update registry entries for ``chunks``.

        Args:
            chunks: Chunk payloads that should be tracked by the registry.

        Returns:
            None
        """
        for chunk in chunks:
            self._chunks[chunk.vector_id] = chunk
            self._bridge[vector_uuid_to_faiss_int(chunk.vector_id)] = chunk.vector_id

    def delete(self, vector_ids: Sequence[str]) -> None:
        """Remove registry entries for the supplied vector identifiers.

        Args:
            vector_ids: Identifiers associated with chunks to delete.

        Returns:
            None

        Raises:
            None
        """
        for vector_id in vector_ids:
            self._chunks.pop(vector_id, None)
            self._bridge.pop(vector_uuid_to_faiss_int(vector_id), None)

    def get(self, vector_id: str) -> Optional[ChunkPayload]:
        """Return the chunk payload for ``vector_id`` when available.

        Args:
            vector_id: Identifier of the chunk to retrieve.

        Returns:
            Matching chunk payload, or ``None`` if the identifier is unknown.
        """
        return self._chunks.get(vector_id)

    def bulk_get(self, vector_ids: Sequence[str]) -> List[ChunkPayload]:
        """Return chunk payloads for identifiers present in the registry.

        Args:
            vector_ids: Identifiers of the desired chunk payloads.

        Returns:
            List of chunk payloads that are present in the registry.
        """
        return [self._chunks[vid] for vid in vector_ids if vid in self._chunks]

    def resolve_faiss_id(self, internal_id: int) -> Optional[str]:
        """Translate a FAISS integer id back to the original vector identifier.

        Args:
            internal_id: Integer id assigned by the FAISS vector store.

        Returns:
            Associated vector identifier when the mapping exists.
        """
        return self._bridge.get(internal_id)

    def all(self) -> List[ChunkPayload]:
        """Return all cached chunk payloads.

        Args:
            None

        Returns:
            List containing every chunk payload stored in the registry.
        """
        return list(self._chunks.values())

    def iter_all(self) -> typing.Iterator[ChunkPayload]:
        """Yield chunk payloads without materialising the full list."""

        return iter(self._chunks.values())

    def count(self) -> int:
        """Return the number of chunks tracked by the registry.

        Args:
            None

        Returns:
            Number of chunk payloads stored in the registry.
        """
        return len(self._chunks)

    def vector_ids(self) -> List[str]:
        """Return all vector identifiers in insertion order.

        Args:
            None

        Returns:
            Vector identifiers ordered by insertion time.
        """
        return list(self._chunks.keys())


def matches_filters(chunk: ChunkPayload, filters: Mapping[str, object]) -> bool:
    """Check whether ``chunk`` satisfies the provided OpenSearch-style filters.

    Args:
        chunk: Chunk payload whose metadata should be evaluated.
        filters: Mapping of filter keys to expected values.

    Returns:
        ``True`` if the chunk matches every filter, otherwise ``False``.
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
