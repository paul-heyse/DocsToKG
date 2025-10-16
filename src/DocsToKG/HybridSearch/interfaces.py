"""Protocols defining hybrid-search integration points.

Args:
    None

Returns:
    None

Raises:
    None
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Protocol, Sequence, Tuple

from .types import ChunkPayload


class LexicalIndex(Protocol):
    """Protocol describing the lexical (BM25/SPLADE) index interface.

    Implementations are expected to manage chunk persistence, sparse search, and
    highlighting in a way that is compatible with the hybrid search pipeline.

    Attributes:
        None

    Examples:
        >>> from DocsToKG.HybridSearch.devtools.opensearch_simulator import OpenSearchSimulator
        >>> simulator: LexicalIndex = OpenSearchSimulator()
        >>> simulator.bulk_upsert([])  # doctest: +SKIP
    """

    def bulk_upsert(self, chunks: Sequence[ChunkPayload]) -> None:
        """Insert or update chunk payloads.

        Args:
            chunks: Chunk payloads that should be persisted in the lexical index.

        Returns:
            None
        """

    def bulk_delete(self, vector_ids: Sequence[str]) -> None:
        """Remove chunk payloads for the supplied vector identifiers.

        Args:
            vector_ids: Identifiers referencing payloads that should be removed.

        Returns:
            None
        """

    def search_bm25(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Execute a BM25-style sparse search.

        Args:
            query_weights: Sparse query representation (token → weight).
            filters: Metadata filters used to restrict the search domain.
            top_k: Maximum number of hits the caller expects in the page.
            cursor: Optional continuation token from a previous search page.

        Returns:
            List of ``(chunk, score)`` pairs and an optional cursor for pagination.
        """

    def search_splade(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Execute a SPLADE sparse search.

        Args:
            query_weights: Sparse SPLADE activations for the query.
            filters: Metadata filters used to restrict the search domain.
            top_k: Maximum number of hits the caller expects in the page.
            cursor: Optional continuation token from a previous search page.

        Returns:
            List of ``(chunk, score)`` pairs and an optional cursor for pagination.
        """

    def highlight(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> List[str]:
        """Return highlight snippets for ``chunk`` given ``query_tokens``.

        Args:
            chunk: Chunk payload whose text should be highlighted.
            query_tokens: Tokens extracted from the query text.

        Returns:
            Highlights appropriate for presentation alongside the chunk.
        """

    def stats(self) -> Mapping[str, float] | Mapping[str, object]:
        """Return implementation-defined statistics about the lexical index.

        Args:
            None

        Returns:
            Mapping containing statistics relevant to the implementation.
        """
