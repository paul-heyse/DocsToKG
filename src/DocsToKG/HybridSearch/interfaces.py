"""Protocols defining hybrid-search integration points.

Args:
    None

Returns:
    None

Raises:
    None
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .config import DenseIndexConfig
from .types import ChunkPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import AdapterStats, FaissSearchResult


class LexicalIndex(Protocol):
    """Protocol describing the lexical (BM25/SPLADE) index interface.

    Implementations are expected to manage chunk persistence, sparse search, and
    highlighting in a way that is compatible with the hybrid search pipeline.

    Attributes:
        None

    Examples:
        >>> from DocsToKG.HybridSearch.store import OpenSearchSimulator
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

        Raises:
            Exception: Implementations may surface provider-specific failures.
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

    def search_bm25_true(
        self,
        query_weights: Mapping[str, float],
        filters: Mapping[str, object],
        top_k: int,
        cursor: Optional[int] = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> Tuple[List[Tuple[ChunkPayload, float]], Optional[int]]:
        """Optional Okapi BM25 search supporting DF/length-aware scoring."""

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


class DenseVectorStore(Protocol):
    """Protocol describing the dense vector index surface area.

    Implementations provide GPU-backed vector search with facilities for
    ingestion, persistence, and statistics reporting. This protocol allows
    tests to swap lightweight stand-ins without relying on FAISS directly.
    """

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality."""

    @property
    def ntotal(self) -> int:
        """Return the number of stored vectors."""

    @property
    def device(self) -> int:
        """Return the CUDA device identifier used by the index."""

    @property
    def config(self) -> DenseIndexConfig:
        """Immutable configuration backing the dense store."""

    @property
    def adapter_stats(self) -> "AdapterStats":
        """Return runtime adapter statistics (device, nprobe, replication state)."""

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        """Insert dense vectors."""

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Delete dense vectors referenced by ``vector_ids``."""

    def search(self, query: np.ndarray, top_k: int) -> Sequence["FaissSearchResult"]:
        """Search for nearest neighbours of ``query``."""

    def search_many(
        self, queries: np.ndarray, top_k: int
    ) -> Sequence[Sequence["FaissSearchResult"]]:
        """Search for nearest neighbours of multiple queries."""

    def search_batch(
        self, queries: np.ndarray, top_k: int
    ) -> Sequence[Sequence["FaissSearchResult"]]:
        """Optional alias for batched search."""

    def range_search(
        self,
        query: np.ndarray,
        min_score: float,
        *,
        limit: Optional[int] = None,
    ) -> Sequence["FaissSearchResult"]:
        """Return all vectors scoring above ``min_score`` for ``query``."""

    def serialize(self) -> bytes:
        """Return a serialised representation of the index."""

    def restore(
        self,
        payload: bytes,
        *,
        meta: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Restore index state from ``payload``."""

    def flush_snapshot(self, *, reason: str = "flush") -> None:
        """Force a snapshot refresh bypassing throttle safeguards."""

    def stats(self) -> Mapping[str, float | str]:
        """Return implementation-defined statistics."""

    def rebuild_if_needed(self) -> bool:
        """Perform compaction when the store indicates a rebuild is required."""

    def needs_training(self) -> bool:
        """Return ``True`` when additional training is required."""

    def train(self, vectors: Sequence[np.ndarray]) -> None:
        """Train the index with representative vectors."""

    def set_id_resolver(self, resolver: Callable[[int], Optional[str]]) -> None:
        """Register a resolver translating FAISS integer IDs to external IDs."""
