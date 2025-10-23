# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.HybridSearch.interfaces",
#   "purpose": "Formal contracts for plugging alternative sparse and dense backends into the.",
#   "sections": [
#     {
#       "id": "lexicalindex",
#       "name": "LexicalIndex",
#       "anchor": "class-lexicalindex",
#       "kind": "class"
#     },
#     {
#       "id": "densevectorstore",
#       "name": "DenseVectorStore",
#       "anchor": "class-densevectorstore",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Formal contracts for plugging alternative sparse and dense backends into the
DocsToKG hybrid search pipeline.

The ingestion and query flows interact with dependencies exclusively through
``LexicalIndex`` and ``DenseVectorStore``. This module documents what those
dependencies *must* guarantee so agents can swap implementations while keeping
fusion, observability, and pagination semantics intact:

- ``LexicalIndex`` represents BM25/SPLADE-capable sparse stores (the production
  integration is OpenSearch; the dev harness uses
  ``devtools.OpenSearchSimulator``). Implementations are responsible for bulk
  upserts/deletes, cursor-based pagination, and deterministic scoring so that
  ``service.HybridSearchService`` can safely blend results across channels.
  Optional ``search_bm25_true`` support exposes exact Okapi BM25 (matching the
  formula in ``devtools.opensearch_simulator``) when dense recall tuning calls
  for DF-aware scoring.
- ``DenseVectorStore`` models the FAISS GPU adapter surface. ``ManagedFaissAdapter``
  is the canonical implementation; it uses ``faiss.StandardGpuResources``,
  ``index_cpu_to_gpu(_multiple)``, and ``knn_gpu`` / ``pairwise_distance_gpu`` to
  serve cosine or inner-product queries. Method signatures here assume callers
  pass contiguous ``float32`` arrays (optionally batched) and expect per-query
  Top-K ``FaissSearchResult`` objects that include scores, vector identifiers,
  and metadata from the ``ChunkRegistry``.

Read these protocols before introducing alternative backends: every method,
return type, and docstring maps directly onto behaviour that ``pipeline``,
``router``, and ``service`` rely on for concurrency control, snapshotting, and
fusion math.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

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
        cursor: int | None = None,
    ) -> tuple[list[tuple[ChunkPayload, float]], int | None]:
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
        cursor: int | None = None,
    ) -> tuple[list[tuple[ChunkPayload, float]], int | None]:
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
        cursor: int | None = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> tuple[list[tuple[ChunkPayload, float]], int | None]:
        """Optional Okapi BM25 search supporting DF/length-aware scoring."""

    def highlight(self, chunk: ChunkPayload, query_tokens: Sequence[str]) -> list[str]:
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
    def adapter_stats(self) -> AdapterStats:
        """Return runtime adapter statistics (device, nprobe, replication state)."""

    def add(self, vectors: Sequence[np.ndarray], vector_ids: Sequence[str]) -> None:
        """Insert dense vectors."""

    def remove(self, vector_ids: Sequence[str]) -> None:
        """Delete dense vectors referenced by ``vector_ids``."""

    def search(self, query: np.ndarray, top_k: int) -> Sequence[FaissSearchResult]:
        """Search for nearest neighbours of ``query``."""

    def search_many(self, queries: np.ndarray, top_k: int) -> Sequence[Sequence[FaissSearchResult]]:
        """Search for nearest neighbours of multiple queries."""

    def search_batch(
        self, queries: np.ndarray, top_k: int
    ) -> Sequence[Sequence[FaissSearchResult]]:
        """Optional alias for batched search."""

    def range_search(
        self,
        query: np.ndarray,
        min_score: float,
        *,
        limit: int | None = None,
    ) -> Sequence[FaissSearchResult]:
        """Return all vectors scoring above ``min_score`` for ``query``."""

    def reconstruct_batch(self, vector_ids: Sequence[str]) -> np.ndarray:
        """Reconstruct embeddings for ``vector_ids`` from the dense store."""

    def reconstruct_vector(self, vector_id: str) -> np.ndarray:
        """Reconstruct a single embedding referenced by ``vector_id``."""

    def serialize(self) -> bytes:
        """Return a serialised representation of the index."""

    def restore(self, payload: bytes, *, meta: Mapping[str, object] | None = None) -> None:
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

    def set_id_resolver(self, resolver: Callable[[int], str | None]) -> None:
        """Register a resolver translating FAISS integer IDs to external IDs."""
