"""Operational tooling for FAISS/OpenSearch maintenance."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Mapping, Sequence

from .dense import FaissIndexManager
from .retrieval import HybridSearchRequest, HybridSearchService
from .storage import ChunkRegistry, OpenSearchSimulator


@dataclass(slots=True)
class PaginationCheckResult:
    """Result of a pagination verification run.

    Attributes:
        cursor_chain: Sequence of pagination cursors encountered.
        duplicate_detected: True when duplicate results were observed.

    Examples:
        >>> result = PaginationCheckResult(cursor_chain=[\"cursor1\"], duplicate_detected=False)
        >>> result.duplicate_detected
        False
    """

    cursor_chain: Sequence[str]
    duplicate_detected: bool


def build_stats_snapshot(
    faiss_index: FaissIndexManager,
    opensearch: OpenSearchSimulator,
    registry: ChunkRegistry,
) -> Mapping[str, object]:
    """Capture a lightweight snapshot of hybrid search storage metrics.

    Args:
        faiss_index: Dense vector index manager.
        opensearch: OpenSearch simulator representing lexical storage.
        registry: Chunk registry tracking vector-to-payload mappings.

    Returns:
        Mapping describing FAISS stats, OpenSearch stats, and chunk counts.
    """
    return {
        "faiss": faiss_index.stats(),
        "opensearch": opensearch.stats(),
        "registry": {"chunks": registry.count()},
    }


def verify_pagination(
    service: HybridSearchService, request: HybridSearchRequest
) -> PaginationCheckResult:
    """Ensure pagination cursors produce non-duplicated results.

    Args:
        service: Hybrid search service to execute paginated queries.
        request: Initial hybrid search request payload.

    Returns:
        PaginationCheckResult detailing encountered cursors and duplicates.
    """
    seen: set[tuple[str, str]] = set()
    cursor_chain: list[str] = []
    next_request = request
    duplicate = False
    seen_cursors: set[str] = set()
    while True:
        response = service.search(next_request)
        for result in response.results:
            key = (result.doc_id, result.chunk_id)
            if key in seen:
                duplicate = True
            seen.add(key)
        next_cursor = response.next_cursor
        if not next_cursor or next_cursor in seen_cursors:
            break
        cursor_chain.append(next_cursor)
        seen_cursors.add(next_cursor)
        next_request = HybridSearchRequest(
            query=request.query,
            namespace=request.namespace,
            filters=request.filters,
            page_size=request.page_size,
            cursor=next_cursor,
            diversification=request.diversification,
            diagnostics=request.diagnostics,
        )
    return PaginationCheckResult(cursor_chain=cursor_chain, duplicate_detected=duplicate)


def serialize_state(
    faiss_index: FaissIndexManager, registry: ChunkRegistry
) -> Mapping[str, object]:
    """Serialize FAISS state and registry metadata for snapshotting.

    Args:
        faiss_index: Dense index manager whose state should be captured.
        registry: Chunk registry providing vector identifiers.

    Returns:
        Mapping containing base64-encoded FAISS bytes and registered vector IDs.
    """
    faiss_bytes = faiss_index.serialize()
    encoded = base64.b64encode(faiss_bytes).decode("ascii")
    return {
        "faiss": encoded,
        "vector_ids": registry.vector_ids(),
    }


def restore_state(faiss_index: FaissIndexManager, payload: Mapping[str, object]) -> None:
    """Restore FAISS index state from a serialized payload.

    Args:
        faiss_index: Dense index manager to restore into.
        payload: Snapshot mapping produced by `serialize_state`.

    Raises:
        ValueError: If the payload does not include a FAISS snapshot.

    Returns:
        None
    """
    encoded = payload.get("faiss")
    if not isinstance(encoded, str):
        raise ValueError("Missing FAISS payload")
    faiss_index.restore(base64.b64decode(encoded.encode("ascii")))


def should_rebuild_index(
    registry: ChunkRegistry, deleted_since_snapshot: int, threshold: float = 0.2
) -> bool:
    """Heuristic to determine when FAISS should be rebuilt after deletions.

    Args:
        registry: Chunk registry reflecting current vector count.
        deleted_since_snapshot: Number of vectors deleted since the last snapshot.
        threshold: Fraction of deletions that triggers a rebuild.

    Returns:
        True when the proportion of deletions exceeds `threshold`.
    """
    total = registry.count() + deleted_since_snapshot
    if total == 0:
        return False
    return deleted_since_snapshot / total >= threshold
