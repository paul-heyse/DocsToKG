"""Operational tooling for FAISS/OpenSearch maintenance."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from .dense import FaissIndexManager
from .retrieval import HybridSearchRequest, HybridSearchService
from .storage import ChunkRegistry, OpenSearchSimulator


@dataclass(slots=True)
class PaginationCheckResult:
    cursor_chain: Sequence[str]
    duplicate_detected: bool


def build_stats_snapshot(
    faiss_index: FaissIndexManager,
    opensearch: OpenSearchSimulator,
    registry: ChunkRegistry,
) -> Mapping[str, object]:
    return {
        "faiss": faiss_index.stats(),
        "opensearch": opensearch.stats(),
        "registry": {"chunks": registry.count()},
    }


def verify_pagination(service: HybridSearchService, request: HybridSearchRequest) -> PaginationCheckResult:
    seen: set[tuple[str, str]] = set()
    cursor_chain: list[str] = []
    next_request = request
    duplicate = False
    while True:
        response = service.search(next_request)
        for result in response.results:
            key = (result.doc_id, result.chunk_id)
            if key in seen:
                duplicate = True
            seen.add(key)
        if not response.next_cursor:
            break
        cursor_chain.append(response.next_cursor)
        next_request = HybridSearchRequest(
            query=request.query,
            namespace=request.namespace,
            filters=request.filters,
            page_size=request.page_size,
            cursor=response.next_cursor,
            diversification=request.diversification,
            diagnostics=request.diagnostics,
        )
    return PaginationCheckResult(cursor_chain=cursor_chain, duplicate_detected=duplicate)


def serialize_state(faiss_index: FaissIndexManager, registry: ChunkRegistry) -> Mapping[str, object]:
    faiss_bytes = faiss_index.serialize()
    encoded = base64.b64encode(faiss_bytes).decode("ascii")
    return {
        "faiss": encoded,
        "vector_ids": registry.vector_ids(),
    }


def restore_state(faiss_index: FaissIndexManager, payload: Mapping[str, object]) -> None:
    encoded = payload.get("faiss")
    if not isinstance(encoded, str):
        raise ValueError("Missing FAISS payload")
    faiss_index.restore(base64.b64decode(encoded.encode("ascii")))


def should_rebuild_index(registry: ChunkRegistry, deleted_since_snapshot: int, threshold: float = 0.2) -> bool:
    total = registry.count() + deleted_since_snapshot
    if total == 0:
        return False
    return deleted_since_snapshot / total >= threshold

