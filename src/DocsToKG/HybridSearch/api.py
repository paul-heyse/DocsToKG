"""HTTP-style interface for the hybrid search service."""
from __future__ import annotations

from http import HTTPStatus
from typing import Any, Mapping, MutableMapping, Sequence

from .retrieval import HybridSearchRequest, HybridSearchService, RequestValidationError


class HybridSearchAPI:
    """Minimal synchronous handler for `POST /v1/hybrid-search`."""

    def __init__(self, service: HybridSearchService) -> None:
        self._service = service

    def post_hybrid_search(self, payload: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        try:
            request = self._parse_request(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return HTTPStatus.BAD_REQUEST, {"error": str(exc)}

        try:
            response = self._service.search(request)
        except RequestValidationError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive guard
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}

        body = {
            "results": [
                {
                    "doc_id": result.doc_id,
                    "chunk_id": result.chunk_id,
                    "namespace": result.namespace,
                    "score": result.score,
                    "fused_rank": result.fused_rank,
                    "text": result.text,
                    "highlights": list(result.highlights),
                    "provenance_offsets": [list(offset) for offset in result.provenance_offsets],
                    "metadata": dict(result.metadata),
                    "diagnostics": {
                        "bm25": result.diagnostics.bm25_score,
                        "splade": result.diagnostics.splade_score,
                        "dense": result.diagnostics.dense_score,
                    },
                }
                for result in response.results
            ],
            "next_cursor": response.next_cursor,
            "total_candidates": response.total_candidates,
            "timings_ms": dict(response.timings_ms),
        }
        return HTTPStatus.OK, body

    def _parse_request(self, payload: Mapping[str, Any]) -> HybridSearchRequest:
        query = str(payload["query"])
        namespace = payload.get("namespace")
        filters = self._normalize_filters(payload.get("filters", {}))
        page_size = int(payload.get("page_size", 10))
        cursor = payload.get("cursor")
        diversification = bool(payload.get("diversification", False))
        diagnostics = bool(payload.get("diagnostics", True))
        return HybridSearchRequest(
            query=query,
            namespace=str(namespace) if namespace is not None else None,
            filters=filters,
            page_size=page_size,
            cursor=str(cursor) if cursor is not None else None,
            diversification=diversification,
            diagnostics=diagnostics,
        )

    def _normalize_filters(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        normalized: MutableMapping[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                normalized[str(key)] = [str(item) for item in value]
            else:
                normalized[str(key)] = value
        return normalized

