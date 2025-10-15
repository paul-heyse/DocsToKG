"""
HTTP-style interface for the hybrid search service.

This module provides RESTful API endpoints for hybrid search operations,
exposing DocsToKG's search capabilities through HTTP interfaces for
integration with external systems and applications.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Mapping, MutableMapping, Sequence

from .retrieval import HybridSearchRequest, HybridSearchService, RequestValidationError


class HybridSearchAPI:
    """Minimal synchronous handler for `POST /v1/hybrid-search`.

    This class provides HTTP-style interface for hybrid search operations,
    translating REST API requests into service calls and formatting responses
    according to standard HTTP conventions.

    The API supports document search with hybrid retrieval combining BM25
    lexical search and dense vector similarity search, with configurable
    fusion strategies for optimal result ranking.

    Attributes:
        _service: The underlying hybrid search service instance

    Examples:
        >>> from docstokg.hybrid_search import HybridSearchService, HybridSearchAPI
        >>> service = HybridSearchService()
        >>> api = HybridSearchAPI(service)
        >>> status, response = api.post_hybrid_search({"query": "machine learning"})
        >>> print(f"Status: {status}, Results: {len(response['results'])}")
    """

    def __init__(self, service: HybridSearchService) -> None:
        """Initialize the hybrid search API with a service instance.

        Args:
            service: The hybrid search service to handle search operations

        Raises:
            TypeError: If service is not a HybridSearchService instance

        Returns:
            None
        """
        if not isinstance(service, HybridSearchService):
            raise TypeError("service must be a HybridSearchService instance")
        self._service = service

    def post_hybrid_search(self, payload: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        """Handle POST request for hybrid search operations.

        This method processes hybrid search requests by parsing the payload,
        executing the search through the service layer, and formatting the
        response according to HTTP conventions.

        Args:
            payload: Request payload containing search parameters including:
                - query: Search query string
                - limit: Maximum number of results (optional)
                - filters: Additional search filters (optional)
                - fusion_strategy: Result fusion method (optional)

        Returns:
            Tuple of (HTTP status code, response dictionary) containing:
            - 200: Successful search with results
            - 400: Bad request (invalid parameters or validation error)
            - 500: Internal server error

        Examples:
            >>> api = HybridSearchAPI(service)
            >>> status, response = api.post_hybrid_search({
            ...     "query": "machine learning",
            ...     "limit": 10
            ... })
            >>> print(f"Found {len(response['results'])} results")
        """
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
        """Parse HTTP request payload into HybridSearchRequest object.

        This method validates and transforms the incoming HTTP request payload
        into a structured HybridSearchRequest object for the search service.

        Args:
            payload: Raw request payload from HTTP endpoint containing:
                - query: Required search query string
                - namespace: Optional namespace for scoped search
                - filters: Optional search filters as key-value pairs
                - page_size: Optional result pagination size
                - cursor: Optional pagination cursor for continuation
                - diversification: Optional MMR diversification flag
                - diagnostics: Optional diagnostics inclusion flag

        Returns:
            Validated HybridSearchRequest object

        Raises:
            KeyError: If required 'query' field is missing
            TypeError: If payload fields have incorrect types
            ValueError: If payload values are invalid
        """
        query = str(payload["query"])
        namespace = payload.get("namespace")
        filters = self._normalize_filters(payload.get("filters", {}))
        page_size = int(payload.get("page_size", payload.get("limit", 10)))
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
        """Normalize filter values for consistent processing.

        This method ensures all filter values are properly typed and formatted
        for the search service, converting sequences to string lists and
        maintaining type consistency.

        Args:
            payload: Raw filter payload from request containing:
                - Keys: Filter field names
                - Values: Filter values (may be single values or sequences)

        Returns:
            Normalized filters with consistent typing:
            - String keys for all filter names
            - String lists for sequence values
            - Original values for single-value filters

        Examples:
            >>> api = HybridSearchAPI(service)
            >>> filters = api._normalize_filters({
            ...     "tags": ["ml", "ai"],
            ...     "author": "john_doe"
            ... })
            >>> print(filters)
            {'tags': ['ml', 'ai'], 'author': 'john_doe'}
        """
        normalized: MutableMapping[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                # Convert sequences to string lists for consistent processing
                normalized[str(key)] = [str(item) for item in value]
            else:
                # Keep single values as-is
                normalized[str(key)] = value
        return normalized
