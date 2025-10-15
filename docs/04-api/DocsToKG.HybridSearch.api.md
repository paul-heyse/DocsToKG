# 1. Module: api

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.service``.

HTTP-style interface for the hybrid search service.

This module provides RESTful API endpoints for hybrid search operations,
exposing DocsToKG's search capabilities through HTTP interfaces for
integration with external systems and applications.

## 1. Functions

### `post_hybrid_search(self, payload)`

Handle POST request for hybrid search operations.

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

### `_parse_request(self, payload)`

Parse HTTP request payload into HybridSearchRequest object.

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

### `_normalize_filters(self, payload)`

Normalize filter values for consistent processing.

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

## 2. Classes

### `HybridSearchAPI`

Minimal synchronous handler for `POST /v1/hybrid-search`.

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
