# 1. Module: service

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.service``.

Hybrid search service and synchronous HTTP-style API.

## 1. Functions

### `search(self, request)`

Execute a hybrid retrieval round trip for ``request``.

Args:
request: Fully validated hybrid search request describing the query,
namespace, filters, and pagination parameters.

Returns:
HybridSearchResponse: Ranked hybrid search results enriched with channel-level
diagnostics and pagination cursor metadata.

Raises:
RequestValidationError: If ``request`` fails validation checks.

### `_validate_request(self, request)`

*No documentation available.*

### `_build_cursor(self, results, page_size)`

*No documentation available.*

### `_execute_bm25(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_execute_splade(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_execute_dense(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_filter_dense_hits(self, hits, filters)`

*No documentation available.*

### `_dedupe_candidates(self, candidates, fused_scores)`

*No documentation available.*

### `post_hybrid_search(self, payload)`

Process a synchronous hybrid search HTTP-style request payload.

Args:
payload: JSON-like mapping containing the hybrid search request body.

Returns:
tuple[int, Mapping[str, Any]]: HTTP status code and serialized response body.

### `_parse_request(self, payload)`

*No documentation available.*

### `_normalize_filters(self, payload)`

*No documentation available.*

## 2. Classes

### `RequestValidationError`

Raised when the caller submits an invalid search request.

Attributes:
None

Examples:
>>> raise RequestValidationError("page_size must be positive")
Traceback (most recent call last):
...
RequestValidationError: page_size must be positive

### `ChannelResults`

Results from a single retrieval channel (BM25, SPLADE, or dense).

Attributes:
candidates: Ordered list of channel-specific fusion candidates.
scores: Mapping of vector identifiers to raw channel scores.

Examples:
>>> ChannelResults(candidates=[], scores={})
ChannelResults(candidates=[], scores={})

### `HybridSearchService`

Execute BM25, SPLADE, and dense retrieval with fusion.

Attributes:
_config_manager: Source of runtime hybrid-search configuration.
_feature_generator: Component producing BM25/SPLADE/dense features.
_faiss: GPU-backed FAISS index manager for dense retrieval.
_opensearch: Simulator used for BM25 and SPLADE lookups.
_registry: Chunk registry providing metadata and FAISS id lookups.
_observability: Telemetry facade for metrics and traces.

Examples:
>>> service = HybridSearchService(  # doctest: +SKIP
...     config_manager=HybridSearchConfigManager.from_dict({}),
...     feature_generator=FeatureGenerator(embedding_dim=16),
...     faiss_index=FaissIndexManager(dim=16, config=HybridSearchConfig().dense),
...     opensearch=OpenSearchSimulator(),
...     registry=ChunkRegistry(),
... )
>>> request = HybridSearchRequest(query="example", namespace="demo", filters={}, page_size=5)
>>> isinstance(service.search(request), HybridSearchResponse)  # doctest: +SKIP
True

### `HybridSearchAPI`

Minimal synchronous handler for ``POST /v1/hybrid-search``.

Attributes:
_service: Underlying :class:`HybridSearchService` instance.

Examples:
>>> api = HybridSearchAPI(service)  # doctest: +SKIP
>>> status, body = api.post_hybrid_search({"query": "example"})  # doctest: +SKIP
>>> status
200
