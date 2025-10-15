# Module: retrieval

Hybrid search execution across sparse and dense channels.

This module provides the core hybrid search service for DocsToKG, orchestrating
multiple retrieval methods (BM25, SPLADE, dense vectors) and fusing their results
for optimal document retrieval performance.

The service supports configurable search strategies, real-time observability,
and comprehensive result ranking through advanced fusion techniques.

## Functions

### `search(self, request)`

Execute a hybrid search request across all retrieval channels.

Args:
request: Validated hybrid search request describing query parameters.

Returns:
HybridSearchResponse containing fused results and cursor metadata.

Raises:
RequestValidationError: If the request fails validation checks.

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

### `_validate_request(self, request)`

*No documentation available.*

## Classes

### `RequestValidationError`

Raised when the caller submits an invalid search request.

This exception is raised when a hybrid search request contains invalid
parameters, malformed data, or violates system constraints.

Attributes:
message: Description of the validation error
field: Optional field name that caused the error

Examples:
>>> try:
...     service.search(invalid_request)
... except RequestValidationError as e:
...     print(f"Invalid request: {e.message}")

### `ChannelResults`

Results from a single retrieval channel (BM25, SPLADE, or dense).

This class encapsulates the candidates and scoring information returned
by a specific retrieval method, preparing them for fusion with results
from other channels.

Attributes:
candidates: List of fusion candidates from this channel
scores: Performance metrics for this channel's execution

Examples:
>>> bm25_results = ChannelResults(
...     candidates=[candidate1, candidate2],
...     scores={"recall": 0.85, "latency": 45}
... )

### `HybridSearchService`

Execute BM25, SPLADE, and dense retrieval with fusion.

This service orchestrates hybrid search operations by:
1. Executing parallel retrieval across multiple channels
2. Fusing results using configurable strategies
3. Applying diversification and ranking optimizations
4. Providing comprehensive observability and metrics

The service is designed for high-performance document retrieval
combining traditional lexical search with modern semantic methods.

Attributes:
_config_manager: Configuration management for search parameters
_feature_generator: Feature extraction for query processing
_faiss: Dense vector search using FAISS
_opensearch: Lexical search using OpenSearch
_registry: Document and chunk registry management
_observability: Performance monitoring and metrics collection

Examples:
>>> service = HybridSearchService(
...     config_manager=config_manager,
...     feature_generator=feature_generator,
...     faiss_index=faiss_index,
...     opensearch=opensearch,
...     registry=registry
... )
>>> results = service.search(request)
