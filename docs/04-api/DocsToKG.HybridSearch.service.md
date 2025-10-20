# 1. Module: service

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.service``.

## 1. Overview

Hybrid search orchestration, pagination guards, and synchronous HTTP-style API.

## 2. Functions

### `apply_mmr_diversification(fused_candidates, fused_scores, lambda_param, top_k)`

Diversify fused candidates using Maximum Marginal Relevance.

Args:
fused_candidates: Ranked candidates in fused order.
fused_scores: Combined scores keyed by vector identifier.
lambda_param: Trade-off between relevance and diversity (0.0-1.0).
top_k: Maximum number of diversified candidates to retain.
embeddings: Optional dense embedding matrix aligned with ``fused_candidates``.
device: GPU device id when leveraging FAISS GPU routines.
resources: Optional FAISS GPU resources handle reused across calls.
block_rows: Corpus rows processed per block when estimating diversity on GPU.
use_cuvs: Optional override that controls cuVS usage during GPU similarity
lookups. ``None`` defers to runtime detection; ``True`` forces cuVS when
supported and ``False`` disables it.

Returns:
List of diversified `FusionCandidate` objects.

### `build_stats_snapshot(faiss_index, opensearch, registry)`

Capture a lightweight snapshot of hybrid search storage metrics.

Args:
faiss_index: Dense vector index manager.
opensearch: Lexical index representing sparse storage.
registry: Chunk registry tracking vector-to-payload mappings.

Returns:
Mapping describing FAISS stats, OpenSearch stats, and chunk counts.

### `verify_pagination(service, request)`

Ensure pagination cursors produce non-duplicated results.

Args:
service: Hybrid search service to execute paginated queries.
request: Initial hybrid search request payload.

Returns:
PaginationCheckResult detailing encountered cursors and duplicates.

### `should_rebuild_index(registry, deleted_since_snapshot, threshold)`

Heuristic to determine when FAISS should be rebuilt after deletions.

Args:
registry: Chunk registry reflecting current vector count.
deleted_since_snapshot: Number of vectors deleted since the last snapshot.
threshold: Fraction of deletions that triggers a rebuild.

Returns:
True when the proportion of deletions exceeds ``threshold``.

### `load_dataset(path)`

Load a JSONL dataset describing documents and queries.

Args:
path: Path to a JSONL file containing dataset entries.

Returns:
List of parsed dataset rows suitable for validation routines.

Raises:
FileNotFoundError: If the dataset file does not exist.
json.JSONDecodeError: If any line contains invalid JSON.

### `infer_embedding_dim(dataset)`

Infer dense embedding dimensionality from dataset vector artifacts.

Args:
dataset: Sequence of dataset entries containing vector metadata.

Returns:
Inferred embedding dimensionality, defaulting to 2560 when unknown.

### `plan(self, signature)`

Return the requested ``k`` and oversampling knobs for a dense search.

When a cached depth exists for ``signature`` it is treated as a lower bound
only. The planner still honours runtime knobs such as
``RetrievalConfig.dense_top_k`` by returning ``max(computed_k, cached_k)``.

### `observe_pass_rate(self, signature, observed)`

Blend ``observed`` into the running EMA and return the updated value.

### `remember(self, signature, k)`

Cache ``k`` for ``signature`` to seed future requests.

### `has_cache(self, signature)`

Return ``True`` when ``signature`` has a cached ``k`` value.

### `current_pass_rate(self)`

Expose the current blended pass rate.

### `persist(self)`

Persist the adaptive cache to disk when a cache path is configured.

### `_encode_signature(self, value)`

*No documentation available.*

### `_decode_signature(self, value)`

*No documentation available.*

### `_persist_cache(self)`

*No documentation available.*

### `_load_cache(self)`

*No documentation available.*

### `fuse(self, candidates)`

Combine channel rankings into fused scores keyed by vector id.

Args:
candidates: Ranked fusion candidates from individual retrieval channels.

Returns:
Mapping from vector identifiers to fused RRF scores.

### `shape(self, ordered_chunks, fused_scores, request, channel_scores)`

Shape ranked chunks into API responses with highlights and diagnostics.

Args:
ordered_chunks: Chunks ordered by fused score.
fused_scores: Combined score per vector identifier.
request: Incoming hybrid search request.
channel_scores: Per-channel score maps for diagnostics emission.
precomputed_embeddings: Optional dense embeddings aligned with chunks.

Returns:
List of `HybridSearchResult` instances ready for serialization.

### `_within_doc_limit(self, doc_id, doc_buckets)`

*No documentation available.*

### `_is_near_duplicate(self, embeddings, current_idx, emitted_indices)`

*No documentation available.*

### `_build_highlights(self, chunk, query_tokens)`

*No documentation available.*

### `_cosine_cpu(query, pool)`

*No documentation available.*

### `close(self)`

Release pooled resources held by the service.

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

### `_assert_managed_store(store)`

*No documentation available.*

### `_flush_dense_snapshots(self)`

*No documentation available.*

### `_dense_store(self, namespace)`

*No documentation available.*

### `_slice_from_cursor(self, results, cursor, page_size, fingerprint)`

*No documentation available.*

### `_build_cursor(self, results, page_size, fingerprint)`

*No documentation available.*

### `_slice_after_anchor(self, results, vector_id, score, rank)`

*No documentation available.*

### `run_compaction_cycle(self)`

Trigger FAISS maintenance (rebuild/compact) and emit diagnostics.

### `_execute_bm25(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_execute_splade(self, request, filters, config, query_features, timings)`

*No documentation available.*

### `_execute_dense(self, request, filters, config, query_features, timings, store)`

*No documentation available.*

### `_filter_dense_hits(self, hits, filters, score_floor)`

*No documentation available.*

### `_dense_request_signature(self, request, filters)`

*No documentation available.*

### `_normalize_signature_value(self, value)`

*No documentation available.*

### `_cursor_fingerprint(self, request, filters)`

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

### `run(self, dataset, output_root)`

Execute standard validation against a dataset.

Args:
dataset: Loaded dataset entries containing document and query payloads.
output_root: Optional directory where reports should be written.

Returns:
ValidationSummary capturing per-check reports.

### `run_scale(self, dataset)`

Execute a comprehensive scale validation suite.

Args:
dataset: Dataset entries containing documents and queries.
output_root: Optional directory for detailed metrics output.
thresholds: Optional overrides for scale validation thresholds.
query_sample_size: Number of queries sampled for specific checks.

Returns:
ValidationSummary with detailed per-check reports.

### `_to_document(self, payload)`

Transform dataset document payload into a `DocumentInput`.

Args:
payload: Dataset document dictionary containing paths and metadata.

Returns:
Corresponding `DocumentInput` instance for ingestion.

### `_check_ingest_integrity(self)`

Verify that ingested chunks contain valid embeddings.

Args:
None

Returns:
ValidationReport describing integrity checks for ingested chunks.

### `_check_dense_self_hit(self)`

Ensure dense search returns each chunk as its own nearest neighbor.

Args:
None

Returns:
ValidationReport summarizing dense self-hit accuracy.

### `_check_sparse_relevance(self, dataset)`

Check that sparse channels retrieve expected documents.

Args:
dataset: Dataset entries with expected relevance metadata.

Returns:
ValidationReport covering sparse channel relevance.

### `_check_namespace_filters(self, dataset)`

Ensure namespace-constrained queries do not leak results.

Args:
dataset: Dataset entries containing namespace-constrained queries.

Returns:
ValidationReport indicating whether namespace isolation holds.

### `_check_pagination(self, dataset)`

Verify pagination cursors do not create duplicate results.

Args:
dataset: Dataset entries providing queries for pagination tests.

Returns:
ValidationReport detailing pagination stability.

### `_check_highlights(self, dataset)`

Assert that each result includes highlight snippets.

Args:
dataset: Dataset entries with queries to validate highlight generation.

Returns:
ValidationReport indicating highlight completeness.

### `_check_backup_restore(self, dataset)`

Validate backup/restore by round-tripping the FAISS index.

Args:
dataset: Dataset entries used to verify consistency post-restore.

Returns:
ValidationReport capturing backup/restore status.

### `_request_for_query(self, query, page_size)`

Construct a `HybridSearchRequest` for a dataset query payload.

Args:
query: Query payload from the dataset.
page_size: Desired page size for generated requests.

Returns:
HybridSearchRequest ready for execution against the service.

### `_persist_reports(self, summary, output_root, calibration_details)`

Persist validation summaries and detailed metrics to disk.

Args:
summary: Validation summary containing reports.
output_root: Optional directory for storing artifacts.
calibration_details: Optional calibration metrics to persist.
extras: Optional mapping of additional metrics categories.

Returns:
None

### `_collect_queries(self, dataset)`

Collect (document, query) pairs from the dataset.

Args:
dataset: Loaded dataset entries containing documents and queries.

Returns:
List of tuples pairing document payloads with query payloads.

### `_sample_queries(self, dataset, sample_size, rng)`

Sample a subset of (document, query) pairs for randomized checks.

Args:
dataset: Dataset entries containing documents and queries.
sample_size: Maximum number of pairs to return.
rng: Random generator used for sampling.

Returns:
List of sampled (document, query) pairs.

### `_scale_data_sanity(self, documents, dataset)`

Validate that ingested corpus statistics look healthy.

Args:
documents: Document inputs ingested during the scale run.
dataset: Dataset entries used for validation.

Returns:
ValidationReport summarizing data sanity findings.

### `_scale_crud_namespace(self, documents, dataset, inputs_by_doc, rng)`

Exercise CRUD operations to ensure namespace isolation stays intact.

Args:
documents: Documents ingested for the validation run.
dataset: Dataset entries providing queries for verification.
inputs_by_doc: Mapping back to original document inputs.
rng: Random generator used to select documents.

Returns:
ValidationReport describing CRUD and namespace violations if any.

### `_scale_dense_metrics(self, thresholds, rng)`

Assess dense retrieval self-hit, perturbation, and recall metrics.

Args:
thresholds: Threshold values for dense metric success.
rng: Random generator for selecting sample chunks.

Returns:
ValidationReport containing dense metric measurements.

### `_scale_channel_relevance(self, dataset, thresholds, rng, query_sample_size)`

Measure top-10 relevance for each retrieval channel across sampled queries.

Args:
dataset: Dataset entries containing queries for evaluation.
thresholds: Threshold mapping for hit-rate expectations.
rng: Random generator used for sampling query pairs.
query_sample_size: Number of query pairs sampled.

Returns:
ValidationReport capturing per-channel relevance metrics.

### `_scale_fusion_mmr(self, dataset, thresholds, rng, query_sample_size)`

Evaluate diversification impact of MMR versus RRF on sampled queries.

Args:
dataset: Dataset entries containing documents and queries.
thresholds: Diversification thresholds for redundancy and hit-rate deltas.
rng: Random generator for sampling queries.
query_sample_size: Number of query pairs to evaluate.

Returns:
ValidationReport detailing redundancy reduction and hit-rate deltas.

### `_scale_pagination(self, dataset, rng, sample_size)`

Ensure page cursors generate disjoint result sets across pages.

Args:
dataset: Dataset entries containing documents and queries.
rng: Random generator for sampling query pairs.
sample_size: Number of queries to evaluate for pagination.

Returns:
ValidationReport indicating pagination overlap issues.

### `_scale_result_shaping(self, dataset, rng, sample_size)`

Confirm per-document limits, deduping, and highlights behave as expected.

Args:
dataset: Dataset entries containing documents and queries.
rng: Random generator used for sampling query pairs.
sample_size: Number of sampled queries to evaluate.

Returns:
ValidationReport summarizing result shaping concerns.

### `_scale_backup_restore(self, dataset, rng, query_sample_size)`

Validate snapshot/restore under randomized workloads.

Args:
dataset: Dataset entries with documents and queries.
rng: Random generator for sampling queries.
query_sample_size: Maximum number of query pairs to evaluate.

Returns:
ValidationReport highlighting snapshot mismatches, if any.

### `_scale_acl(self, dataset)`

Ensure per-namespace ACL tags are enforced across queries.

Args:
dataset: Dataset entries providing documents and queries.

Returns:
ValidationReport listing ACL violations, if discovered.

### `_scale_performance(self, dataset, thresholds, rng, query_sample_size)`

Benchmark latency and throughput against configured thresholds.

Args:
dataset: Dataset entries containing documents and queries.
thresholds: Threshold values for latency and headroom metrics.
rng: Random generator used for sampling query pairs.
query_sample_size: Number of queries to sample for benchmarking.

Returns:
ValidationReport detailing performance metrics.

### `_run_calibration(self, dataset)`

Record calibration metrics (self-hit accuracy) across oversampling factors.

Args:
dataset: Dataset entries (unused, provided for symmetry).

Returns:
ValidationReport summarizing calibration accuracy per oversample setting.

### `_embeddings_for_results(self, results, chunk_lookup, limit)`

Retrieve embeddings for the top-N results using a chunk lookup.

Args:
results: Search results from which embeddings are needed.
chunk_lookup: Mapping from (doc_id, chunk_id) to stored payloads.
limit: Maximum number of results to consider.

Returns:
List of embedding vectors associated with the results.

### `_average_pairwise_cos(self, embeddings)`

Compute average pairwise cosine similarity for a set of embeddings.

Args:
embeddings: Sequence of embedding vectors.

Returns:
Mean pairwise cosine similarity, or 0.0 when insufficient points exist.

### `_ensure_validation_resources(self)`

Lazy-create and cache GPU resources for validation-only cosine checks.

Args:
None

Returns:
FAISS GPU resources reused across validation runs.

### `_percentile(self, values, percentile)`

Return percentile value for a sequence; defaults to 0.0 when empty.

Args:
values: Sequence of numeric values.
percentile: Desired percentile expressed as a value between 0 and 100.

Returns:
Percentile value cast to float, or 0.0 when the sequence is empty.

### `_scale_stability(self, dataset, inputs_by_doc, rng, query_sample_size)`

Stress-test search stability under repeated queries and churn.

Args:
dataset: Dataset entries containing documents and queries.
inputs_by_doc: Mapping of document IDs to ingestion inputs.
rng: Random generator used for sampling queries and churn operations.
query_sample_size: Number of queries to sample for stability checks.

Returns:
ValidationReport detailing stability mismatches.

### `_dirty(snapshot, namespace)`

*No documentation available.*

### `run_dense_search(current_k)`

Query FAISS for dense document candidates at the requested depth.

Args:
current_k: Number of vector matches to request from the dense index.

Returns:
Dense similarity matches ordered by score for the current document search.

### `_resolve_embedding(candidate)`

*No documentation available.*

## 3. Classes

### `RequestValidationError`

Raised when the caller submits an invalid search request.

Attributes:
None

Examples:
>>> raise RequestValidationError("page_size must be positive")
Traceback (most recent call last):
...
RequestValidationError: page_size must be positive

### `DenseSearchStrategy`

Stateful helper encapsulating dense search heuristics.

### `ChannelResults`

Results from a single retrieval channel (BM25, SPLADE, or dense).

``embeddings`` stores an optional matrix aligned with ``candidates`` for
downstream GPU deduplication and diversification reuse.

### `ReciprocalRankFusion`

Combine ranked lists using Reciprocal Rank Fusion.

### `ResultShaper`

Collapse duplicates, enforce quotas, and generate highlights.

### `HybridSearchService`

Execute BM25, SPLADE, and dense retrieval with fusion.

Attributes:
_config_manager: Source of runtime hybrid-search configuration.
_feature_generator: Component producing BM25/SPLADE/dense features.
_faiss: Default FAISS index used for namespaces routed to the shared store.
_faiss_router: Namespace-aware router managing FAISS stores.
_opensearch: Lexical index used for BM25 and SPLADE lookups.
_registry: Chunk registry providing metadata and FAISS id lookups.
_observability: Telemetry facade for metrics and traces.

Examples:
>>> # File-backed config manager (JSON/YAML on disk)
>>> from pathlib import Path  # doctest: +SKIP
>>> manager = HybridSearchConfigManager(Path("config.json"))  # doctest: +SKIP
>>> from DocsToKG.HybridSearch.interfaces import DenseVectorStore  # doctest: +SKIP
>>> from DocsToKG.HybridSearch.router import FaissRouter  # doctest: +SKIP
>>> from DocsToKG.HybridSearch.store import FaissVectorStore, ManagedFaissAdapter, OpenSearchSimulator  # doctest: +SKIP
>>> dense_store: DenseVectorStore = ManagedFaissAdapter(  # doctest: +SKIP
...     FaissVectorStore(dim=16, config=HybridSearchConfig().dense)
... )
>>> router = FaissRouter(per_namespace=False, default_store=dense_store)  # doctest: +SKIP
>>> service = HybridSearchService(  # doctest: +SKIP
...     config_manager=manager,
...     feature_generator=FeatureGenerator(embedding_dim=16),
...     faiss_index=dense_store,
...     opensearch=OpenSearchSimulator(),
...     registry=ChunkRegistry(),
...     faiss_router=router,
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

### `PaginationCheckResult`

Result of a pagination verification run.

Attributes:
cursor_chain: Sequence of pagination cursors encountered.
duplicate_detected: True when duplicate results were observed.

Examples:
>>> result = PaginationCheckResult(cursor_chain=["cursor1"], duplicate_detected=False)
>>> result.duplicate_detected
False

### `HybridSearchValidator`

Execute validation sweeps and persist reports.

Attributes:
_ingestion: Chunk ingestion pipeline used for preparing test data.
_service: Hybrid search service under validation.
_registry: Registry exposing ingested chunk metadata.
_opensearch: OpenSearch simulator for lexical storage inspection.

Examples:
>>> validator = HybridSearchValidator(
...     ingestion=ChunkIngestionPipeline(...),  # doctest: +SKIP
...     service=HybridSearchService(...),      # doctest: +SKIP
...     registry=ChunkRegistry(),
...     opensearch=OpenSearchSimulator(),
... )
>>> isinstance(validator, HybridSearchValidator)
True
