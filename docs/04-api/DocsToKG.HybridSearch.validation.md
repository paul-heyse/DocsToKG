# 1. Module: validation

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.validation``.

Automated validation harness for the hybrid search stack.

## 1. Functions

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

### `main(argv)`

CLI entrypoint for running hybrid search validation suites and consolidated
pytest workflows.

Args:
argv: Optional list of command-line arguments overriding `sys.argv`.

Returns:
None

Notes:
- ``--mode`` selects between ``basic`` and ``scale`` validation sweeps.
- ``--run-tests`` forwards to the hybrid search pytest suites (``synthetic``, ``real``, ``scale``, ``all``).
- ``--run-real-ci`` executes the real-vector regression harness and stores artifacts under ``--output-dir``.

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

## 2. Classes

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
