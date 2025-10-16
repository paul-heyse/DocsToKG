# 1. Module: pipeline

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.pipeline``.

## 1. Overview

Ingestion pipeline, feature generation, and observability utilities.

## 2. Functions

### `increment(self, name, amount)`

Increase a counter metric by ``amount`` for the supplied label set.

Args:
name: Counter metric identifier.
amount: Amount to add to the counter.
**labels: Key/value labels that partition the metric stream.

### `observe(self, name, value)`

Record a new observation for a histogram metric.

Args:
name: Histogram metric identifier.
value: Observation value to append.
**labels: Key/value labels that partition the metric stream.

### `set_gauge(self, name, value)`

Store the latest value for a gauge metric.

Args:
name: Gauge metric identifier.
value: Current value to store.
**labels: Key/value labels that partition the metric stream.

### `percentile(self, name, percentile)`

Return the requested percentile for a histogram metric if available.

Args:
name: Histogram metric identifier.
percentile: Desired percentile expressed between 0.0 and 1.0.
**labels: Key/value labels that partition the metric stream.

Returns:
The percentile value when samples exist, otherwise ``None``.

### `export_counters(self)`

Yield counter samples suitable for serialization.

### `export_histograms(self)`

Yield histogram samples enriched with common percentiles.

### `export_gauges(self)`

Yield gauge samples representing the latest recorded values.

### `span(self, name)`

Record a timed span, emitting metrics and logs with ``attributes``.

### `metrics(self)`

Return the metrics collector used by the ingestion pipeline.

### `logger(self)`

Return the structured logger used for observability events.

### `trace(self, name)`

Create a tracing span that records timing and metadata.

### `metrics_snapshot(self)`

Export a JSON-serializable snapshot of counters, histograms, and gauges.

### `metrics(self)`

Expose cumulative ingestion metrics for inspection.

Args:
None

Returns:
IngestMetrics capturing counts of upserts and deletions.

### `faiss_index(self)`

Access the FAISS index manager used for vector persistence.

Args:
None

Returns:
DenseVectorStore associated with the ingestion pipeline.

### `upsert_documents(self, documents)`

Ingest pre-computed chunk artifacts into FAISS and OpenSearch.

Args:
documents: Sequence of document inputs referencing chunk/vector files.

Returns:
List of `ChunkPayload` objects that were successfully upserted.

Raises:
RetryableIngestError: When transformation fails due to transient issues.

### `delete_chunks(self, vector_ids)`

Delete chunks from FAISS, OpenSearch, and the registry by vector id.

Args:
vector_ids: Collection of vector identifiers to remove.

Returns:
None

Raises:
None

### `_prepare_faiss(self, new_chunks)`

Train the FAISS index if required before adding new vectors.

Args:
new_chunks: Newly ingested chunks whose embeddings may train the index.

Returns:
None

### `_training_sample(self, new_chunks)`

Select representative embeddings for FAISS training.

Args:
new_chunks: Candidate chunks from the current ingestion batch.

Returns:
Sequence of embedding vectors used for index training.

### `_load_precomputed_chunks(self, document)`

Load chunk and vector artifacts from disk for a document.

Args:
document: Document input describing artifact locations and metadata.

Returns:
List of populated `ChunkPayload` instances.

Raises:
IngestError: If chunk and vector artifacts are inconsistent or missing.

### `_delete_existing_for_doc(self, doc_id, namespace)`

Remove previously ingested chunks for a document/namespace pair.

Args:
doc_id: Document identifier whose chunks should be removed.
namespace: Namespace to scope the deletion.

Returns:
None

### `_features_from_vector(self, payload)`

Convert stored vector payload into ChunkFeatures.

Args:
payload: Serialized feature payload from the vector JSONL artifact.

Returns:
ChunkFeatures object with BM25, SPLADE, and dense embeddings.

Raises:
IngestError: If the dense embedding has unexpected dimensionality.

### `_weights_from_payload(self, payload)`

Deserialize sparse weight payloads into a term-to-weight mapping.

Args:
payload: Mapping containing `terms`/`tokens` and `weights` arrays.

Returns:
Dictionary mapping each term to its corresponding weight.

### `_read_jsonl(self, path)`

Load JSONL content from disk and parse each line into a dictionary.

Args:
path: Path to the JSONL artifact.

Returns:
List of parsed entries.

Raises:
IngestError: If the artifact file is missing.
json.JSONDecodeError: If any JSON line cannot be parsed.

### `_deque_factory()`

*No documentation available.*

## 3. Classes

### `CounterSample`

Sample from a counter metric with labels and value.

### `HistogramSample`

Sample from a histogram metric with percentile statistics.

### `GaugeSample`

Sample from a gauge metric capturing the latest recorded value.

### `MetricsCollector`

In-memory metrics collector compatible with Prometheus-style summaries.

### `TraceRecorder`

Context manager producing timing spans for tracing.

### `Observability`

Facade for metrics, structured logging, and tracing.

### `IngestError`

Base exception for ingestion failures.

Args:
message: Description of the ingestion failure.

Examples:
>>> raise IngestError("invalid chunk metadata")
Traceback (most recent call last):
...
IngestError: invalid chunk metadata

### `RetryableIngestError`

Errors that callers should retry (e.g., transient model inference).

Args:
message: Description of the transient ingestion issue.

Examples:
>>> raise RetryableIngestError("embedding service unavailable")
Traceback (most recent call last):
...
RetryableIngestError: embedding service unavailable

### `IngestMetrics`

Simple metrics bundle used by tests.

Attributes:
chunks_upserted: Number of chunks upserted during ingestion runs.
chunks_deleted: Number of chunks removed from storage backends.

Examples:
>>> metrics = IngestMetrics(chunks_upserted=3)
>>> metrics.chunks_upserted
3

### `ChunkIngestionPipeline`

Coordinate loading of chunk/vector artifacts and dual writes.

Attributes:
_faiss: FAISS index manager responsible for vector persistence.
_opensearch: Lexical index handling sparse storage.
_registry: Registry mapping vector identifiers to chunk metadata.
_metrics: Aggregated ingestion metrics recorded during operations.
_observability: Observability facade for tracing and logging.

Examples:
>>> pipeline = ChunkIngestionPipeline(
...     faiss_index=DenseVectorStore(...),
...     opensearch=OpenSearchSimulator(),  # from DocsToKG.HybridSearch.store  # doctest: +SKIP
...     registry=ChunkRegistry(),
... )
>>> isinstance(pipeline.metrics.chunks_upserted, int)
True
