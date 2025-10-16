# 1. Module: ingest

This reference documents the DocsToKG module ``DocsToKG.HybridSearch.ingest``.

## 1. Overview

Ingestion pipeline that materializes pre-computed chunk artifacts.

## 2. Functions

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
FaissVectorStore associated with the ingestion pipeline.

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

## 3. Classes

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
_opensearch: OpenSearch simulator handling lexical storage.
_registry: Registry mapping vector identifiers to chunk metadata.
_metrics: Aggregated ingestion metrics recorded during operations.
_observability: Observability facade for tracing and logging.

Examples:
>>> pipeline = ChunkIngestionPipeline(
...     faiss_index=FaissVectorStore.build_in_memory(),
...     opensearch=OpenSearchSimulator(),
...     registry=ChunkRegistry(),
... )
>>> isinstance(pipeline.metrics.chunks_upserted, int)
True
