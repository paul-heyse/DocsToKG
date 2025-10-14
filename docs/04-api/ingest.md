# Module: ingest

Ingestion pipeline that materializes pre-computed chunk artifacts.

## Functions

### `metrics(self)`

*No documentation available.*

### `faiss_index(self)`

*No documentation available.*

### `upsert_documents(self, documents)`

*No documentation available.*

### `delete_chunks(self, vector_ids)`

*No documentation available.*

### `_prepare_faiss(self, new_chunks)`

*No documentation available.*

### `_training_sample(self, new_chunks)`

*No documentation available.*

### `_load_precomputed_chunks(self, document)`

*No documentation available.*

### `_delete_existing_for_doc(self, doc_id, namespace)`

*No documentation available.*

### `_features_from_vector(self, payload)`

*No documentation available.*

### `_weights_from_payload(self, payload)`

*No documentation available.*

### `_read_jsonl(self, path)`

*No documentation available.*

## Classes

### `IngestError`

Base exception for ingestion failures.

### `RetryableIngestError`

Errors that callers should retry (e.g., transient model inference).

### `IngestMetrics`

Simple metrics bundle used by tests.

### `ChunkIngestionPipeline`

Coordinate loading of chunk/vector artifacts and dual writes.
