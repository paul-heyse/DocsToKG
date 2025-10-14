# Module: validation

Automated validation harness for the hybrid search stack.

## Functions

### `load_dataset(path)`

Load a JSONL dataset describing documents and queries.

### `infer_embedding_dim(dataset)`

Infer dense embedding dimensionality from dataset vector artifacts.

### `main(argv)`

*No documentation available.*

### `run(self, dataset, output_root)`

*No documentation available.*

### `_to_document(self, payload)`

*No documentation available.*

### `_check_ingest_integrity(self)`

*No documentation available.*

### `_check_dense_self_hit(self)`

*No documentation available.*

### `_check_sparse_relevance(self, dataset)`

*No documentation available.*

### `_check_namespace_filters(self, dataset)`

*No documentation available.*

### `_check_pagination(self, dataset)`

*No documentation available.*

### `_check_highlights(self, dataset)`

*No documentation available.*

### `_check_backup_restore(self, dataset)`

*No documentation available.*

### `_request_for_query(self, query, page_size)`

*No documentation available.*

### `_persist_reports(self, summary, output_root, calibration_details)`

*No documentation available.*

### `_run_calibration(self, dataset)`

*No documentation available.*

## Classes

### `HybridSearchValidator`

Execute validation sweeps and persist reports.
