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

### `run_scale(self, dataset)`

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

### `_collect_queries(self, dataset)`

*No documentation available.*

### `_sample_queries(self, dataset, sample_size, rng)`

*No documentation available.*

### `_scale_data_sanity(self, documents, dataset)`

*No documentation available.*

### `_scale_crud_namespace(self, documents, dataset, inputs_by_doc, rng)`

*No documentation available.*

### `_scale_dense_metrics(self, thresholds, rng)`

*No documentation available.*

### `_scale_channel_relevance(self, dataset, thresholds, rng, query_sample_size)`

*No documentation available.*

### `_scale_fusion_mmr(self, dataset, thresholds, rng, query_sample_size)`

*No documentation available.*

### `_scale_pagination(self, dataset, rng, sample_size)`

*No documentation available.*

### `_scale_result_shaping(self, dataset, rng, sample_size)`

*No documentation available.*

### `_scale_backup_restore(self, dataset, rng, query_sample_size)`

*No documentation available.*

### `_scale_acl(self, dataset)`

*No documentation available.*

### `_scale_performance(self, dataset, thresholds, rng, query_sample_size)`

*No documentation available.*

### `_run_calibration(self, dataset)`

*No documentation available.*

### `_embeddings_for_results(self, results, chunk_lookup, limit)`

*No documentation available.*

### `_average_pairwise_cos(self, embeddings)`

*No documentation available.*

### `_percentile(self, values, percentile)`

*No documentation available.*

### `_scale_stability(self, dataset, inputs_by_doc, rng, query_sample_size)`

*No documentation available.*

## Classes

### `HybridSearchValidator`

Execute validation sweeps and persist reports.
