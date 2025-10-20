# QA & Rollout Artifacts

## JSONL Run Snapshot

The following example records illustrate a `docparse embed` execution using the default JSONL writer. Paths are relative to the DocsToKG data root.

### Log Excerpt

```json
{"timestamp":"2025-02-20T14:22:11Z","level":"INFO","message":"Embedding file written","extra_fields":{"stage":"embedding","status":"success","doc_id":"teamA/report.doctags","input_relpath":"Data/ChunkedDocTagFiles/teamA/report.chunks.jsonl","output_relpath":"Data/Embeddings/teamA/report.vectors.jsonl","elapsed_ms":842,"vectors":12,"splade_avg_nnz":6.4,"qwen_avg_norm":0.998231,"vector_format":"jsonl"}}
```

### Manifest Entry

```json
{"stage":"manifest","doc_id":"teamA/report.doctags","status":"success","duration_s":0.842,"schema_version":"embeddings/1.0.0","input_path":"Data/ChunkedDocTagFiles/teamA/report.chunks.jsonl","input_hash":"8c041e5ba1f0841f2f4bcf7f4ff0cd8e6db6e0e0c0ad0e7e2407c4f418b6f9bb","hash_alg":"sha256","output_path":"Data/Embeddings/teamA/report.vectors.jsonl","vector_format":"jsonl","vector_count":12,"run_id":"2025-02-20T14:21:58Z-docparse-embed"}
```

## Parquet Run Snapshot

This example was taken from the same corpus executed with `--format parquet` after installing `DocsToKG[docparse-parquet]`.

### Log Excerpt

```json
{"timestamp":"2025-02-20T15:07:49Z","level":"INFO","message":"Embedding file written","extra_fields":{"stage":"embedding","status":"success","doc_id":"teamA/report.doctags","input_relpath":"Data/ChunkedDocTagFiles/teamA/report.chunks.jsonl","output_relpath":"Data/Embeddings/teamA/report.vectors.parquet","elapsed_ms":879,"vectors":12,"splade_avg_nnz":6.4,"qwen_avg_norm":0.998115,"vector_format":"parquet"}}
```

### Manifest Entry

```json
{"stage":"manifest","doc_id":"teamA/report.doctags","status":"success","duration_s":0.879,"schema_version":"embeddings/1.0.0","input_path":"Data/ChunkedDocTagFiles/teamA/report.chunks.jsonl","input_hash":"8c041e5ba1f0841f2f4bcf7f4ff0cd8e6db6e0e0c0ad0e7e2407c4f418b6f9bb","hash_alg":"sha256","output_path":"Data/Embeddings/teamA/report.vectors.parquet","vector_format":"parquet","vector_count":12,"run_id":"2025-02-20T15:07:23Z-docparse-embed"}
```

## Downstream Communications

To ensure a smooth rollout, share the following guidance with downstream owners:

1. **HybridSearch ingestion team**
   - Install the DocsToKG `docparse-parquet` extra (adds `pyarrow`) in every environment that ingests parquet vectors.
   - Set dataset entries to `vector_format: "parquet"` (or pass `--vector-format parquet` to the quickstart harness) so ingestion selects the correct reader.
   - Mixed datasets are now rejected with an actionable `IngestError`; keep namespaces homogeneous or continue using JSONL.

2. **Analytics / Manifest consumers**
   - Manifests now emit `vector_format` for every embedding row. Update dashboards or ETL jobs to record and filter on this field.
   - JSONL remains the default. No action is required unless opting into parquet artifacts.

3. **General reminder**
   - Refresh runbooks to call out the dependency and explicitly state that parquet is opt-in. Include the sample log and manifest snippets above in release notes so operators can verify deployments quickly.
