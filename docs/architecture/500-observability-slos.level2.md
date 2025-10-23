# DocsToKG • Observability & SLOs (Level-2 Spec)

## Metrics Catalog
- Content: `content_http_requests_total`, `content_http_latency_ms_bucket`, `content_yield_ratio`, `content_breaker_open_total`
- DocParsing: `docparse_doctags_pages_total`, `docparse_embed_rows_total`, `docparse_embed_dim_mismatch_total`, `docparse_stage_latency_ms_bucket{stage}`
- Hybrid: `hybrid_ingest_rows_total{ns}`, `hybrid_query_latency_ms_bucket{ns}`, `hybrid_snapshot_age_seconds{ns}`, `hybrid_gpu_memory_bytes{device}`
- KG: `kg_alignment_yield{ns}`, `kg_shacl_failures_total{shape,ns}`, `kg_tx_retries_total{ns}`
- RAG/Gateway: `rag_latency_ms_bucket{path}`, `rag_degraded_total{reason}`, `gateway_budget_denials_total{reason}`

## SLOs (initial targets)
- Content: yield ≥ 85%, TTFP p50 ≤ 3s
- DocParsing: embed p50 ≤ 2.2s per 1k chunks; 0 dim mismatches
- Hybrid: p50 ≤ 150ms, p99 ≤ 600ms (no-gen)
- RAG (no-gen): p50 ≤ 300ms; Gateway error_rate < 1%

## PromQL Examples
```promql
sum(rate(hybrid_query_latency_ms_bucket{le="600"}[5m])) / sum(rate(hybrid_query_latency_ms_count[5m])) < 0.99
```

## Tracing
OpenTelemetry with spans per stage; carry `run_id`, `config_hash`, `namespace`.

## Dashboards & Alerts
Golden dashboards per subsystem; multi-window burn alerts.
