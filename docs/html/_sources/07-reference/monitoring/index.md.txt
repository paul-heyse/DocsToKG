# 1. Monitoring & Observability

Proactive monitoring keeps ingestion pipelines, ontology downloaders, and hybrid search responsive.

## 2. Key Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `hybrid_search_timings_total_ms` (histogram) | `DocsToKG.HybridSearch.service` | End-to-end latency per request; label by namespace. |
| `hybrid_search_results` (histogram) | `HybridSearchService` | Number of results emitted per query. |
| `hybrid_lexical_hit_rate` (histogram) | `HybridSearchService` | Ratio of lexical hits per request, useful for adaptive weighting. |
| `faiss_ntotal` (gauge) | `DocsToKG.HybridSearch.store` | Total vectors stored in the FAISS index. |
| `faiss_gpu_bytes` / `faiss_gpu_mem_free_bytes` (gauges) | `FaissVectorStore.stats()` | GPU memory footprint and headroom for the dense index. |
| `faiss_rebuilds` / `faiss_nprobe` (counters/gauges) | `FaissVectorStore` | Tracks rebuild events and effective `nprobe` tuning. |
| `ingestion.chunks_upserted` (counter) | `ChunkIngestionPipeline.metrics` | Number of chunks processed per ingestion run. |
| `ontology.download_success_rate` | Ontology CLI summaries | Successful fetch percentage across sources. |

### Content Download Telemetry

- Aggregate reason codes from manifest/attempt logs to distinguish conditional hits (`conditional_not_modified`) and voluntary skips (`skip_large_download`).
- Surface the `resume_disabled` flag emitted in attempt metadata when callers still request HTTP range resume—the downloader ignores the flag but the tag helps spot outdated automation.
- Treat missing reason codes as successful downloads in dashboards and notebooks.

## 3. Dashboards

- Create Grafana dashboards grouping latency, recall, and ingestion throughput.
- Highlight SLO targets (e.g., 95th percentile latency < 300 ms, self-hit accuracy ≥ 0.95).
- Include alerting thresholds that page operators when metrics exceed bounds.

## 4. Logging

- Configure structured logging for ingestion jobs and hybrid search API.
- Persist ontology download manifests and validation reports under `~/.data/ontology-fetcher`.
- Send logs to centralised platforms (ELK, OpenSearch) with namespaces tags for filtering.

## 5. Alerting

- **Latency**: Alert when 95th percentile exceeds SLO for 5 minutes.
- **Accuracy**: Alert when validation harness reports <0.9 self-hit accuracy.
- **Ingestion Backlog**: Alert when queued documents exceed threshold or pipeline stalls.
- **Ontology Failures**: Alert on consecutive download failures for key ontologies.

Refer to `docs/hybrid_search_runbook.md` for response procedures when alerts trigger.
