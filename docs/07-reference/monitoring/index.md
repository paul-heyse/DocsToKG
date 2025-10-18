# 1. Monitoring & Observability

Proactive monitoring keeps ingestion pipelines, ontology downloaders, and hybrid search responsive.

## 2. Key Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `hybrid_search.query_latency_ms` | `DocsToKG.HybridSearch.observability` | End-to-end latency per namespace |
| `hybrid_search.self_hit_accuracy` | Validation harness output | Accuracy drift indicator |
| `faiss.index_size` | FAISS manager | Number of vectors stored in the dense index |
| `faiss.rebuild_required` | `should_rebuild_index` flag | Signals when to rebuild dense index |
| `ontology.download_success_rate` | Ontology CLI logs | Successful fetch percentage |
| `ingestion.documents_processed` | Content pipeline logs | Document throughput |

### Content Download Telemetry

- Aggregate reason codes from manifest/attempt logs to distinguish conditional hits (`conditional_not_modified`), voluntary skips (`skip_large_download`), and enforced quota exits (`domain_max_bytes`).
- Surface the `resume_disabled` flag emitted in attempt metadata when callers still request HTTP range resume—the downloader now ignores the flag but the tag helps spot outdated automation.
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
