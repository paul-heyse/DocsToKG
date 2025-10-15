# Monitoring & Observability

Proactive monitoring keeps ingestion pipelines, ontology downloaders, and hybrid search responsive.

## Key Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `hybrid_search.query_latency_ms` | `DocsToKG.HybridSearch.observability` | End-to-end latency per namespace |
| `hybrid_search.self_hit_accuracy` | Validation harness output | Accuracy drift indicator |
| `faiss.index_size` | FAISS manager | Number of vectors stored in the dense index |
| `faiss.rebuild_required` | `should_rebuild_index` flag | Signals when to rebuild dense index |
| `ontology.download_success_rate` | Ontology CLI logs | Successful fetch percentage |
| `ingestion.documents_processed` | Content pipeline logs | Document throughput |

## Dashboards

- Create Grafana dashboards grouping latency, recall, and ingestion throughput.
- Highlight SLO targets (e.g., 95th percentile latency < 300 ms, self-hit accuracy ≥ 0.95).
- Include alerting thresholds that page operators when metrics exceed bounds.

## Logging

- Configure structured logging for ingestion jobs and hybrid search API.
- Persist ontology download manifests and validation reports under `~/.data/ontology-fetcher`.
- Send logs to centralised platforms (ELK, OpenSearch) with namespaces tags for filtering.

## Alerting

- **Latency**: Alert when 95th percentile exceeds SLO for 5 minutes.
- **Accuracy**: Alert when validation harness reports <0.9 self-hit accuracy.
- **Ingestion Backlog**: Alert when queued documents exceed threshold or pipeline stalls.
- **Ontology Failures**: Alert on consecutive download failures for key ontologies.

Refer to `docs/hybrid_search_runbook.md` for response procedures when alerts trigger.
