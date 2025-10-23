# DocsToKG • Observability & SLOs

## Goals
Detect regressions quickly, explain incidents, and budget performance by stage.

## Signals by Subsystem
**ContentDownload**
- SLIs: download_yield, http_429_rate, ttfp_ms (p50/p95), cache_hit_rate
- Logs: per-host rate decisions, circuit breaker opens, retry reasons
- Traces: resolver pipeline spans

**DocParsing**
- SLIs: stage_latency_ms (doctags/chunk/embed), embed_dim_mismatch, resume_rate
- Logs: per-stage manifest writes, hash drift warnings
- Traces: file-level spans with GPU utilization samples

**HybridSearch**
- SLIs: ingest_throughput, query_latency_ms (p50/p99), snapshot_age_s, gpu_mem_used
- Logs: adapter/router stats, UUID mapping diffs
- Traces: per-request fan-out and fusion timing

**KnowledgeGraph**
- SLIs: alignment_yield, shacl_failure_rate, tx_retry_rate
- Logs: validation errors, ontology lockfile diffs
- Traces: batch write spans

**RAG Service / Agent Gateway**
- SLIs: non-gen latency (p50/p90), gen latency (p50/p90), citation_coverage, error_rate
- Logs: budget denials, degraded-mode flags
- Traces: retrieval plan + synthesis spans

## SLO Targets (initial)
- ContentDownload: yield ≥ 85%, ttfp_p50 ≤ 3s
- DocParsing (embed, GPU): p50 ≤ 2.2s per 1k chunks; zero dim mismatches
- HybridSearch: p50 ≤ 150ms, p99 ≤ 600ms (no-gen path)
- RAG (no-gen): p50 ≤ 300ms; Gateway error_rate < 1%

## Implementation
- Metrics: Prometheus/OpenMetrics exposition per service.
- Logs: structured JSONL + logfmt; sampling on hot paths.
- Tracing: OpenTelemetry SDK; W3C trace context; span attributes keyed by stage.

## Dashboards & Alerts
- Golden dashboards per subsystem.
- Burn-rate alerts for SLOs (multi-window).
