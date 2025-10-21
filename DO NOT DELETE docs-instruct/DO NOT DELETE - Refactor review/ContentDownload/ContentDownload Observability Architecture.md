# ContentDownload Observability — **Architecture & Narrative Companion**

> This document complements the Grafana dashboard you just imported.
> It explains **how the system is wired end-to-end**, what each metric/panel means, recommended thresholds/SLOs, alert examples, and a quick troubleshooting playbook.

---

## 0) The Big Picture (what’s running where)

```mermaid
flowchart LR
  subgraph App["ContentDownload process"]
    RP[ResolverPipeline]
    RRC[RateRetryClient (per resolver)]
    HC[(httpx.Client)]
    HT[hishel CacheTransport + Controller + Storage]
    NET[[Origin servers]]
    TEL[RunTelemetry (fan-out sinks)]
    CSV[(attempts.csv)]
    JNL[(manifest.jsonl)]
    OTEL[OTel SDK (traces+metrics)]
  end

  RP -- plan->client map --> RRC
  RRC --> HC
  HC --> HT
  HT --> NET
  RRC --> TEL
  TEL --> CSV
  TEL --> JNL
  TEL --> OTEL

  subgraph Obs["Observability path"]
    COL[OTel Collector]
    PROM[(Prometheus)]
    GRAF[Grafana]
  end

  OTEL --> COL
  COL --> PROM
  PROM --> GRAF
```

**Key ideas**

* **One shared httpx.Client** (connection pool) uses **hishel** to cache and revalidate `GET`s.
* **Per-resolver RateRetryClient** wraps that client to enforce **token-bucket rate limits**, **retry/backoff**, and to **emit attempt telemetry**.
* **RunTelemetry** fans out to **CSV/JSONL** for audit and **OTel** for real-time dashboards/alerts.
* Grafana reads **Prometheus** (via the OTel Collector) to render the dashboard panels.

---

## 1) Signals we produce (and who produces them)

| Signal                                                                                         | Producer                              | Why it exists                                                       |
| ---------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------- |
| **Attempt CSV row** (`status`, `reason`, `verb`, `http_status`, `bytes_written`, `elapsed_ms`) | RateRetryClient + execution functions | Audit trail; cheapest, greppable source of truth.                   |
| **Manifest JSONL** (final `outcome` per artifact)                                              | Pipeline via RunTelemetry             | Long-term record used for bookkeeping and backfills.                |
| **OTel Metrics** (`*_total`, `*_ms_bucket`)                                                    | OTelAttemptSink + OTelManifestSink    | Real-time rates, latencies, cache KPIs, retries.                    |
| **OTel Spans** (`artifact`, `prepare`, `http.request`, `stream`, `finalize`)                   | Wrapper + execution + pipeline        | Deep timing and attributes (cache flags, status codes).             |
| **hishel metadata** (`resp.extensions.from_cache`, `revalidated`)                              | hishel                                | The authoritative cache signals used in metrics/spans and attempts. |

**Stable tokens** (used across CSV, metrics labels, and dashboards)

* `status`: `http-head`, `http-get`, `cache-hit`, `http-304`, `http-200`, `retry`, `size-mismatch`, `download-error`
* `reason`: `ok`, `not-modified`, `retry-after`, `backoff`, `conn-error`, `policy-type`, `policy-size`, `size-mismatch`

---

## 2) How the dashboard panels map to your code paths

| Panel                                     | PromQL (summary)                                                 | Code path (source)                                                 | What it answers                                                                     |
| ----------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Cache Hit Ratio (overall/by resolver)** | hit / (hit+200+304)                                              | `resp.extensions.from_cache && !revalidated` → attempt `cache-hit` | “How often are GETs served *entirely* from cache?”                                  |
| **Revalidation Ratio**                    | 304 / (200+304)                                                  | `revalidated && status=304` → attempt `http-304`                   | “Given a network check, how often do we avoid body transfer?”                       |
| **Success Rate**                          | success / (success+error)                                        | Final manifest outcomes                                            | “Are we pulling cleanly end-to-end?”                                                |
| **GET/HEAD latency p50/p90/p99**          | `histogram_quantile` on `contentdownload_http_latency_ms_bucket` | Attempt `http-get` / `http-head` with `elapsed_ms`                 | “Handshake & initial response latency (not total bytes stream time).”               |
| **Retries by reason**                     | rate of `contentdownload_retries_total{reason}`                  | RateRetryClient retry events                                       | “Are we back-pressured (429), flaky (conn-error), or just exponential backing off?” |
| **Bytes/sec by resolver**                 | rate of `contentdownload_bytes_total`                            | `http-200` attempt with `bytes_written`                            | “Throughput of actual payload bytes to disk.”                                       |
| **Attempt rate by status**                | rate by `status`                                                 | Attempts                                                           | “Composition of work: cache-hit vs 200 vs 304.”                                     |
| **Outcomes by resolver**                  | rate of `contentdownload_outcomes_total{outcome}`                | Final outcomes                                                     | “Success/skip/error shape over time.”                                               |
| **Rate-limit sleep (optional)**           | quantiles on `contentdownload_rate_sleep_ms_bucket`              | Token bucket sleep histograms                                      | “How much time are we waiting to be polite?”                                        |

> **Tip:** Keep **labels low-cardinality** in metrics (`resolver`, `status`, `reason`, `verb`). Avoid URL labels.

---

## 3) What “healthy” looks like (targets & thresholds)

Use these as starting points; tune by resolver/domain:

* **Cache Hit Ratio**: 40–80% on steady workloads that re-visit URLs; lower for novelty-heavy ingest.

  * **Yellow**: < 40%; **Green**: > 70%.
* **Revalidation Ratio**: 20–60% (depends on origin headers).

  * **Yellow**: < 20%; **Green**: > 50%.
* **Success Rate**: > 95% (exclude `skip` as non-error).

  * **Red**: < 90%, **Yellow**: 90–95%, **Green**: > 95%.
* **GET p90**: target < 500 ms to first byte for fast APIs; PDFs may be higher.

  * **Red**: > 1500 ms (tune per network).
* **Retries (retry-after)**: spikes indicate provider throttling → consider lowering rate or adding jitter.
* **Bytes/sec**: baseline for capacity; sudden drops with steady work = upstream or network issue.

---

## 4) Suggested alerts (Prometheus rules)

```yaml
groups:
- name: contentdownload.rules
  rules:
  - alert: ContentDownloadLowSuccessRate
    expr: sum(rate(contentdownload_outcomes_total{outcome="success"}[5m])) /
          sum(rate(contentdownload_outcomes_total{outcome=~"success|error"}[5m])) < 0.9
    for: 10m
    labels: { severity: page }
    annotations:
      summary: "Low success rate (<90%)"
      description: "Success rate is {{ $value | printf \"%.2f\" }} over 10m."

  - alert: ContentDownloadCacheHitCollapse
    expr: sum(rate(contentdownload_attempts_total{status="cache-hit"}[10m])) /
          sum(rate(contentdownload_attempts_total{status=~"cache-hit|http-200|http-304"}[10m])) < 0.25
    for: 30m
    labels: { severity: ticket }
    annotations:
      summary: "Cache hit ratio dropped below 25%"
      description: "Cache may be cold or storage unhealthy."

  - alert: ContentDownloadThrottled
    expr: sum(rate(contentdownload_retries_total{reason="retry-after"}[5m])) > 0
    for: 15m
    labels: { severity: ticket }
    annotations:
      summary: "Provider throttling (Retry-After)"
      description: "Consistent 429s. Consider lowering rate_limit capacity/refill or add jitter."

  - alert: ContentDownloadLatencySpikes
    expr: histogram_quantile(0.90, sum by (le) (rate(contentdownload_http_latency_ms_bucket[5m]))) > 1500
    for: 15m
    labels: { severity: ticket }
    annotations:
      summary: "High p90 GET/HEAD latency"
      description: "Investigate network or origin slowness."
```

---

## 5) Bring-up checklist (first run validation)

1. **Run with OTel enabled** (collector reachable) and CSV sinks on.
2. **Open the dashboard**; verify:

   * Attempts/time are non-zero; Outcomes are changing with work.
   * GET latency curves appear; cache-hit is near zero on the first warm-up.
3. **Run the same workload again**:

   * Cache hit ratio rises; revalidation ratio stabilizes.
   * Retry-after should be ~0 unless you’re pushing a provider.
4. **Spot check CSV vs panels**:

   * Pick a time; grep `attempts.csv` for `cache-hit` lines and compare rates.

---

## 6) Troubleshooting (symptoms → likely causes → next steps)

| Symptom                                 | Likely causes                                                                            | Where to look                                                               | Next steps                                                                                 |
| --------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Cache Hit Ratio plummets**            | Cache path cleared; controller set to `always_revalidate`; origin cache headers disabled | Cache panels; OTel spans (`hishel.from_cache=false`); hishel storage        | Check config; verify storage health; ensure `allow_heuristics=false` unless you intend it. |
| **High Retry-After**                    | You’re throttled (429)                                                                   | “Retries by reason” panel; `http.request` spans with `http.status_code=429` | Reduce rate limit capacity/refill; stagger workloads; add jitter.                          |
| **High GET p90**                        | Network or origin slowness; DNS issues                                                   | Latency panel; spans with long GET; resolver-specific breakdown             | Compare by resolver/host; enable HEAD gating; consider parallelism cap.                    |
| **Bytes/sec drops but attempts steady** | Failures or frequent 304s/cache-hits (no bytes streamed)                                 | Attempt composition; Outcomes panel                                         | Confirm expected (lots of revalidations/cache hits) vs unexpected (errors).                |
| **Spans empty; panels flatline**        | Collector down; OTel disabled; wrong datasource                                          | Grafana “datasource” var; Collector logs                                    | Toggle `otel_exporter=console` to verify local emit; fix collector/datasource.             |

---

## 7) How each dashboard panel helps you decide

* **Cache Hit Ratio (overall/by resolver)**
  *Tuning lever*: hishel controller; backends (file vs redis), cache horizon, workload mix.
  *Interpretation*: rising hit ratio after warm-up = healthy.

* **Revalidation Ratio**
  *Tuning lever*: origin compliance; `always_revalidate` (avoid unless needed).
  *Interpretation*: moderate 304 share implies efficient freshness checks.

* **Success Rate**
  *Tuning lever*: retry/backoff windows; rate limit; robots gating; integrity checks.
  *Interpretation*: dips likely correlate with provider incidents, policy changes, or integrity failures.

* **Latency p50/p90/p99**
  *Tuning lever*: per-resolver retry policies; `timeout_*`; avoid chatty HEADs.
  *Interpretation*: spikes focused on specific resolvers/hosts narrow the incident scope.

* **Retries by Reason**
  *Tuning lever*: rate limit & jitter; max attempts; base/max delay.
  *Interpretation*: `retry-after` = external throttle; `conn-error` = network/TLS or transient.

* **Bytes/sec & Attempt rate**
  *Tuning lever*: concurrency; rate buckets; I/O chunk size; destination storage.
  *Interpretation*: steady attempts but zero bytes often indicates 304s or cache-hits (expected).

* **Outcomes stacked**
  *Tuning lever*: fix recurrent `error` reasons (size-mismatch, download-error), or accept `skip` when correct (robots/not-modified).

---

## 8) SLO proposals (write them down)

* **SLO-1**: Success rate ≥ **95%** (rolling 24h), excluding `skip`.
* **SLO-2**: GET p90 ≤ **500 ms** for API domains; ≤ **1500 ms** for publisher file domains.
* **SLO-3**: Cache hit ratio ≥ **50%** for repeat workloads (moving 24h window).
* **SLO-4**: Retry-after rate ≤ **1 per 100 attempts** (5m rollup).

> Tie alerts to **error budget burn** (failure rate × time) to reduce noise.

---

## 9) Operational guidance (knobs that matter)

* **Rate limit**: If `retry-after` rises, lower `capacity`/`refill_per_sec` or increase jitter.
* **Retries**: If errors remain, increase `max_attempts` but cap `max_delay_ms` to avoid long stalls.
* **HEAD policy**: For heavy origins, consider disabling HEAD except for large-file detection (saves a roundtrip).
* **Chunk size**: Increase `download.chunk_size_bytes` on high-bandwidth links to reduce syscall overhead.
* **Caching backend**:

  * `file` for single host (fast, simple),
  * `redis` for shared cache across workers/hosts,
  * `s3` for durable global caches (higher latency).
* **Sampling**: Reduce `otel_sample_ratio` for spans when volume grows; metrics remain aggregated.

---

## 10) Glossary

* **Cache hit**: hishel served the full response from storage; no network I/O.
* **Revalidation**: a conditional GET was sent; `304 Not Modified` means we avoided the body.
* **Attempt**: a single decision point or network step (HEAD, GET, retry, terminal 200/304).
* **Outcome**: the final classification (`success|skip|error`) for an artifact.

---

## 11) Quick “Runbook” example

**Page: “Low success rate”**

1. Open **Outcomes by Resolver** → identify the worst resolver.
2. Check **Retries by Reason** → is it `conn-error` (transient) or `retry-after` (throttle)?
3. Inspect **Latency p90** → high latency often precedes errors.
4. Look at **Attempt composition** → lots of `http-304` or `cache-hit`? Then low bytes/sec is expected.
5. If **size-mismatch** spikes → verify origin `Content-Length` correctness; consider disabling strict CL verify for that domain temporarily.
6. Adjust **rate limit** or **retry policy** for the offending resolver; watch metrics for 15–30 min.

---

### That’s it—your dashboard now has a story

If you want a **second “Resolver Drilldown” dashboard** (focused on a single resolver/host with deeper legends: status/verb/reason split and span-link panels), I can generate it too.
