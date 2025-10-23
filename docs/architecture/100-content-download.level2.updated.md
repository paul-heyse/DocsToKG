# DocsToKG • ContentDownload (Level-2 Spec)

> **Go‑Forward Decisions (2025-10-23) — ContentDownload**
>
> 1. **CLI alignment.** Keep the current entrypoint `python -m DocsToKG.ContentDownload.cli_v2` (**`run`**, **`print-config`**, **`validate-config`**, **`explain`**, **`schema`**) and **ship a thin alias CLI** with verbs `content pull|resume|report|catalog` for external tooling. Both surfaces are supported; examples in docs show **both** until the alias is ubiquitous.
> 2. **Default orchestrator.** Make the **SQLite‑backed WorkQueue + Orchestrator + KeyedLimiter** the **default execution mode** (bounded concurrency, tier‑by‑tier). Keep `ResolverPipeline` for **debug/single‑shot** runs.
> 3. **Politeness, retries & robots.** Use **Tenacity** for backoff/jitter and honor **`Retry‑After`**. **HTTPS required** by default; host allowlist gates exceptions. **robots.txt obeyed** by default (API endpoints may opt out).
> 4. **HTTP caching & canonicalization.** Use **Hishel (RFC‑9111)** for request/response caching. Canonicalize URLs and strip tracking params to improve cache & rate‑limit keys.
> 5. **Storage & CAS.** Default to **local FS + SQLite manifest** (JSONL + SQLite). Enable **CAS** by default for dedupe/integrity. **Postgres + S3** remain recommended for multi‑node; keep as first‑class backends.
> 6. **Telemetry.** Emit structured JSONL + **Prometheus** metrics (`content_*`) and **OpenTelemetry traces**. Provide dashboards and SLOs wired to `429 ratio`, `TTFP`, yield, and rate‑delay.
> 7. **Resume & idempotency.** Re‑runs with unchanged inputs are **idempotent**. Resume consults **SQLite/JSONL/CSV** (in that order).
> 8. **Config precedence & env.** Adopt **`DOCSTOKG_CONTENT_*`** env keys going forward; continue to read legacy **`DTKG_*`** for backward compatibility.
> 9. **Acceptance gates.** Enforce **Yield ≥ 85%**, **TTFP p50 ≤ 3s / p95 ≤ 10s**, **429 ratio < 1%**, **Corruption = 0**, aligning with the end‑to‑end north‑star.


## Purpose & Non-Goals

**Purpose:** Deterministically acquire open research artifacts with **resume**, **politeness**, **append-only telemetry**, and **fallback resilience**.  
**Non-Goals:** Parsing/embedding/indexing, ontology download, knowledge graph construction.

## External Interfaces

**CLI (current commands via `python -m DocsToKG.ContentDownload.cli_v2`)**

- `run` – primary entry point (supports `--config`, `--workers`, `--resolver-order`, `--dry-run`)
- `print-config` – show the merged configuration (JSON pretty-print or raw)
- `validate-config` – validate a config file
- `explain` – list resolver ordering and registry status
- `schema` – emit the JSON Schema for `ContentDownloadConfig`

> Planned: thin aliases (`content pull|resume|report|catalog …`) will be added on top of these
> commands so external tooling can keep using the simplified verbs documented elsewhere.

**Python API (synchronous)**

```python
from DocsToKG.ContentDownload.config import load_config
from DocsToKG.ContentDownload.runner import DownloadRun

cfg = load_config("contentdownload.yaml")

with DownloadRun(cfg) as run:
    result = run.process_artifacts(artifacts)
    print(f"Saved {result.successful} / {result.total_processed}")
```

## Directory Layout

```
<output-root>/                         # defaults derive from config.telemetry.manifest_path
  manifest.jsonl                       # append-only JSONL (enabled when sinks include "jsonl")
  manifest.sqlite                      # SQLite manifest/telemetry store
  summary.json                         # aggregated summary payload
  last.csv                             # most recent successful attempts (when CSV sink enabled)
  attempts.csv                         # optional CSV sink for attempts
  index.json                           # URL → manifest metadata index
```

Artifacts are written under the storage root configured by `StorageConfig.root_dir`
(`data/docs` by default) using the policy-path layout. Future revisions will expose a
convenience runner that creates the `runs/content/<run_id>/…` hierarchy described in the
original architecture notes.

## Manifest Schema (JSONL line)

```json
{
  "run_id": "01J...",
  "doc_id": "openalex:W12345",
  "stage": "fetch",
  "url": "https://...",
  "canonical_url": "https://...",
  "role": "artifact",
  "status": "ok",
  "http_status": 200,
  "mime": "application/pdf",
  "size_bytes": 3441123,
  "sha256": "...",
  "path": "artifacts/pdf/example.pdf",
  "attempt": 1,
  "duration_ms": 812,
  "retry_after_s": 0,
  "cache_hit": true,
  "resolver": "unpaywall_pdf",
  "tier": "direct_oa",
  "fallback_chain": ["unpaywall_pdf"],
  "created_at": "2025-10-23T00:00:00Z",
  "config_hash": "sha256:...",
  "worker_id": "worker-01"
}
```

## SQLite Manifest & Telemetry Schema (current implementation)

- `manifests` – canonical manifest entries (one row per attempt)
- `downloads` – finalized artifacts with dedupe metadata
- `summaries` – latest summary payload
- `http_events`, `rate_events`, `breaker_transitions`, `fallback_attempts`, `run_summary` – telemetry for HTTP, rate limiting, circuit breakers, fallback tiers, and run-level KPIs

`telemetry_schema.sql` documents the full schema emitted by `RunTelemetry`. Schema versioning is managed internally by `telemetry.ManifestSchemaMigrator`.

## Fallback Orchestration (Tiered Resolvers)

The system attempts resolvers in **tiers** with configurable parallelism per tier. Execution proceeds tier-by-tier; stops immediately upon first success.

**Default Tier Configuration:**

```yaml
version: 1
tiers:
  # Tier 1: Direct Open Access (parallel=2)
  - name: direct_oa
    parallel: 2
    sources:
      - unpaywall_pdf      # Unpaywall API (cached)
      - arxiv_pdf          # arXiv direct
      - pmc_pdf            # PMC direct

  # Tier 2: DOI Redirect (serial)
  - name: doi_follow
    parallel: 1
    sources:
      - doi_redirect_pdf   # Follow DOI to publisher

  # Tier 3: Landing Page Scrape (parallel=2)
  - name: landing_scrape
    parallel: 2
    sources:
      - landing_scrape_pdf # Scrape landing page
      - europe_pmc_pdf     # Europe PMC API

  # Tier 4: Archive (last resort)
  - name: archive
    parallel: 1
    sources:
      - wayback_pdf        # Wayback Machine CDX
```

**Per-Source Rate Policy:**

```yaml
policies:
  unpaywall_pdf:
    timeout_s: 10
    max_retries: 3
    rate_limit: "10/second"
    obey_robots: false      # API endpoint
    
  arxiv_pdf:
    timeout_s: 30
    max_retries: 5
    rate_limit: "0.33/second"  # 1 req / 3 seconds
    obey_robots: true
    custom_headers:
      User-Agent: "DocsToKG/1.0 (mailto:yourlab@university.edu)"
  
  pmc_pdf:
    timeout_s: 20
    max_retries: 3
    rate_limit: "3/second"
    obey_robots: true
```

## Work Orchestration & Concurrency

The repository includes a SQLite-backed work queue and bounded worker pool in
`DocsToKG.ContentDownload.orchestrator` (queue leasing, keyed limiters, threaded workers).
The default CLI is the **default** execution path (bounded concurrency). The prior sequential `ResolverPipeline` remains available for debug/single‑shot runs.

```mermaid
flowchart LR
  A[Artifact Iterator] --> B[ResolverPipeline (current default)]
  subgraph Planned Integration
    B2[WorkQueue] --> C2[Orchestrator]
    C2 --> D2{Tier Execution}
  end
```

**Key Components (available today):**

- `orchestrator.queue.WorkQueue` – SQLite lease/ack queue
- `orchestrator.scheduler.Orchestrator` – dispatcher & bounded worker pool
- `orchestrator.limits.KeyedLimiter` – per-host/per-resolver semaphores
- `pipeline.ResolverPipeline` – deterministic resolver + download stages
- `fallback.FallbackOrchestrator` – tiered resolver retries (optional)

## State Machine (Extended)

```mermaid
stateDiagram-v2
  [*] --> Queued
  Queued --> Scheduled : worker available
  Scheduled --> ResolveTier1 : start tier 1
  ResolveTier1 --> HeadCheck : candidate found
  ResolveTier1 --> ResolveTier2 : tier 1 exhausted
  HeadCheck --> Fetch : 200 OK / robots allowed
  HeadCheck --> ResolveTier2 : 404/403/robots denied
  Fetch --> Verify : download complete
  Verify --> Store : checksum OK
  Store --> Index : atomic write
  Index --> Success : catalog updated
  ResolveTier2 --> HeadCheck : candidate found
  ResolveTier2 --> ResolveTier3 : tier 2 exhausted
  ResolveTier3 --> HeadCheck : candidate found
  ResolveTier3 --> Failed : all tiers exhausted
  HeadCheck --> Backoff : 429/5xx
  Fetch --> Backoff : transient error
  Backoff --> ResolveTier1 : retry with delay
  Verify --> Failed : checksum mismatch
  Failed --> [*]
  Success --> [*]
```

## Invariants

- **Append-only manifest**: JSONL records never deleted, only appended
- **Content-addressed storage**: Optional CAS with SHA-256 for deduplication
- **Canonical URL deduplication**: URLs normalized before rate-limiter keying and catalog lookups
- **Role-specific headers**: Different User-Agent/Accept headers for metadata vs landing vs artifact requests
- **SQLite schema version pinned**: Schema v4 current; migrations required for breaking changes
- **Atomic writes**: All artifact writes use atomic rename from temporary files
- **Idempotent resume**: Re-running with same inputs produces identical catalog state

## Configuration Precedence

Configuration precedence today:

1. **Defaults** (Pydantic model defaults)
2. **File** (YAML/JSON via `--config`)
3. **Environment** (`DTKG_*` prefix, e.g. `DTKG_HTTP__USER_AGENT`)
4. **CLI overrides** (dict passed to `load_config`, e.g. `--workers`)

> Upcoming: support for the documented `DOCSTOKG_CONTENT_*` prefix so deployment
> guides and the CLI share the same environment variable naming.

**Example Environment Variables:**

```bash
DOCSTOKG_CONTENT_WORKERS=16
DOCSTOKG_CONTENT_TIMEOUT=30
DOCSTOKG_CONTENT_RATE_LIMIT_arxiv="0.33/second"
DOCSTOKG_CONTENT_ROBOTS_OBEY=true
DOCSTOKG_CONTENT_CACHE_ENABLED=true
```

## Resume Semantics

The system supports three resume strategies:

1. **JSONL Resume** – `telemetry.load_previous_manifest()` streams append-only records when `manifest.jsonl` exists.
2. **CSV Resume** – `last.csv` provides a compact hint file (when CSV sink enabled).
3. **SQLite Resume** – `JsonlResumeLookup` and friends query the `manifests` table inside `manifest.sqlite`.

**Resume Flow:**

```mermaid
sequenceDiagram
  participant CLI
  participant Resume as ResumeController
  participant SQLite
  participant Queue as WorkQueue
  
  CLI->>Resume: initialize(manifest_path)
  Resume->>SQLite: SELECT doc_id FROM catalog_entries WHERE status='ok'
  SQLite-->>Resume: completed_ids[]
  Resume->>Queue: filter_works(completed_ids)
  Queue-->>CLI: remaining_works[]
  CLI->>CLI: process(remaining_works)
```

## Performance & SLOs

- **TTFP p50 ≤ 3s**: Time to first PDF (median)
- **TTFP p95 ≤ 10s**: 95th percentile first PDF
- **Yield ≥ 85%**: Successfully acquired artifacts / attempted works
- **Cache hit ≥ 40%**: HTTP cache hit rate across re-runs
- **Throughput ≥ 500 works/hour**: Per worker, tier-1-only workload
- **Catalog write latency p95 ≤ 50ms**: SQLite transaction time

## Observability

**Prometheus Metrics:**

- `content_http_requests_total{host, status, resolver}` - Counter of HTTP requests
- `content_http_latency_ms_bucket{host, resolver}` - Histogram of request latency
- `content_yield_ratio{tier}` - Gauge of successful acquisitions per tier
- `content_breaker_open_total{host}` - Counter of circuit breaker opens
- `content_cache_hit_ratio` - Gauge of HTTP cache effectiveness
- `content_queue_depth` - Gauge of pending work items
- `content_workers_active` - Gauge of active worker count

**Structured Logs:**

- Backoff reasons (429, 5xx, robots denied, rate limit exceeded)
- Robots.txt decisions (allowed, blocked, timeout, parse error)
- Resolver fallback chains (tier progression, exhausted tiers)
- Worker lifecycle events (started, idle, shutdown)

**OpenTelemetry Traces:**

- Span per work item: `content.process_work{doc_id}`
- Child spans: `resolve.tier{tier}`, `download{url}`, `verify`, `catalog.write`
- Span attributes: `tier`, `resolver`, `http.status_code`, `cache.hit`, `duration_ms`

## Failure Modes

| Failure | Detection | Recovery | Prevention |
|---------|-----------|----------|------------|
| HTTP 429/5xx | Status code + retry-after | Exponential backoff + circuit breaker | Per-host rate limiting |
| Checksum mismatch | SHA-256 verification | Purge artifact, retry download | Atomic writes |
| Partial writes | File size check | Abort + cleanup temp files | Atomic rename |
| Network timeout | httpx timeout exception | Retry with exponential backoff | Conservative timeout values |
| Robots denied | RobotFileParser | Skip host, try next tier | Pre-flight robots check |
| Disk full | IOError on write | Graceful shutdown, alert | Pre-run disk space check |
| SQLite lock | sqlite3.OperationalError | Exponential backoff retry | Single writer per database |


## Networking & Reliability

- **Retries & Backoff**: **Tenacity** with exponential backoff + jitter; **`Retry‑After`** honored (429/5xx).
- **TLS & Allowlist**: HTTPS required; explicit allowlist for exceptions; HTTP→HTTPS upgrade when possible.
- **Robots Compliance**: obey robots.txt by default; API endpoints may opt out.
- **Caching**: **Hishel (RFC 9111)** for HTTP caching.


## Security
- **HTTPS required**: All downloads over HTTPS unless explicitly allowlisted
- **Redact query params**: Remove tracking params from landing page URLs
- **Honor robots.txt**: Respect crawl-delay and disallow directives
- **Rate limiting**: Per-host limits prevent abuse
- **User-Agent identification**: Include contact email for attribution
- **Secrets masking**: API keys/tokens never logged or in manifests

## Tests

**Unit Tests:**

- URL canonicalization (schema, case, trailing slash, query param order)
- Robots.txt parsing (allow/disallow, crawl-delay extraction)
- CAS path generation (sharding, determinism)
- Manifest record serialization (JSON schema validation)
- Resume logic (JSONL/CSV/SQLite correctness)

**Integration Tests:**

- Replay with HTTP cache (mock httpx responses)
- Host-specific rate limit accuracy (timing assertions)
- Fallback tier progression (mock resolver failures)
- SQLite catalog consistency (concurrent writes)
- Checkpoint and resume (partial run → resume → completion)

**Property Tests:**

- Canonical URL function is idempotent: `canon(canon(url)) == canon(url)`
- Manifest append maintains temporal ordering
- CAS storage is deterministic: same content → same path

## Deployment Configuration

**Development (SQLite + Local FS):**

```yaml
storage:
  backend: sqlite
  path: ./data/catalog.db
artifacts:
  root: ./data/artifacts
  cas_enabled: false
workers: 4
http:
  timeout: 15
  cache_enabled: true
```

**Production (Postgres + S3):**

```yaml
storage:
  backend: postgres
  connection_string: postgresql://user:pass@host:5432/catalog
  pool_size: 20
artifacts:
  root: s3://bucket/artifacts
  cas_enabled: true
  cas_backend: s3
workers: 64
http:
  timeout: 30
  cache_enabled: true
  cache_backend: redis
rate_limit:
  backend: redis
  redis_url: redis://host:6379/0
```
