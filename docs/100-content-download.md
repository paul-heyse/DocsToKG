# DocsToKG • ContentDownload — Subsystem Architecture

## Purpose & Scope

Acquire scholarly artifacts from OpenAlex-seeded sources using **resolver pipelines** with **centralized rate limiting**, **polite headers**, **robots compliance**, **resume semantics**, **durable telemetry**, and **fallback resilience**.

**In scope**: Resolver orchestration, download strategies (PDF/HTML/XML), caching/resume, manifests/SQLite catalog, polite networking, work orchestration, bounded concurrency, HTTP/S3/Postgres backend support.  
**Out of scope**: Chunking/embedding, ontology alignment, downstream analytics, document parsing, knowledge graph construction.

## Architecture Overview

ContentDownload orchestrates multiple **resolver tiers** to acquire scholarly articles, falling back through tiers until successful. It maintains a **catalog** of artifacts (SQLite or Postgres), enforces **per-host rate limits**, respects **robots.txt**, and emits **append-only telemetry** for reproducibility.

```mermaid
flowchart TB
  subgraph Input
    A[OpenAlex Query API]
    B[Work IDs File]
  end
  
  subgraph Orchestration
    C[Work Queue<br/>Priority + Heartbeat]
    D[Worker Pool<br/>Fixed Size]
    E[Job Dispatcher<br/>Rate-Aware]
  end
  
  subgraph Resolution
    F1[Tier 1: Direct OA<br/>Unpaywall+arXiv+PMC]
    F2[Tier 2: DOI Redirect]
    F3[Tier 3: Landing Scrape]
    F4[Tier 4: Wayback Archive]
  end
  
  subgraph HTTP Layer
    G[Polite HTTP Client<br/>Rate Limiter]
    H[Robots Cache]
    I[HTTP Cache<br/>Hishel RFC 9111]
  end
  
  subgraph Storage
    J[Artifact Store<br/>FS / S3]
    K[Catalog DB<br/>SQLite / Postgres]
    L[Telemetry Sink<br/>JSONL + SQLite]
  end
  
  A --> C
  B --> C
  C --> D
  D --> E
  E --> F1
  F1 -.->|fail| F2
  F2 -.->|fail| F3
  F3 -.->|fail| F4
  F1 & F2 & F3 & F4 --> G
  G --> H
  G --> I
  G --> J
  J --> K
  K --> L
  
  style F1 fill:#90ee90
  style F2 fill:#ffeb3b
  style F3 fill:#ffa500
  style F4 fill:#ff6b6b
```

## External Interfaces

**CLI**: `python -m DocsToKG.ContentDownload.cli`

- `content pull --query '<openalex|ids file>' --out runs/content --workers 8`
- `content resume --run runs/content/<run_id> --verify`
- `content report --run runs/content/<run_id> --format md`
- `content catalog stats --db-path <path> --format {json|table}`
- `content catalog query --db-path <path> --sql '<query>'`

**Programmatic**:

```python
from DocsToKG.ContentDownload.orchestrator import ContentOrchestrator
from DocsToKG.ContentDownload.config import ContentDownloadConfig

# Load config from YAML + environment variables
config = ContentDownloadConfig.from_yaml("config.yaml")

# Create orchestrator
orchestrator = ContentOrchestrator(config)

# Run acquisition (async)
await orchestrator.run()

# Or use sync wrapper
orchestrator.run_sync()
```

**Key APIs:**

- `request_with_retries()`: Polite HTTP with rate limiting and circuit breakers
- `DownloadRun`: Main orchestration class for a download session
- `ResolverPipeline`: Tier-based resolver execution with fallback
- `CatalogConnector`: Pluggable storage backend (SQLite/Postgres/RDS+S3)

## Inputs & Outputs

### Inputs

1. **OpenAlex works query**
   - Topic/topic-id filters (e.g., `topics.id:T12345`)
   - Years range (`from_publication_date`, `to_publication_date`)
   - Source filters (`sources.id`, `institutions.id`)
   - Returns: Work metadata (DOI, PMID, PMCID, arxiv_id, locations, open_access status)

2. **Resolver presets**
   - Tiered configuration (see `config/fallback.yaml`)
   - Per-source policies (timeout, retries, rate limits)
   - Custom headers (User-Agent with contact email)

3. **Rate policies**
   - Global defaults (`max_concurrent`, `backoff_base_ms`, `backoff_cap_ms`)
   - Per-host overrides (`arxiv.org: 0.33/second`, `unpaywall.org: 10/second`)
   - Obey robots.txt directives

4. **Accept rules**
   - Content-type filters (`application/pdf`, `text/html`, `application/xml`)
   - Role-specific Accept headers (metadata/landing/artifact)

### Outputs

**Artifacts**: Organized by type and year

- `PDF/<year>__<title-slug>__<work_id>.pdf`
- `HTML/<year>__<title-slug>__<work_id>.html`
- `XML/<year>__<title-slug>__<work_id>.xml`

**Telemetry**: Multi-format for different use cases

- `manifest.jsonl`: Append-only JSONL (fast appends, streaming ingest)
- `manifest.sqlite3`: SQLite catalog (complex queries, resume logic)
- `manifest.last.csv`: Fast resume hint (last N successful)
- `manifest.metrics.json`: Run-level statistics (yield, cache hits, timing)

**Schema Examples:**

```json
// JSONL manifest line
{
  "run_id": "01J...",
  "doc_id": "openalex:W12345",
  "stage": "fetch",
  "url": "https://arxiv.org/pdf/2401.00001",
  "canonical_url": "https://arxiv.org/pdf/2401.00001",
  "role": "artifact",
  "status": "ok",
  "http_status": 200,
  "mime": "application/pdf",
  "size_bytes": 3441123,
  "sha256": "abc123...",
  "path": "artifacts/pdf/2024__example__W12345.pdf",
  "attempt": 1,
  "duration_ms": 812,
  "retry_after_s": 0,
  "cache_hit": false,
  "resolver": "arxiv_pdf",
  "tier": "direct_oa",
  "fallback_chain": ["unpaywall_pdf", "arxiv_pdf"],
  "created_at": "2025-10-23T00:00:00Z",
  "config_hash": "sha256:...",
  "worker_id": "worker-01"
}
```

## Control Flow (Runtime)

### High-Level Flow

```mermaid
sequenceDiagram
  participant CLI
  participant Orchestrator
  participant Queue as WorkQueue
  participant Worker as Worker Pool
  participant Pipeline as ResolverPipeline
  participant HTTP as PoliteHttpClient
  participant Storage as ArtifactStore
  participant Catalog
  
  CLI->>Orchestrator: run(config)
  Orchestrator->>Queue: enqueue_works(openalex_works[])
  Orchestrator->>Worker: start_workers(count=8)
  
  loop While work pending
    Worker->>Queue: claim_work()
    Queue-->>Worker: work_item
    Worker->>Pipeline: execute_tiers(work_item)
    
    loop For each tier
      Pipeline->>HTTP: resolve_candidate(tier, work)
      HTTP-->>Pipeline: candidate_url
      Pipeline->>HTTP: head_check(url)
      HTTP-->>Pipeline: 200 OK
      Pipeline->>HTTP: download(url)
      HTTP-->>Storage: stream_bytes()
      Storage-->>Pipeline: artifact_path
    end
    
    Pipeline-->>Worker: artifact_result
    Worker->>Catalog: upsert_entry(doc_id, artifact)
    Worker->>Catalog: emit_telemetry(manifest_line)
  end
  
  Orchestrator-->>CLI: run_complete
```

### Detailed Steps

1. **Initialization** (`cli.main()` → `args.resolve_config()` → `DownloadRun.setup_*`)
   - Parse CLI arguments or load config file
   - Override with environment variables (`DOCSTOKG_CONTENT_*`)
   - Initialize storage backend (SQLite/Postgres)
   - Setup HTTP client with Hishel caching
   - Initialize rate limiter and circuit breakers

2. **Work Queueing** (`DownloadRun.enqueue_works()`)
   - Query OpenAlex API or read work IDs from file
   - Filter works already in catalog (resume logic)
   - Prioritize works by publication year (recent first)
   - Add works to priority queue with heartbeat tracking

3. **Worker Pool Execution** (`WorkerPool.run()`)
   - Spawn fixed number of worker threads/processes
   - Each worker claims work from queue (with lease extension)
   - Execute resolver pipeline for claimed work
   - Update catalog and emit telemetry
   - Handle graceful shutdown on SIGTERM/SIGINT

4. **Resolver Pipeline** (`ResolverPipeline.execute()`)
   - Iterate through tiers in order (Tier 1 → 2 → 3 → 4)
   - For each tier, try resolvers in parallel (if configured)
   - Perform HEAD check (robots.txt, rate limit, status code)
   - Download artifact if HEAD check passes
   - Verify checksum (SHA-256)
   - Store artifact atomically
   - Return on first successful download
   - Break to next tier if all resolvers fail

5. **Finalization** (`DownloadRun.finalize()`)
   - Flush all telemetry buffers
   - Write manifest.last.csv hint file
   - Generate manifest.metrics.json
   - Close database connections
   - Emit run summary

## Key Invariants & Policies

### Manifest Schema

- **Schema v3 (JSONL)**: Append-only, one record per attempt
- **SQLite schema v4**: Normalized tables (artifacts, attempts, catalog_entries)
- **Bump versions in lockstep** on breaking changes
- **Include `config_hash`** for reproducibility tracking

### URL Deduplication

- **Canonical index**: URLs normalized before catalog/rate-limiter lookup
- **Normalization rules**:
  - Lowercase scheme and host
  - Remove trailing slash
  - Sort query parameters
  - Remove tracking parameters (utm_*, fbclid, etc.)
  - Strip fragment (`#...`)
- **Prevents duplicate downloads** across workers and re-runs

### Role-Specific Headers

- **Metadata role** (OpenAlex API):
  - `Accept: application/json`
  - `User-Agent: DocsToKG/1.0 (mailto:lab@university.edu)`
- **Landing role** (publisher landing pages):
  - `Accept: text/html`
  - `User-Agent: Mozilla/5.0 ... DocsToKG/1.0`
- **Artifact role** (PDF/XML downloads):
  - `Accept: application/pdf, application/xml`
  - `User-Agent: DocsToKG/1.0 (mailto:lab@university.edu)`

### Rate Limiting

**Backend Options:**

- `memory`: In-memory token bucket (single process)
- `multiprocess`: Shared memory (multiple processes, single machine)
- `sqlite`: File-based coordination (multiple processes)
- `redis`: Distributed (multiple machines)
- `postgres`: Database-backed (multiple machines)

**Keying Strategy:**

- Primary key: `(host, role)` tuple
- Secondary key: `(service_name)` for API endpoints
- Example: `("arxiv.org", "artifact")` → `0.33 req/second`

**Policy Format:**

```yaml
rate_limit:
  backend: redis
  redis_url: redis://host:6379/0
  policies:
    arxiv.org:
      artifact: "0.33/second"  # 1 req / 3 seconds
    unpaywall.org:
      metadata: "10/second"
```

## Observability & SLOs

### Service Level Indicators (SLIs)

- **Yield**: Successfully acquired artifacts / attempted works
- **TTFP (Time To First PDF)**: p50 and p95 latency
- **Cache Hit %**: HTTP cache effectiveness
- **Rate Delay p95**: Time spent waiting for rate limiter
- **429 Ratio**: Rate limit violations / total requests
- **Breaker Opens**: Circuit breaker activations per host
- **Corruption**: Checksum mismatches (must be 0)

### Service Level Objectives (SLOs)

- Yield ≥ 85%
- TTFP p50 ≤ 3s
- TTFP p95 ≤ 10s
- Cache Hit ≥ 40% (second run)
- 429 Ratio < 0.01 (less than 1%)
- Corruption = 0 (hard requirement)

### CLI Telemetry Commands

```bash
# Summary statistics
content report --run runs/content/<run_id> --format md

# Export to Parquet for analysis
content telemetry export --run <run_id> --format parquet --out analysis.parquet

# SQL inspection
content catalog query --db-path manifest.sqlite3 --sql '
  SELECT resolver, tier, COUNT(*) as downloads, AVG(duration_ms) as avg_ms
  FROM attempts WHERE status="ok"
  GROUP BY resolver, tier ORDER BY downloads DESC
'

# Yield by tier
content catalog stats --db-path manifest.sqlite3 --group-by tier
```

## Failure Modes & Recovery

### HTTP Failures

**429 (Rate Limit Exceeded)**

- **Detection**: HTTP status 429 + optional `Retry-After` header
- **Recovery**: Honor `Retry-After` (capped at 60s), exponential backoff
- **Prevention**: Per-host rate limiter with conservative defaults

**5xx (Server Errors)**

- **Detection**: HTTP status 500-599
- **Recovery**: Exponential backoff (250ms base, 8000ms cap)
- **Prevention**: Circuit breaker opens after 5 consecutive failures

### Circuit Breaker States

```mermaid
stateDiagram-v2
  [*] --> Closed
  Closed --> Open : 5 failures in 60s
  Open --> HalfOpen : 30s timeout
  HalfOpen --> Closed : 3 successes
  HalfOpen --> Open : 1 failure
```

**State Transitions:**

- **Closed**: Normal operation
- **Open**: Block all requests for 30s
- **Half-Open**: Allow 3 test requests
- Exponential backoff on Open → HalfOpen (30s, 60s, 120s, ...)

### Data Integrity

**Checksum Mismatches**

- **Detection**: SHA-256 mismatch after download
- **Recovery**:
  1. Purge partial artifact
  2. Clear HTTP cache entry
  3. Increment attempt counter
  4. Retry (up to 3 times)
  5. Mark as failed if exhausted
- **Prevention**: Atomic writes (temp file → rename)

**Partial Writes**

- **Detection**: File size check before rename
- **Recovery**: Cleanup temp file, retry download
- **Prevention**: Use atomic rename, verify size

### Resume from Failures

```bash
# Resume from last checkpoint
content resume --run runs/content/<run_id>

# Force redownload specific works
content resume --run <run_id> --force --filter "status=failed"

# Verify checksums and redownload mismatches
content resume --run <run_id> --verify
```

## Security & Politeness

### HTTPS Enforcement

- **Default**: All downloads over HTTPS
- **Allow HTTP**: Explicit allowlist per host (e.g., localhost, internal networks)
- **Certificate validation**: Enabled by default, can disable for testing

### Robots.txt Compliance

- **Respect by default**: `obey_robots: true` for all resolvers
- **Override per resolver**: API endpoints can set `obey_robots: false`
- **Caching**: Robots.txt cached for 24 hours per host
- **Crawl-delay**: Honor `Crawl-delay` directive as minimum between requests

### URL Canonicalization

- **Remove tracking parameters**: utm_*, fbclid, gclid, etc.
- **Prevents**:
  - Rate limiter key drift (same resource, different URLs)
  - Cache misses (same content, different tracking params)
  - Duplicate downloads (work around URL variations)

### Secrets Management

- **Environment variables**: `DOCSTOKG_CONTENT_API_KEY_*`
- **Never logged**: Secrets masked in structured logs
- **Never in manifests**: Only include public URLs and identifiers
- **Rotation support**: Reload config without restart (SIGHUP)

## Deployment Architectures

### Development (SQLite + Local FS)

**Use Case**: Single researcher, local machine, < 100k papers

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
  cache_dir: ./cache/http
rate_limit:
  backend: memory
```

**Scaling Limits**: ~500 works/hour, ~1TB storage, single machine

### Enterprise (Postgres + NFS)

**Use Case**: Research lab, shared NFS, < 1M papers

```yaml
storage:
  backend: postgres
  connection_string: postgresql://user:pass@host:5432/catalog
  pool_size: 20
artifacts:
  root: /nfs/artifacts
  cas_enabled: true
workers: 32
http:
  timeout: 30
  cache_enabled: true
  cache_dir: /nfs/cache/http
rate_limit:
  backend: postgres
  connection_string: postgresql://user:pass@host:5432/ratelimit
```

**Scaling Limits**: ~5,000 works/hour, ~10TB storage, 4-8 machines

### Cloud (RDS + S3)

**Use Case**: Multi-tenant, cloud-native, > 1M papers

```yaml
storage:
  backend: postgres
  connection_string: postgresql://user:pass@rds.amazonaws.com:5432/catalog
  pool_size: 50
artifacts:
  root: s3://bucket/artifacts
  cas_enabled: true
  cas_backend: s3
  multipart_threshold: 50MB
workers: 128
http:
  timeout: 30
  cache_enabled: true
  cache_backend: redis
  cache_url: redis://elasticache:6379/0
rate_limit:
  backend: redis
  redis_url: redis://elasticache:6379/1
```

**Scaling Limits**: ~50,000 works/hour, unlimited storage, auto-scaling workers

## Roadmap

### Completed (v1.0)

- ✅ Tiered resolver architecture with fallback
- ✅ SQLite and Postgres catalog backends
- ✅ S3 artifact storage
- ✅ Per-host rate limiting with multiple backends
- ✅ Circuit breakers and exponential backoff
- ✅ Robots.txt enforcement
- ✅ HTTP caching (RFC 9111 via Hishel)
- ✅ Resume semantics (JSONL/CSV/SQLite)
- ✅ Content-addressed storage (CAS)
- ✅ Bounded concurrency with work queues

### In Progress (v1.1)

- 🔄 Additional resolvers (Semantic Scholar, CORE, BASE)
- 🔄 Richer artifact classification (supplementary materials, code, data)
- 🔄 Policy packs for specific domains (biomedicine, computer science)

### Future (v2.0)

- 📋 Incremental delta ingestion from OpenAlex
- 📋 Provenance tracking for derived works
- 📋 Multi-region S3 replication
- 📋 gRPC API for programmatic access
- 📋 Integration with ContentDownload catalog for deduplication
