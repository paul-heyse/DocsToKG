Here’s a **single, self-contained architecture guide** you can drop in as
`src/DocsToKG/ContentDownload/ARCHITECTURE.md`.
It explains how the pieces fit, what to configure, and how to extend—using the explicit-DI, strong-telemetry, atomic-write, and Pydantic-v2 config design we’ve aligned on.

---

# ContentDownload — Architecture & Operator Guide

## TL;DR (mental model)

ContentDownload is a **pipeline** that takes an *artifact/work item* (e.g., a DOI or URL), asks a **priority-ordered set of resolvers** for a concrete download plan, **fetches** the content with **rate-limited + retried HTTP**, writes to disk **atomically**, and emits **structured telemetry** for every attempt and the final outcome.
Everything is **explicitly injected**—no globals. All behavior is controlled via a **typed Pydantic v2 config** loaded from file ⊕ env ⊕ CLI.

```
[Runner/CLI]
   │  builds
   ▼
[ContentDownloadConfig] ───────────────┐
   │  into                             │  into
   ▼                                   ▼
[Resolver Registry] → ordered resolvers → [ResolverPipeline]
                                           │
                                           ▼
                                      [Download Execution]
                                           │
                                ┌──────────┴──────────┐
                                ▼                     ▼
                         [Robots Guard]       [HTTP (rate/retry/breaker)]
                                │                     │
                                └──────────┬──────────┘
                                           ▼
                                      [Atomic Write]
                                           │
                                           ▼
                                 [Manifest + Telemetry]
```

---

## 1) Goals & Non-Goals

**Goals**

* Deterministic, testable: **explicit dependencies** (telemetry, run_id, config, client) passed through your call chain.
* **Full observability**: attempt-level telemetry for all network decisions; a single place to record final outcomes.
* **Safety**: atomic writes with integrity checks (`Content-Length`, PDF tail, etc.).
* **Pluggable**: resolvers live as modules and register themselves.
* **Operator friendly**: one **typed config**; clear precedence (file < env < CLI); strong validation.

**Non-Goals**

* Hidden context/globals; implicit magic is avoided.
* Resolver-specific scraping heuristics: resolvers should *determine a plan*, not perform content transformation.

---

## 2) Runtime Actors (and what they own)

* **Runner/CLI** (`cli/app.py`): loads **ContentDownloadConfig**, constructs dependencies (HTTP client, telemetry sinks), builds resolvers list from the **registry**, creates the **ResolverPipeline**, and starts processing.
* **ContentDownloadConfig** (`config/models.py`): **single source of truth** for all knobs—HTTP, retries/backoff, rate limits, robots policy, download policy (atomic, chunks, size cap), telemetry sinks, resolver order/enablement.
* **Resolver Registry** (`resolvers/`): discovers and instantiates resolvers in the configured order. Each resolver implements `resolve(...) -> ResolverResult`.
* **ResolverPipeline** (`pipeline.py`): orchestrates the work: iterate resolvers, log attempts, select a plan, call Download Execution, finalize + record manifest.
* **Download Execution** (`download_execution.py`): three composable phases:

  * `prepare_candidate_download(...)` — HEAD/probe/policy decisions.
  * `stream_candidate_payload(...)` — GET with retries/backoff and streaming.
  * `finalize_candidate_download(...)` — integrity checks, file move, outcome build.
* **HTTP Layer**: a single injected client (requests/httpx) wrapped with:

  * **Rate limiter** (token bucket per resolver),
  * **Retry/backoff** (tunable per resolver, honors `Retry-After`),
  * Optional **Circuit Breaker** (to fail fast for sick upstreams).
* **Robots Guard** (`robots.py`): cached, ttl-bound robots.txt check for landing-page fetches.
* **Telemetry/Manifest** (`telemetry.py`): uniform **AttemptRecord** and **record_pipeline_result**; emits to CSV/console/OTLP/etc. via sinks.

---

## 3) Data Contracts (shared types)

Use these thin, explicit types across the pipeline:

```python
@dataclass(frozen=True)
class DownloadPlan:
    url: str
    resolver_name: str
    referer: str | None = None
    expected_mime: str | None = None

@dataclass(frozen=True)
class DownloadStreamResult:
    path_tmp: str
    bytes_written: int
    http_status: int
    content_type: str | None

@dataclass(frozen=True)
class DownloadOutcome:
    ok: bool
    path: str | None
    classification: str          # "success" | "skip" | "error"
    reason: str | None = None
    meta: dict[str, Any] | None = None

@dataclass(frozen=True)
class AttemptRecord:
    ts: datetime
    run_id: str | None
    resolver: str | None
    url: str
    verb: str                    # "HEAD" | "GET" | "ROBOTS"
    status: str                  # "http-head" | "http-get" | "retry" | ...
    http_status: int | None = None
    content_type: str | None = None
    elapsed_ms: int | None = None
    bytes_written: int | None = None
    content_length_hdr: int | None = None
    reason: str | None = None    # normalized reason token
    extra: Mapping[str, Any] = field(default_factory=dict)
```

**Status & reason taxonomy** (stable, grep-able):

* `status`: `http-head`, `http-get`, `http-200`, `http-304`, `robots-fetch`, `robots-disallowed`, `retry`, `size-mismatch`, `content-policy-skip`, `download-error`.
* `reason`: `ok`, `not-modified`, `retry-after`, `backoff`, `robots`, `policy-type`, `policy-size`, `timeout`, `conn-error`, `tls-error`, `too-large`, `unexpected-ct`.

---

## 4) End-to-End Control Flow (golden path & branches)

### 4.1 Golden path (200 OK)

```
Artifact → ResolverPipeline
   ├─ build resolvers from registry (ordered)
   ├─ for resolver in order:
   │    ├─ resolver.resolve(...) → DownloadPlan
   │    └─ if plan:
   │         ├─ prepare_candidate_download(..., telemetry, run_id)
   │         │    └─ HEAD → Attempt(status="http-head", ...)
   │         ├─ stream_candidate_payload(..., telemetry, run_id)
   │         │    ├─ GET (rate-limited, retried) → Attempt("http-get", ...)
   │         │    └─ atomic write to tmp + fsync + rename
   │         ├─ finalize_candidate_download(...)
   │         │    └─ integrity checks → Outcome(ok=True, path=...)
   │         └─ record_pipeline_result(outcome)
   └─ return
```

### 4.2 Conditional 304 → skip download

* If you have ETag/Last-Modified cached, GET includes conditional headers.
* `304 Not Modified` → Attempt(status=`"http-304"`, reason=`"not-modified"`), `Outcome(classification="skip")`.

### 4.3 Robots disallowed (landing page)

* RobotsCache denies → Attempt(`"robots-disallowed"`), `Outcome("skip", reason="robots")`.

### 4.4 Size mismatch / truncated

* `Content-Length` present but bytes mismatch → Attempt(`"size-mismatch"`), remove tmp, `Outcome("error", reason="size-mismatch")`.

### 4.5 Backoff/retry

* 429/5xx/connection errors → bounded retries with jitter; each retry emits Attempt(`"retry"`, `extra={"attempt": i, "sleep_ms": ...}`).

---

## 5) Configuration (Pydantic v2) — model & precedence

**Single top-level: `ContentDownloadConfig`** (file < env < CLI).
Highlights:

```python
class ContentDownloadConfig(BaseModel):
    run_id: str | None
    http: HttpClientConfig                # UA, timeouts, TLS, proxies
    robots: RobotsPolicy                  # enabled, ttl
    download: DownloadPolicy              # atomic_write, chunk_size, verify_content_length, max_bytes
    telemetry: TelemetryConfig            # sinks (csv/...), manifest paths
    resolvers: ResolversConfig            # order + per-resolver enable/overrides
```

**Example (YAML)**

```yaml
run_id: "2025-10-20T231500Z"
http:
  user_agent: "DocsToKG/ContentDownload (+mailto:research@example.com)"
  timeout_connect_s: 10
  timeout_read_s: 60
robots:
  enabled: true
  ttl_seconds: 3600
download:
  atomic_write: true
  verify_content_length: true
  chunk_size_bytes: 1048576
  max_bytes: 104857600    # 100 MiB cap
telemetry:
  sinks: [csv]
  csv_path: "logs/attempts.csv"
  manifest_path: "logs/manifest.jsonl"
resolvers:
  order: ["unpaywall","crossref","arxiv","europe_pmc","core","doaj","s2","landing","wayback"]
  unpaywall:
    enabled: true
    retry: { max_attempts: 4, retry_statuses: [429,500,502,503,504] }
  crossref:
    enabled: true
```

**Env overlay** (double underscore → nest):

```
DTKG_DOWNLOAD__MAX_BYTES=209715200
DTKG_RESOLVERS__ORDER='["arxiv","landing","wayback"]'
```

**CLI overlay** (Typer):

```
contentdownload run --config cd.yaml --resolver-order arxiv,landing --no-robots --chunk-size 2097152
```

---

## 6) Resolver System

* Each resolver lives in `resolvers/<name>.py` and registers itself:

```python
@register("unpaywall")
class UnpaywallResolver:
    priority = 10
    @classmethod
    def from_config(cls, rcfg: UnpaywallConfig, root: ContentDownloadConfig):
        return cls(email=rcfg.email, timeout=rcfg.timeout_read_s or root.http.timeout_read_s)
    def resolve(self, artifact, session, ctx, telemetry, run_id) -> ResolverResult:
        # return DownloadPlan(...) or empty result
```

* **Guidelines**:

  * Do **not** perform heavy network I/O in `resolve` unless it’s the goal of that resolver (e.g., landing page fetch).
  * Apply **rate limiting & retry** via the shared HTTP layer; do not reinvent it.
  * Set `expected_mime` in your `DownloadPlan` to help type policy decisions (e.g., skip when `Content-Type` mismatches).

* **Add a new resolver** (5 steps):

  1. Create `resolvers/<name>.py`, implement class, decorate with `@register("<name>")`.
  2. Add a per-resolver config model if needed (extending `ResolverCommonConfig`).
  3. Wire `from_config` optional overrides (timeouts, rate limits).
  4. Add it to default `resolvers.order` if broadly useful.
  5. Create focused tests: order, enable/disable, resolution behavior.

---

## 7) HTTP, Retries, Rate Limits, Breaker

* **Rate limiter**: token-bucket per resolver (`capacity`, `refill_per_sec`, `burst`). Log sleep as Attempt with `status="retry"`, `reason="backoff"` and `extra.sleep_ms`.
* **Retry policy**: `max_attempts`, `retry_statuses`, `base_delay_ms`, `max_delay_ms`, `jitter_ms`, `strategy` (`exponential`/`constant`). Honor `Retry-After` when present.
* **Circuit Breaker** (optional): **Open** on consecutive failures, **Half-Open** after `reset_timeout`, close on success. Emits state-change telemetry via listeners.

**Design rule:** these controls live in **config**, not code; download execution uses them **uniformly**. No ad-hoc sleeps in resolvers.

---

## 8) Disk I/O Safety & Integrity

* **Atomic write**: stream GET → temp file in destination dir → `fsync()` → `os.replace()` final path → directory `fsync()`.
* **Content-Length**: if header exists and verification is enabled, **enforce exact match**; mismatch → delete temp, `Outcome(error, "size-mismatch")`.
* **PDF tail check**: for PDFs, scan last bytes for `%%EOF` (defense against truncation).
* **Max bytes**: if configured, stop streaming once threshold exceeded; log `content-policy-skip` with `reason="policy-size"`.

---

## 9) Telemetry & Manifest

* **Every network action** produces an `AttemptRecord` with `run_id`, `resolver`, `url`, `verb`, `status`, `http_status`, `content_type`, `elapsed_ms`, and optional `bytes_written` / `content_length_hdr`.
* **Final manifest** recorded through `record_pipeline_result(...)` with:

  * `artifact_id`, `resolver`, `outcome.classification`, `reason`, `path`, `content_type`, `html_paths` (if any), `run_id`, and a **`config_hash`** (SHA256 of the normalized config) to make runs reproducible.

**Sinks**: CSV, JSONL, console, future OTLP. Emission is **best-effort**: telemetry must never crash the pipeline.

---

## 10) CLI Surfaces (operator UX)

* `run` — process work with the effective config.
* `print-config` — show the **merged** config (file ⊕ env ⊕ CLI).
* `validate-config` — fail fast on schema errors.
* `explain` — show resolver order, disabled/unknown resolvers.

**Examples**

```bash
# Dry-run with new order
contentdownload run -c cd.yaml --resolver-order unpaywall,crossref,landing

# Inspect effective config
contentdownload print-config -c cd.yaml | jq .

# Validate a config file
contentdownload validate-config -c cd.yaml
```

---

## 11) Testing Strategy (what to test where)

* **Unit (fast)**:

  * Config parsing & precedence; bad-key rejections.
  * Rate limiter math (token bucket edge cases).
  * Retry jitter bounds and `Retry-After` handling.
  * Robots cache TTL behavior; deny/allow decisions.
  * Atomic writer behaviors (happy path, exceptions, size mismatch).

* **Integration (mock HTTP)**:

  * Golden path (HEAD 200, GET 200), attempt records & manifest parity.
  * 304 conditional path → skip outcome + attempt.
  * 429 backoff → retry → success; attempts contain `sleep_ms`.
  * Robots disallow → no GET attempted; proper attempt + outcome.

* **Resolver**:

  * Registry order & enable/disable.
  * Minimal resolver returns a valid `DownloadPlan` for expected artifacts.

---

## 12) Extension Playbooks

### Change rate limits for a resolver

```yaml
resolvers:
  order: ["unpaywall","landing"]
  unpaywall:
    rate_limit: { capacity: 2, refill_per_sec: 0.5, burst: 1 }
```

### Disable robots (debug only)

```
contentdownload run -c cd.yaml --no-robots
```

### Enforce 100 MiB hard cap

```yaml
download:
  max_bytes: 104857600
```

### Add a new telemetry sink

* Implement `AttemptSink` with `log_attempt(self, rec: AttemptRecord)`.
* Add to `telemetry.sinks` (e.g., `"otlp"`), wire config (endpoint, headers).

---

## 13) Design Tenets (to keep code intuitive)

1. **Explicit in, explicit out**: pass `telemetry` + `run_id` + `config` explicitly.
2. **Typed edges**: use `DownloadPlan/Outcome/AttemptRecord`; avoid dict-shaped contracts.
3. **One place for truth**: `ContentDownloadConfig` governs behavior; the code reads, doesn’t reinterpret.
4. **Emit early, emit often**: log attempts at the *source* of I/O and decisions.
5. **Fail safe on disk**: if integrity is uncertain, delete the partial and surface a clear `reason`.
6. **Small resolvers**: resolve *what* to fetch; **execution** handles *how*.
7. **Test the contract, not internals**: assert on outcomes + attempts + manifest rows.
8. **No globals**: DI over contextvars; `DownloadContext.extra` is not a hiding place for core services.
9. **Stable tokens**: telemetry `status/reason` strings are part of the public contract.
10. **Backwards compatible by default**: new features ship enabled with sane defaults, but are toggleable for troubleshooting.

---

## 14) FAQ

**Q: Where do I add per-resolver timeouts?**
A: Prefer `ResolverCommonConfig.timeout_read_s` override; `from_config` copies it into the resolver’s calls.

**Q: How do I guarantee reproducibility?**
A: Run with a fixed `run_id`; keep the emitted `config_hash` alongside logs/manifests and pin resolver order.

**Q: Can I pass artifacts with prior ETag/IMS?**
A: Yes—stash them in your artifact metadata and let Download Execution attach conditional headers; 304s will be recorded properly.

---

That’s the whole picture. If you’d like, I can drop this into an `ARCHITECTURE.md` file with a small `README.md` pointer and add an `explain` CLI command that prints the loaded resolver list with per-resolver effective policies (limits, retries, timeouts) for quick operator introspection.
