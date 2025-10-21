# ContentDownload × hishel — Caching Architecture & Intuition Guide

> Drop this into your repo as `src/DocsToKG/ContentDownload/ARCHITECTURE_hishel.md`.
> It explains how we integrate **hishel** with **httpx**, our **rate-limit + retry** wrapper, the **pipeline**, and **telemetry**—including flow diagrams, config, edge cases, and concrete examples.

---

## 1) Why hishel here (intuition first)

Think of **hishel** as a “smart front door” for HTTP GETs:

* When a URL is fresh in the cache → serve instantly (**cache-hit**, no network).
* When freshness is uncertain → send a **conditional GET** (ETag/Last-Modified) and often get a **304 Not Modified** (cheap network, no body).
* When content changed → fetch a full **200** and update the cache.

This saves network, makes retries safer, and shortens end-to-end time. Our job is to **compose** hishel with:

* **httpx** for a modern client,
* a **RateRetryClient** wrapper for **per-resolver politeness** (token bucket + backoff),
* **telemetry** to make every cache/network decision observable,
* atomic writing & integrity checks so we never persist partial content.

---

## 2) Mental model (layers & responsibilities)

```
[ResolverPipeline]
   │ chooses a plan (url, resolver_name, expected_mime, …)
   ▼
[RateRetryClient for that resolver]  ← per-resolver rate-limit + retry + telemetry
   │ wraps
   ▼
[httpx.Client (shared)]             ← shared connection pools
   │ uses
   ▼
[hishel CacheTransport + Controller + Storage]   ← RFC 9111 caching (ETag/IMS, 304)
   │ sits over
   ▼
[HTTP transport / network]
```

* **One shared hishel cache** across the process (all resolvers benefit).
* **One shared httpx.Client**, wrapped by **per-resolver** `RateRetryClient` objects (to apply resolver-specific rate/retry).
* The **pipeline** passes the right client to execution based on `plan.resolver_name`.

---

## 3) Data flow (first run → cache-hit → revalidate)

```
First run:
  GET → 200 w/ ETag|Last-Modified → write body → hishel stores metadata → attempts: http-get, http-200

Second run (unchanged):
  GET → served by cache (no network) → attempts: http-get, cache-hit

Later (revalidate):
  GET + If-None-Match/If-Modified-Since → 304 → attempts: http-get, http-304 → outcome=skip
```

---

## 4) Config (single source of truth)

### 4.1 `ContentDownloadConfig.hishel` (excerpt)

```yaml
hishel:
  enabled: true
  backend: file               # file | sqlite | redis | s3
  base_path: state/hishel-cache
  ttl_seconds: 2592000        # at-rest storage TTL (not RFC freshness)
  check_ttl_every_seconds: 600

  # RFC controller knobs
  force_cache: false
  allow_heuristics: false
  allow_stale: false
  always_revalidate: false
  cache_private: true
  cacheable_methods: ["GET"]
```

### 4.2 HTTP & resolver policy (reused)

```yaml
http:
  user_agent: "DocsToKG/ContentDownload (+mailto:data@example.org)"
  timeout_connect_s: 10
  timeout_read_s: 60
resolvers:
  order: ["unpaywall","crossref","landing","wayback"]
  unpaywall:
    enabled: true
    retry: { max_attempts: 4, retry_statuses: [429,500,502,503,504], base_delay_ms: 200, max_delay_ms: 4000, jitter_ms: 100 }
    rate_limit: { capacity: 5, refill_per_sec: 1.0, burst: 2 }
download:
  atomic_write: true
  verify_content_length: true
  chunk_size_bytes: 1048576
```

---

## 5) Key modules (what each file owns)

### 5.1 `httpx/hishel_build.py`

* **Builds the hishel stack** from config:

  * **Storage**: file/sqlite/redis/s3 (with at-rest TTL sweeps).
  * **Controller**: RFC 9111 knobs (force_cache, always_revalidate, etc.).
  * **CacheTransport** chained over `httpx.HTTPTransport`.
* Builds the **shared `httpx.Client`** with timeouts, limits, headers, TLS, proxies.

### 5.2 `httpx/client.py` → `RateRetryClient`

* Wraps **the shared httpx.Client** to add:

  * **Token bucket rate limit** per resolver.
  * **Retry/backoff** with `Retry-After` & jitter.
  * **Telemetry** for each retry (`status="retry"`, `reason` = `retry-after | backoff | conn-error`).
  * **Refund tokens** for **pure cache hits** (no network used).

### 5.3 `bootstrap.py`

* Creates shared **hishel transport** and **httpx.Client**.
* Builds **per-resolver RateRetryClient** wrappers using resolver policy.
* Instantiates pipeline with:

  * resolvers,
  * **client map** (`resolver_name` → RateRetryClient),
  * telemetry + run_id,
  * robots & download policy knobs.

### 5.4 `download_execution.py`

* **prepare**: optional fast checks; may skip/err early.
* **stream**: does HEAD (optional) + **GET through hishel**:

  * Reads **`resp.extensions["from_cache"]`** and **`["revalidated"]`**.
  * Emits attempts: `http-get`, plus:

    * `cache-hit` (pure cache hit),
    * `http-304` with `reason="not-modified"`,
    * or continues to stream and then `http-200`.
* **finalize**: atomic rename, verify `Content-Length`, PDF tail, and build outcome.

### 5.5 `cli/app.py` (cache helpers)

* `contentdownload cache stats` / `clear` for file backend (simple and safe).

---

## 6) Telemetry contract (stable tokens)

Attempt CSV columns (unchanged):

```
ts,run_id,resolver,url,verb,status,http_status,content_type,elapsed_ms,bytes_written,content_length_hdr,reason
```

We use:

* `http-head` → HEAD probe
* `http-get` → GET handshake latency
* `cache-hit` → served from cache, no network
* `http-304` → revalidated, not modified (reason=`not-modified`)
* `http-200` → streamed body; emit `bytes_written` + `content_length_hdr`
* `retry` → backoff sleeps or `Retry-After` honored (reason=`backoff|retry-after|conn-error`)

Final **manifest JSONL** is unchanged (`outcome: success|skip|error`).

---

## 7) End-to-end example (one artifact, three runs)

**Artifact**: `doi:10.1234/abcd.5678`, plan → `unpaywall`, url → `https://cdn.publisher.org/article/abcd5678.pdf`, expected_mime `application/pdf`.

### Run 1 — initial fetch (200)

```
… http-head,200,application/pdf,92,,,
… http-get,200,application/pdf,145,,,
… http-200,200,application/pdf,,1245721,1245721,
```

Manifest: `outcome="success"`, `bytes=1245721`, `path=/data/docs/.../abcd5678.pdf`.

### Run 2 — cache-hit (no network)

```
… http-get,200,application/pdf,2,,,
… cache-hit,200,application/pdf,,,ok
```

Manifest: still `success` for the new artifact (if you created a new work item), or `skip` if your pipeline treats “already present” as no-op; both are fine as long as tokens are consistent.

### Run 3 — revalidate (304)

```
… http-get,304,application/pdf,40,,,
… http-304,304,application/pdf,,,not-modified
```

Manifest: `outcome="skip"`, `reason="not-modified"`.

---

## 8) Edge cases & invariants

**Atomicity & integrity**

* Final writes are **atomic**: write to tmp, `fsync`, `os.replace`.
* If `Content-Length` is present and **mismatch** after streaming → delete tmp, **error** (`reason="size-mismatch"`) and **no final file**.

**HEAD policy**

* HEAD is optional; useful to enforce **type/size** policies before a large GET.
* HEAD is not cached by hishel (cache is for GET); that’s OK.

**Cache storage TTL vs RFC freshness**

* `ttl_seconds` and `check_ttl_every_seconds` are **storage eviction** knobs (housekeeping).
* Freshness/validation logic is governed by the **Controller** (RFC 9111).

**Per-resolver differences**

* Keep rate-limits and retries **per resolver** (e.g., Crossref vs Unpaywall).
* Cache is **shared**, so the first hit by **any** resolver warms it for others pointing at the same URL.

**Failures**

* **Network exceptions**: retry with backoff; log `retry` attempts.
* **Too many redirects / protocol errors**: final `error` outcome; temp cleaned up.
* **Robots**: handled in `prepare` (skip early, `reason="robots"`; no GET).

---

## 9) Performance & safety notes

* **Cache hits refund tokens**: we only rate-limit true network calls.
* **Connection reuse**: one shared httpx.Client = fewer TLS handshakes.
* **Backoff spacing**: avoid thundering herds; `Retry-After` honored when present.
* **I/O chunk size**: tuned by `download.chunk_size_bytes` (1 MiB default); increase for high-bandwidth links.

---

## 10) Testing strategy (what to validate)

1. **hishel flow** with `httpx.MockTransport`:

   * Run 1 → 200 (with `ETag`) → cache populated.
   * Run 2 → **cache-hit** (`resp.extensions["from_cache"]=True`, `revalidated=False`); attempts include `cache-hit`.
   * (Optional) Toggle `always_revalidate=True` → Run 3 → **304** path; attempts include `http-304`.

2. **Rate-limit + Retry** still works with hishel:

   * Simulate `429 Retry-After: 1` then 200; assert one `retry` attempt w/ `reason="retry-after"` and ~1000ms sleep.

3. **Atomic write & size mismatch**:

   * GET 200 with `Content-Length=500000` but stream 100KB → **error**, no final file, attempt `size-mismatch`.

4. **Telemetry**:

   * Ensure attempt CSV lines match tokens & fields.
   * Ensure manifest rows include correct `outcome`/`reason` and stable keys.

---

## 11) Minimal code skeletons (reference)

### Build hishel transport & httpx client

```python
# httpx/hishel_build.py (abridged)
transport = hishel.CacheTransport(
    transport=httpx.HTTPTransport(),
    controller=hishel.Controller(always_revalidate=False, cache_private=True, cacheable_methods=["GET"]),
    storage=hishel.FileStorage(base_path=cfg.hishel.base_path, ttl=cfg.hishel.ttl_seconds, check_ttl_every=cfg.hishel.check_ttl_every_seconds),
)
client = httpx.Client(transport=transport, headers={"User-Agent": ua}, timeout=httpx.Timeout(connect=10, read=60), limits=httpx.Limits(128,64))
```

### Wrapper client with refund on cache-hit

```python
resp = self.c.request("GET", url, **kw)
from_cache = bool(resp.extensions.get("from_cache"))
revalidated = bool(resp.extensions.get("revalidated"))
if from_cache and not revalidated:
    self.bucket.refund(1.0)
```

### Telemetry mapping in `stream_candidate_payload`

```python
_emit(..., status="http-get", http_status=resp.status_code)
if from_cache and not revalidated:
    _emit(..., status="cache-hit", reason="ok")
elif revalidated and resp.status_code == 304:
    _emit(..., status="http-304", reason="not-modified")
else:
    # stream → _emit(..., status="http-200", bytes_written=..., content_length_hdr=...)
```

---

## 12) FAQ (operational intuition)

**Q: Why a shared cache across resolvers?**
A: It maximizes hit rates. If two resolvers converge on the same URL, the second benefits immediately.

**Q: When should I enable `always_revalidate`?**
A: Rarely. It forces conditional GET every time (more network). Use for content known to change without proper cache headers.

**Q: We saw a wrong content type served from cache—what now?**
A: That’s the origin’s metadata; hishel stores what the server sent. Use `prepare` + HEAD/type policies to gate unexpected types.

**Q: Do we cache POST/PUT?**
A: No. `cacheable_methods` defaults to `["GET"]`. Keep it that way unless you have a specialized API + body keying strategy.

**Q: Can we see hishel’s internal entries?**
A: File backend: yes—directory under `hishel.base_path`. We provide `cache stats/clear` for file storage. Redis/SQLite/S3 require provider tools.

---

## 13) Future extensions (kept simple today)

* **OTLP telemetry sink**: forward attempts & manifests to your observability backend.
* **Content hashing**: store sha256 in outcome meta; deduplicate files with hardlinks.
* **Shared cache across processes**: use Redis/S3 backends with modest code changes.
* **Heuristic caching**: if origin headers are weak, opt in to `allow_heuristics`.

---

## 14) Design tenets (to keep code intuitive)

1. **Explicit DI:** pass `telemetry`, `run_id`, `cfg`—no globals.
2. **Stable tokens:** attempt `status`/`reason` are a public contract.
3. **Single shared client, many policy wrappers:** reuse connections, tune behavior per resolver.
4. **Emit early and often:** every network or cache decision logs an attempt.
5. **Never leak partial files:** atomic writes, strict integrity checks.
6. **Resolvers resolve:** they propose plans; **execution fetches** (through the policy client).

---

### Glossary (hishel terms)

* **Storage**: where cache entries live (file/sqlite/redis/s3).
* **Controller**: RFC 9111 policy engine (freshness, revalidation, heuristics).
* **CacheTransport**: httpx transport that intercepts GETs and applies the controller+storage logic.
* **`resp.extensions`**: hishel metadata on responses (`from_cache`, `revalidated`, plus cache metadata under the hood).

---

If you’d like, I can also generate a **short “operator cheat-sheet”** (one page) with the most common toggles (`--no-robots`, `--resolver-order`, `hishel.*` knobs) and example CLI invocations; or produce **fixtures** (CSV + JSONL) you can load into a small dashboard to visualize cache behavior.
