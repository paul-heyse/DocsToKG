Love the call to align on one HTTP stack. Below is a concrete, repo-shaped plan to migrate **`src/DocsToKG/ContentDownload`** to **HTTPX + Hishel**, assuming you’ve already landed Tenacity in `networking` (per your commit). The goal is: one canonical HTTP client (HTTPX), one caching layer (Hishel), zero custom caching/requests code elsewhere.

---

# High-level shape (what “max consolidation” looks like)

* **One hub module** (keep using `ContentDownload/networking.py`) that exposes:

  * A **cached HTTPX client** (Hishel) for *metadata & landing pages*.
  * A **raw HTTPX client** (no cache) for *PDF artifacts*.
  * A single **`request_with_retries()`** that wraps HTTPX calls with your Tenacity policy (now catching HTTPX exceptions), so resolvers/pipeline never sleep or loop themselves.

* **Hishel** is wired **once** at the client/transport level to implement RFC-9111 caching (ETag/Last-Modified/Cache-Control/Vary), including revalidation; you delete local conditional-request logic. ([hishel.com][1])

* **HTTPX** becomes the only HTTP client: pooled connections, explicit timeouts, optional HTTP/2, mounts for proxy/routing. No `requests` left anywhere.

---

# Step-by-step migration (repo modules & what to change)

## 1) `ContentDownload/networking.py` → your **HTTPX + Hishel hub**

**a) Build two clients**

* **Cached client** for metadata/HTML:

  * Pick storage: start with **filesystem**; set `default_ttl` and `check_ttl_every`.
    (Hishel storages: FileStorage, InMemory; both are first-class.) ([hishel.com][2])
  * Wire Hishel via **CacheClient** *or* via a **CacheTransport** layered on an HTTPX `HTTPTransport` (recommended pattern; more flexible):
    `transport = hishel.CacheTransport(transport=httpx.HTTPTransport(...), storage=hishel.FileStorage(...))`
    `client = httpx.Client(transport=transport, timeout=..., limits=..., http2=True, follow_redirects=False)` ([hishel.com][2])
  * Configure cacheable methods: **GET, HEAD** (read-only). You can do this with Hishel **Controller** (cacheable_methods) or CacheOptions. ([hishel.com][3])

* **Raw client** for PDFs and other binaries:

  * Plain `httpx.Client` with the *same* pool/timeout/policy, **no Hishel**. This guarantees large binaries are never cached (no ambiguity about content-type filters).

**b) Client options (align across both)**

* Pool limits: `Limits(max_connections=..., max_keepalive_connections=..., keepalive_expiry=...)`.
* Timeouts: use an explicit `httpx.Timeout(connect=…, read=…, write=…, pool=…)`. (HTTPX defaults are strict 5s; set what matches your SLOs.)
* HTTP/2: **enable** (`http2=True`) to improve high-concurrency metadata fan-outs; keep an escape hatch to force HTTP/1.1 per host by mounting an `HTTPTransport` for that host (mounts override client defaults).
* Transport-level connect retries: you can set `HTTPTransport(retries=1–2)` to smooth **connect** errors only; keep Tenacity for status- & header-aware retries.
* Redirects: keep **off** by default; opt in per call where needed (publisher chains) to stay explicit.
* SSL: pass an `SSLContext` (not string paths) to be future-proof against HTTPX 0.28+ deprecations.

**c) Tenacity: update exception map**

You already centralized Tenacity. Swap your `requests.*` exceptions to HTTPX’s:

* Retry on: `httpx.ConnectError`, `httpx.ReadError`, `httpx.WriteError`, `httpx.TimeoutException`, and friends. Use your existing `retry_if_result` for 429/5xx + `Retry-After` parsing. (HTTPX keeps the same status on `response.status_code`.)

**d) Normalize URL just-in-time**

Right before every send, normalize via your `urls.canonical_for_request(...)` (from your prior url-normalize plan) so:

* cache keys are stable,
* request routing/mounts match consistently.

**e) Per-host cache TTL / policy**

Use Hishel’s **Controller** once on the cached client to express global policy:

* `cacheable_methods=["GET","HEAD"]`
* optional `cacheable_status_codes=[200,301,308]` (default)
* optional `allow_heuristics=True` if you want limited heuristic freshness for 200/301/308 (off by default; conservative). ([hishel.com][3])

For *exceptions* (e.g., a single provider where you want a different TTL), set **per-request** metadata via HTTPX `extensions` like `{"hishel_ttl": 3600}` from resolver call sites. These do **not** pollute HTTP headers and are the recommended control surface. ([hishel.com][4])

**f) Offline mode**

When you need “cache only”: set request header `Cache-Control: only-if-cached`. Hishel will serve from cache or return **504** without making a network call. Surface this as `--offline` in your CLI. ([hishel.com][5])

**g) Observability**

Plumb an `httpx` **event hook** to log cache metadata exposed by Hishel:

* Read `response.extensions.get("hishel_from_cache")`, `...("hishel_revalidated")`, `...("hishel_stored")` and emit counters to your run summary. ([hishel.com][4])

---

## 2) `ContentDownload/pipeline.py` → **zero HTTP code here**

* Ensure every HEAD/GET (preflight checks, resolver probes) calls **`networking.request_with_retries(client=…)`**, passing **cached client** for metadata/landing, **raw client** for artifacts. No loops/sleeps/backoffs remain here; Tenacity in `networking` orchestrates retries.

* If you were short-circuiting “recently fetched” endpoints, **delete** that logic; Hishel owns freshness/revalidation (ETag/Last-Modified/Cache-Control/Vary) and will validate or serve stale-if-allowed per RFC. ([hishel.com][1])

---

## 3) `ContentDownload/resolvers/*` → **URL factories, not HTTP clients**

* Providers should **only** build URLs/headers and call the hub (`networking.request_with_retries(...)`). Remove any direct `requests` usage and any per-resolver caching/TTL hacks.

* Where a resolver *knows* a request should have a custom TTL or must include the request body in the cache key (e.g., POST search endpoints), pass per-request **extensions**:

  * `extensions={"hishel_ttl": 3600}` to override TTL
  * `extensions={"hishel_body_key": True}` to include the body in the cache key (for POST/GraphQL-style reads) ([hishel.com][4])

* Landing-page HTML scrapes feed many duplicate probes; the cached client makes those effectively free on repeats.

---

## 4) `ContentDownload/download/*` (streaming PDFs) → **HTTPX streaming**

* Replace any `iter_content` patterns with HTTPX’s streaming API (sync): `with client.stream("GET", url) as r: ... for chunk in r.iter_bytes(): ...`
  Keep your atomic temp-file → rename logic intact; that’s orthogonal to the client.

* Resume support remains the same: send `Range` headers, expect `206`/`Content-Range`, and append to the `.part` file before atomic swap. (No caching client here.)

---

## 5) `ContentDownload/telemetry/*` → **show cache wins**

* Add counters: *from_cache*, *revalidated*, *stored*, *network* to your run summary by reading Hishel metadata on every response (`response.extensions[...]`). ([hishel.com][4])

* Add a toggle to log **only-if-cached** hits vs 504s in offline mode. ([hishel.com][5])

---

# Policy you’ll ship (sane defaults, minimal knobs)

* **HTTPX (both clients)**

  * `Timeout(connect=3, read=15, write=15, pool=3)`
  * `Limits(max_connections=200, max_keepalive_connections=50, keepalive_expiry=10)`
  * `http2=True` (measure; mount HTTP/1.1 for any misbehaving host)
  * `transport=HTTPTransport(retries=2)` to retry **connect** errors only (leave status-code logic to Tenacity).

* **Hishel (cached client only)**

  * Storage: **FileStorage**, `default_ttl=48h`, `check_ttl_every=600s`
  * Controller: `cacheable_methods=["GET","HEAD"]`; leave status codes at default (200/301/308); keep `allow_heuristics=False` unless you decide otherwise. ([hishel.com][3])
  * Per-request overrides via **extensions** (e.g., `hishel_ttl` for Crossref hot paths). ([hishel.com][4])
  * **No binary caching** by construction (artifacts go through the raw client)

* **Tenacity** (already in place)

  * Keep your current status-aware + `Retry-After` wait strategy; just swap exception types to HTTPX.

---

# What you can delete after this

* Any use of `requests` (clients, adapters, hooks).
* All **manual conditional request** helpers (ETag/Last-Modified, 304 logic) — Hishel handles RFC-9111 and revalidation. ([hishel.com][1])
* All per-resolver “TTL guards” and “recently-fetched” caches; **one** caching layer now.
* Any ad-hoc pool logic; HTTPX `Client` holds pooling/limits.

---

# Gotchas & how we avoid them

* **HTTP/2 surprises**: some APIs throttle differently under multiplexing. Keep a small **denylist** that mounts a plain `HTTPTransport` (HTTP/1.1) for those hosts. (Mounts route by scheme/host/port.)

* **Artifact URLs with signed query** (S3, CDN): those go through the **raw client**, so no cache meddling, and you still get Tenacity/streaming/atomic writes.

* **POST/GraphQL metadata**: if any provider uses POST for “reads”, set `extensions={"hishel_body_key": True}` so request bodies participate in the cache key. ([hishel.com][4])

* **Offline runs**: `Cache-Control: only-if-cached` guarantees **no network**; expect 504 when an entry is missing. ([hishel.com][5])

---

# Testing & rollout

1. **Introduce clients** in `networking.py`. Keep your Tenacity wrapper; switch exception classes.
2. **Flip resolvers/pipeline** to call the new hub (cached for metadata, raw for artifacts).
3. **Verify** on a small corpus:

   * Cache hit rate (from Hishel metadata)
   * Revalidation rates (should see 304s and `hishel_revalidated=True`) ([hishel.com][4])
   * No PDF caching (ensure all artifact requests use the raw client)
4. **Mount HTTP/1.1** for any host that dislikes HTTP/2.

---

# Why this matches your “max consolidation” goal

* HTTP semantics (pooling, timeouts, redirects, retries at connect-layer) are **100% HTTPX** now.
* HTTP caching (freshness, validators, Vary, conditional revalidation, offline mode, per-request TTL) is **100% Hishel**; no home-grown code remains. ([hishel.com][1])
* Your Tenacity wrapper stays the single place that expresses **status-aware retry/backoff + Retry-After** (exactly where it belongs with HTTPX).

If you want, I can turn this into a precise “edit list” per file (functions to touch/remove) or sketch the minimal client factory shapes for `networking.py` so your agents can implement it directly.

[1]: https://hishel.com/?utm_source=chatgpt.com "Hishel"
[2]: https://hishel.com/advanced/storages/?utm_source=chatgpt.com "Storages - Hishel"
[3]: https://hishel.com/advanced/controllers/?utm_source=chatgpt.com "Controllers - Hishel"
[4]: https://hishel.com/beta/metadata/?utm_source=chatgpt.com "Metadata - Hishel"
[5]: https://hishel.com/advanced/http_headers/?utm_source=chatgpt.com "HTTP Headers - Hishel"
