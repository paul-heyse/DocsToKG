Absolutely—here’s a **deep, code-free, agent-ready** plan for the next two pillars:

---

# 3) HTTP that never surprises you (HTTPX, sync-first)

## 3.1 Objectives (what “done” means)

* **Exactly one** long-lived HTTPX `Client` powers all network I/O (planner + downloader + validators), configured once with **explicit** timeouts, pool limits, SSL, HTTP/2, and deterministic redirect behavior.
* **Zero bespoke retry logic** inside call-sites: connect-phase retry belongs to the transport; status-aware backoff (429/5xx) lives **one layer up** (Tenacity policy).
* **Uniform instrumentation** via HTTPX hooks: every request is tagged with service/host, emits timing spans, and surfaces cache result (if using Hishel alongside).
* **Redirect safety** is centralized: every hop is re-validated by your `validate_url_security` gate before following.
* **Deterministic tests**: transports are mocked (no network), with fixtures for timeouts, 429s, slow responses, and redirects.

## 3.2 Files & ownership

* `network/client.py` — Client factory + lifecycle + hook wiring (authoritative home).
* `network/policy.py` — HTTP policy constants (timeouts, pool, redirect, headers).
* `network/instrumentation.py` — request/response hooks, timers, structured log payload builders.
* `network/errors.py` — exception mapping → your domain errors.
* Tests: `tests/network/test_client_*.py`.

## 3.3 Client lifecycle

* **Construction** (once, at module init or via lazy singleton):

  * **Timeouts** (per phase): connect / read / write / pool acquire.
  * **Pool**: `max_connections`, `max_keepalive_connections`, keepalive expiry.
  * **HTTP/2**: enabled (benefits for concurrent requests to same host).
  * **Redirects**: **disabled globally**; only allow when the call-site explicitly opts in to “audit + follow”.
  * **SSL**: preload an `SSLContext` with system trust (truststore/certifi), enforce SNI, disable legacy insecure ciphers.
  * **Proxies**: `trust_env=True` by default; allow per-scheme/per-host mounts if needed.
  * **User-Agent**: single place to build final UA (`product/version (+url) run_id`), never overridden ad hoc.

* **Hooks** (attached once):

  * `on_request`: stamp correlation id & run id, attach service/host tags, record `t_start` (monotonic), normalize headers (e.g., `Accept`/`Accept-Encoding` set once).
  * `on_response`: `raise_for_status()` (centralized), compute timings (`dns?` if available, `connect`, `ttfb`, `read_total` approximations), extract cache metadata (hit/revalidate/miss) if Hishel is in play, add structured **metrics** to logs.
  * `on_error`: map HTTPX exceptions (ConnectTimeout, ReadTimeout, TooManyRedirects, RemoteProtocolError, SSLError) to your **error taxonomy** (e.g., `E_NET_CONNECT`, `E_NET_READ`, `E_NET_PROTOCOL`, `E_TLS`), with details `{service, host, method, url_redacted, elapsed_ms}`.

* **Shutdown**:

  * Provide `close()` that clients (or the app) can call at process end; guard to prevent double close.
  * Tests ensure `Client` is not left open (resource warnings off).

## 3.4 Policy defaults (explicit & documented)

* **Timeout budgets**: conservative connect (e.g., 5s), generous read (e.g., 30s for metadata probes, longer for downloads—downloads stream with separate policy).
* **Redirect policy**: never auto-follow; your “audit + follow” helper:

  1. Make request.
  2. If 3xx with `Location`, sanitize and call `validate_url_security` on the target.
  3. Follow only if safe; recurse up to max hops (configurable), recording each hop in the final result.
* **Headers**: set `Accept`, `Accept-Encoding`, and `User-Agent` once; **no per-call mutation** unless a resolver absolutely requires it. Never forward Authorization across hosts.
* **Decompression**: accept gzip/deflate/br; allow opt-out for raw bytes use-cases.

## 3.5 Retrying & backoff (division of responsibilities)

* **Transport-level**: allow a **small** number of retries for *connect* errors only (e.g., ephemeral TCP reset), with capped backoff. Never retry after bytes are sent.
* **Application-level**: Tenacity wrapper where status codes matter:

  * 429: honor `Retry-After` seconds (or sensible default) with jitter.
  * 5xx (idempotent GET/HEAD): bounded retries; never retry POST unless explicitly idempotent.
  * Abort on client-errors 4xx (except 408/409/429 when policy says so).
* Record **attempt number** and **sleep_ms** in your structured logs.

## 3.6 Streaming, memory, and file writes

* Always obtain a **streaming** response (`Client.stream` / `iter_bytes`), and write to:

  * downloader: temp file (same partition) → atomic rename (your extraction logic continues to own fsync policy).
  * planner/meta probes: **never** buffer entire bodies—read headers only unless policy demands.
* **Chunk size**: let HTTPX choose; treat your own IO buffer (for disk writes) in the extraction layer.

## 3.7 Security specifics

* **TLS**: verify on; SNI on; pin minimum TLS version (e.g., 1.2+); optionally expose cipher policy in settings for future tightening.
* **Cookies**: disabled by default (stateless client); enable only when a resolver requires session cookies.
* **Auth**: if any service needs OAuth/SigV4/etc., add a pluggable auth policy (httpx-auth pattern) in one place; never special-case in call-sites.
* **Request body size**: set a sane guard for unexpected large bodies (where applicable).

## 3.8 Instrumentation & observability

* Each request emits **one** `net.request` event with:

  * `{service, host, method, url_redacted, status, attempt, elapsed_ms, ttfb_ms, bytes_read, cache_status, http2, reused_conn}`
* The downloader emits a higher-level `download.fetch` event (file size, speed, cache), referencing the `net.request` `request_id`.
* Counters/histograms you want over time: success rate by service, p50/95 latency, error-type counts, cache hit ratio.

## 3.9 Testing plan (deterministic, no sockets)

* **MockTransport** table-tests:

  * Happy path: 200 with headers.
  * Redirect chain: 301→302→200 with a mixed host hop (ensure validator blocks unsafe).
  * Retries: connect drop → then 200; verify attempt count & backoff.
  * Timeouts: connect vs read; ensure correct mapping to error taxonomy.
  * Content-type mismatch vs alias acceptance (your RDF alias policy).
* **Contract tests**: in-process ASGI/WSGI test app for realistic 206/Range, chunked transfer, gzip/br.
* **Resource tests**: ensure client closes; file descriptors stable pre/post (on Linux, check `/proc/self/fd` count across runs).

## 3.10 Acceptance checklist

* [ ] Exactly one shared Client; redirects off globally; HTTP/2 on.
* [ ] Hook chain active; structured `net.request` logs include timings & cache hints.
* [ ] Transport retries limited to **connect**; Tenacity handles 429/5xx with jitter.
* [ ] Redirect audit validates each hop with your URL security gate.
* [ ] All downloader/probe code paths use streaming, never load whole bodies accidentally.
* [ ] Deterministic tests cover redirects, retries, timeouts, and error mapping.

---

# 4) Rate-limit correctness without bespoke code (pyrate-limiter)

## 4.1 Objectives

* One **authoritative limiter façade**; no custom token math in call-sites.
* Keys are **(service, host)** to mirror external quotas.
* Supports **single-process** (in-memory) and **multi-process** (SQLite) with the same interface.
* **Two behaviors** selectable by context:

  * **Block-until-allowed** (downloader).
  * **Fail-fast** (resolver calls where latency > success).
* **Multi-window** enforcement (e.g., 5/sec **and** 300/min) with optional **weights**.

## 4.2 Files & ownership

* `ratelimit/manager.py` — the façade + lifecycle (registry, get/acquire/release).
* `ratelimit/config.py` — parse & normalize rate specs from settings (default + per-service).
* `ratelimit/instrumentation.py` — counters & “blocked_for_ms” timing logs.
* Tests: `tests/ratelimit/test_manager_*.py`.

## 4.3 Configuration model (from your settings)

* `default_rate: RateSpec | None` — e.g., `8/second` (None = unlimited).
* `per_service: dict[str, RateSpec]` — e.g., `{"ols": "4/second", "bioportal": "2/second"}`
* `shared_dir: Path | None` — when set, use **SQLiteBucket** for cross-process coordination.
* Optional “burst control”: second window + minute window (list of `Rate` per key).

Normalization:

* Accept **semicolon string** (`"ols:4/second;bioportal:2/second"`), JSON/TOML/YAML maps; output canonical dict of `RateSpec`.
* Validate that every `RateSpec` has **unit ∈ {SECOND, MINUTE, HOUR}**; convert to RPS for diagnostics.

## 4.4 Keying & registry

* Compute **key** as: `f"{service or '_'}:{host or 'default'}"`.
* **Registry** (per process): map key → `{ limiter, rates, bucket }`.

  * `bucket` = `InMemoryBucket` (default) or `SQLiteBucket(path=/…/ratelimit.sqlite)` if `shared_dir` set.
  * `limiter` = `Limiter(bucket, …)` constructed with your rate list.
* Pre-warm default keys (for known services) on startup; lazily create others (e.g., new host scattered across a CDN CNAME).

## 4.5 Acquire semantics

Expose a tiny façade:

* `acquire(service: str|None, host: str|None, *, weight: int=1, mode: Literal["block","fail"]) -> None`

  * **block**: `try_acquire(name=key, weight=weight, max_delay=None)`; sleeps until allowed.
  * **fail**: `try_acquire(name=key, weight=weight, max_delay=0, raise_when_fail=True)`; raises `RateLimitExceeded`.
* Weighting: default `1` per request; allow future policy to weigh by **request class** (e.g., heavy ZIP downloads weight 2).

**Where to call**:

* Downloader before any GET/stream.
* Resolver HTTP calls before each attempt (wrap the Tenacity loop).

## 4.6 Multi-window & burst control

* For each key, register **list of `Rate`**:

  * Example: `Rate(5, SECOND)` **and** `Rate(300, MINUTE)`; the limiter enforces both.
* Keep burstiness bounded even if the short-window is temporarily generous.

## 4.7 Retry-After & status integration

* Do **not** mutate buckets upon 429; instead:

  * Tenacity reads `Retry-After` and **sleeps** before the *next* acquire.
  * Optionally add a **cool-down map** per key (`cooldown_until[ key ] = monotonic()+delta`) so the next acquire **waits** (block mode) or **fails fast** (fail mode) without hammering.
* Emit a `ratelimit.cooldown` event with `{key, seconds}` when cooling down.

## 4.8 Cross-process safety (SQLite mode)

* When `shared_dir` is set, all processes point at **one** SQLite DB (under that dir). This enables:

  * Correct global accounting (multiple workers).
  * Graceful restarts without burst overshoot.
* Provide `doctor rate-limits` CLI:

  * Shows configured windows per key and **current tokens** (approximate if API allows), plus bucket type (memory/sqlite).

## 4.9 Instrumentation & SLOs

Emit **one** `ratelimit.acquire` event per attempt:

* `{key, rates: [“5/s”, “300/min”], weight, mode, blocked_ms, outcome: “ok|exceeded|error”}`.
  Surface **aggregate** counters to logs/DuckDB events:
* blocked time total per key (ms), #exceeded per key, top keys by block time (identify offenders).

## 4.10 Testing matrix

* **Unit** (time-controlled):

  * Parse matrix for rate strings; invalid formats raise clean errors.
  * Block mode: acquire N+1 within 1s blocks for ~ (1 / rate) seconds; assert tolerance (use monotonic fake or slowed clock).
  * Fail mode: N+1 within window → raises immediately; no sleep.
  * Multi-window: 5/s and 300/min both enforced; orchestrate to trip minute window.
* **Cross-process** (SQLite):

  * Two subprocesses acquire against same key; combined rate respects the cap (measured via timestamps).
* **Integration**:

  * Downloader under block mode respects quotas; wall-time matches expected shape.
  * Resolver under fail-fast returns `RateLimitExceeded`; Tenacity lifts on relax.
* **Chaos**:

  * Simulate 429 with `Retry-After` = 2s; next acquire waits ≥ 2s; instrumentation shows `cooldown`.

## 4.11 Acceptance checklist

* [ ] Single façade `acquire(service, host, weight, mode)` used across downloader/resolvers; no raw pyrate calls in call-sites.
* [ ] Keys are `(service, host)`; registry pre-warms known keys; lazily adds others.
* [ ] Multi-window rates enforced; option for weights.
* [ ] Block vs fail-fast behavior selectable and test-covered.
* [ ] SQLite mode coordinates multiple processes; default in-memory for single-proc.
* [ ] 429 cooling handled by Tenacity + optional cooldown map; no ad-hoc bucket rewrites.
* [ ] Structured `ratelimit.acquire` events emitted with blocked_ms and outcome.
* [ ] CLI `doctor rate-limits` shows configured keys & windows.

---

## Implementation sequence (low risk PRs)

**PR-R1 — Limiter façade & config wiring**

* Parse settings → canonical per-service `RateSpec` map (+ default).
* Implement registry + keying; in-memory bucket.
* Add `acquire()` (block|fail) + instrumentation.

**PR-R2 — Multi-window, weights, and SQLite mode**

* Accept list of `Rate` per key; support `shared_dir` SQLite bucket.
* Add cooldown map (optional) and doctor command.

**PR-R3 — Call-site integration & tests**

* Downloader + resolvers call `acquire()` before attempts.
* Integration tests with Tenacity (429/5xx) and measured timing.

**PR-R4 — Observability polish**

* Aggregate counters; top-blockers report; CI budget tests to prevent regressions.
