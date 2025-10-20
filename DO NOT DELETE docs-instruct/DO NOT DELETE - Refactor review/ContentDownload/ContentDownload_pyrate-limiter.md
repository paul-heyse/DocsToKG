Perfect—here’s a **repo-shaped, agent-ready implementation plan** to introduce **pyrate-limiter** into `src/DocsToKG/ContentDownload`, designed for **maximal consolidation** (one limiter layer for the whole pipeline) and clean integration with your **HTTPX + Hishel + Tenacity** stack.

---

# What “good” looks like after this refactor

* **One rate-limit layer** in the **networking hub**; *no* sleeps, semaphores, or bespoke token buckets anywhere else.
* Limits are **per host** and optionally **per “role”** (metadata, landing, artifact), with **multi-window** support (e.g., per-second and per-minute) and correct ordering.
* Limiting triggers **only on actual network hits**, not cache hits (by placing the limiter **beneath** Hishel in the transport stack).
* Behavior is **configurable** (CLI/env) and **observable** (counters, logs).
* Works in **single-process** now; can switch to **multiprocess**/**SQLite**/**Redis** backends later without changing call sites.

---

## Implementation Notes

- `src/DocsToKG/ContentDownload/ratelimit.py` now owns the limiter runtime: `LimiterManager`, `RateLimitedTransport`, policy cloning/validation, backend factories, and telemetry helpers. `networking.request_with_retries` delegates role selection (`request.extensions["role"]`) and Tenacity treats `RateLimitError` as non-retryable.
- `httpx_transport._create_client_unlocked()` wraps the inner `HTTPTransport` with `RateLimitedTransport` before handing it to Hishel’s `CacheTransport`, so cache hits never acquire tokens. The stack is persisted for both real transports and test transports (`httpx.MockTransport`).
- CLI surfaces `--rate`, `--rate-mode`, `--rate-max-delay`, `--rate-backend`, and `--rate-disable`. Environment variables (`DOCSTOKG_RATE*`, `DOCSTOKG_RATE_DISABLED`) mirror these switches for automation. Legacy throttling flags are translated into artifact role policies and emit warnings.
- Telemetry: `docs_network_meta.rate_limiter_*` accompanies each attempt, manifest metrics include acquire/block counters, `summary.emit_console_summary` reports per-host waits/blocks, and `DownloadRun` logs the resolved backend/policy table during startup.
- Default policies were normalised (`Rate(2/sec), Rate(120/min)` for artifacts, etc.) to satisfy `pyrate_limiter.utils.validate_rate_list`. Backends ship with in-memory defaults, while SQLite/Redis/Postgres initialisers surface friendly errors when optional dependencies are missing.
- Test coverage includes cache-hit bypass (`only-if-cached` flows), multi-window waits, role-specific wait budgets across sync/streaming paths, CLI override/parsing (including the rollback switch), legacy flag mappings, and backend smoke tests. Redis/Postgres tests are guarded by env flags to avoid accidental network requirements.

---

## A. Where the limiter lives (module boundaries)

### 1) `src/DocsToKG/ContentDownload/networking.py` (the canonical hub)

Add all limiter wiring here—**and only here**. Everything that generates HTTP calls (resolvers, pipeline, download) should already delegate to networking.

* **Add a RatePolicy registry** (module-level) keyed by **host** and **role**:

  * Example roles: `"metadata"` for API/JSON/HTML, `"landing"` for publisher pages, `"artifact"` for PDFs.
  * Values: **ordered** lists of `Rate(...)` (e.g., `Rate(10, SECOND)`, `Rate(1000, HOUR)`), plus a **limiter mode** (raise vs. bounded wait), and **max_delay** per role.
  * Capture known politeness constraints here (e.g., very slow for arXiv metadata, moderate for Crossref/OpenAlex). You can tune later without touching code.

* **Add a Limiter cache** keyed by `(host, role)` → returns a constructed `Limiter(...)` with the chosen bucket backend.

* **Expose a tiny “role selector”** the rest of the code can use: `role="metadata" | "landing" | "artifact"`.

### 2) New helper module (optional): `src/DocsToKG/ContentDownload/ratelimit.py`

If you want to keep `networking.py` lean, move the following into `ratelimit.py` and import from `networking`:

* `RatePolicy` dataclass & registry
* backend selection & limiter factory
* the **transport wrapper** described below

---

## B. Transport wiring (so we only rate-limit real network calls)

Your stack today: **HTTPX Client** → **Hishel CacheTransport** → **HTTP transport**.

To avoid “charging” cache hits against quotas, insert the limiter **under** Hishel:

```
HTTPX Client
  └─ Hishel CacheTransport    (handles RFC cache, revalidation, offline)
      └─ RateLimitedTransport (NEW)  ← acquires tokens only on cache miss/revalidate
          └─ HTTPTransport    (sends bytes on the wire)
```

* **RateLimitedTransport** responsibilities:

  * Determine `host` (from request URL) and **role** (from request `extensions["role"]`—you’ll set this at call time).
  * Look up `(host, role)` in the **RatePolicy** registry to get the right `Limiter`.
  * **Acquire** before performing the network call:

    * For `"metadata"`/`"landing"`, allow a **short** `max_delay` (smoothing).
    * For `"artifact"`, allow a **longer** `max_delay` (prevents bursts of big PDFs).
    * Decide **method weight** (e.g., `GET=1`, `HEAD=0` → skip acquire for HEAD unless a provider counts it; set via policy).
  * If capacity is unavailable **beyond** `max_delay`, raise immediately. Do **not** let Tenacity loop on rate-limit exceptions (see interplay, below).

This placement guarantees:

* Cache hits **bypass** the limiter (because Hishel never calls the inner transport).
* Only real network traffic is paced—exactly what upstream quotas care about.

---

## C. Backend choice (now and later)

Start simple but keep the interface swappable:

* **Now (single process)**: `InMemoryBucket` (fast, zero deps).
* **Later (multi-process on one host)**: `MultiprocessBucket` or `SQLiteBucket` (file-lock mode).
* **Distributed (many hosts)**: `RedisBucket` or `PostgresBucket`.

Encapsulate this choice in one factory (e.g., `get_bucket_for(host, role)`), selected by config/env (e.g., `DOCSTOKG_RATE_BACKEND=memory|mp|sqlite|redis|postgres`).

(Why: different backends, same `Limiter` façade—no call-site changes needed.)

---

## D. Policy model (what to encode, where)

**One mapping** in `networking.py` (or `ratelimit.py`) drives everything:

* `HOST_POLICIES = { "api.crossref.org": RolePolicy(...), "api.openalex.org": RolePolicy(...), "export.arxiv.org": RolePolicy(...), ... }`

  * Each `RolePolicy` holds:

    * `metadata_rates: List[Rate]` (multi-window; **ordered** correctly)
    * `landing_rates: List[Rate]`
    * `artifact_rates: List[Rate]`
    * `metadata_max_delay_ms`, `landing_max_delay_ms`, `artifact_max_delay_ms`
    * `count_head: bool` (default False—most providers don’t charge HEAD)
    * Optional **burst weight** for artifact GET (often unnecessary; start at 1)

* **Validation**: At startup, **validate** every multi-rate list ordering (the library exposes a validator), and log a single error if any list is malformed (fail fast).

* **Overrides**: Accept CLI/env overrides:

  * `--rate api.crossref.org=50/s,2000/m` (parsed into two Rates)
  * `--rate export.arxiv.org=1/3s`
  * Mode & max_delay overrides: `--rate-mode api.crossref.org=wait:1000ms` or `raise`

* **Per-request overrides** (rare): Allow resolvers to pass `extensions={"rate_override": {"max_delay_ms": 250}}` if a special path needs stricter behavior without changing global policy.

---

## E. Call-site minimalism (what other modules must do)

* **Every** HTTP call (HEAD or GET) already flows through `networking`. Great—keep it that way.
* At each call site, set the **role** explicitly:

  * API/JSON/HTML metadata: `role="metadata"`
  * Publisher landing pages: `role="landing"`
  * Direct PDF GET: `role="artifact"`
* That’s it. No sleeps; no local throttles. The hub does the rest.

*(If any resolver or pipeline path still constructs a client or touches sockets directly, delete/route it through `networking` now.)*

---

## F. Interplay with Hishel and Tenacity (critical sequencing)

**Order of operations for each request:**

1. **Hishel** inspects the cache:

   * If it can serve from **cache**, your `RateLimitedTransport` is never called → **no limiter tokens consumed**.
   * On **revalidation** or cache miss, it forwards to the inner transport.

2. **RateLimitedTransport** acquires from the appropriate `Limiter` (host+role).

   * If allowed (or after waiting ≤ `max_delay`), it forwards to the wire.
   * If not allowed within `max_delay`, it raises **immediately** (no double waiting).

3. **Tenacity** wraps the *whole* send (including any 429/503) and applies backoff/`Retry-After`.

   * **Do not** treat `BucketFullException`/`LimiterDelayException` as retryable—those are *your own pacing decisions*, not transient network faults.
   * Keep 429/503 *network responses* retryable so Tenacity respects `Retry-After` and waits appropriately.

This division of labor prevents “retry storms” when you’re simply over your budget and keeps backoff logic scoped to **server-driven** signals.

---

## G. Telemetry & ops (what you must expose)

Add counters and structured logs in `networking`:

* `rate_limited_acquires_total` (by host+role)
* `rate_limited_delays_ms_sum` (and a p95/p99 in summary)
* `rate_limited_blocked_total` (exceeded max_delay)
* Existing HTTP metrics: combine with **Hishel** fields (`from_cache`, `revalidated`, `stored`) to show how much traffic was avoided by cache vs. pacing.

Emit a one-line **policy table** at start (host, role, rates, mode, max_delay) so runs are auditable.

---

## H. Error handling & UX decisions

* **Metadata & landing**: prefer **bounded waits** (e.g., 50–250 ms) to smooth bursts; otherwise **raise** immediately so callers can degrade gracefully.
* **Artifact**: allow **longer waits** (e.g., up to a few seconds) to avoid waves of 429s and oversaturating publishers/CDNs.
* **HEAD**: default **not counted**; flip per host if a provider treats HEAD as chargeable.

When blocking is disallowed or time budget exceeded:

* Raise a clear exception tagged with host/role and the **computed next-allowed-in** (ms), so your run summary can recommend policy tweaks.

---

## I. Concrete work items (checklist for the agent)

1. **Dependency**: add `pyrate-limiter` to your environment.

2. **Networking hub**:

   * Create a **RatePolicy** registry (`HOST_POLICIES`) with sane defaults for your known hosts; include role-specific rates and max_delay.
   * Add a **Limiter cache** keyed `(host, role)`; pick **InMemoryBucket** backend for now.
   * Implement **RateLimitedTransport**:

     * Reads request URL → `host`
     * Reads `request.extensions["role"]` (default `"metadata"` if absent)
     * Selects limiter
     * Applies **acquire** with role-specific `max_delay`
     * Skips acquire for HEAD if `count_head=False`
     * Forwards to inner transport
   * Build clients:

     * **metadata/landing client**: `Client(transport=Hishel(CacheTransport(transport=RateLimitedTransport(HTTPTransport(...)))))`
     * **artifact client**: `Client(transport=RateLimitedTransport(HTTPTransport(...)))`
   * Update your **Tenacity** wrapper to **not** retry on limiter exceptions; keep retry for `httpx` I/O exceptions and 429/503.

3. **Call sites**:

   * Ensure **every** HTTP call goes through networking and sets a **role**:

     * Resolvers → metadata/landing
     * Download strategy → artifact
     * Head-precheck → metadata
   * Remove any sleeps/semaphores/rate code elsewhere.

4. **Config & CLI**:

   * Add env/CLI flags to override host policies and the backend type.
   * Add `--offline` support using Hishel (`only-if-cached`), which naturally bypasses the limiter.

5. **Telemetry**:

   * Add per-host+role counters & delay histograms.
   * Include Hishel cache stats on each response.

6. **Validation**:

   * At startup, **validate rate lists** (ordering) and log the effective policy.
   * Unit tests:

     * Prove cached responses don’t consume tokens.
     * Prove multi-window limits enforce both windows.
     * Prove max_delay behavior (short vs. long) for metadata vs. artifact.
     * Prove limiter exceptions are **not** retried by Tenacity, but 429/`Retry-After` is.

---

## Implementation snapshot (2025-03-18)

* `src/DocsToKG/ContentDownload/ratelimit.py` owns `RolePolicy`, `LimiterManager`, and the `RateLimitedTransport`; limiter instances are cached per `(host, role)` and the transport always sits beneath Hishel to avoid charging cache hits.
* `src/DocsToKG/ContentDownload/httpx_transport._create_client_unlocked` wraps the shared `HTTPTransport` in `RateLimitedTransport` before handing it to `CacheTransport`, so every HTTP client that comes out of `get_http_client()` is centrally throttled.
* CLI and env overrides are parsed in `src/DocsToKG/ContentDownload/args.resolve_config()`, which clone base policies, merge `--rate*` / `DOCSTOKG_RATE*` inputs, translate legacy `--domain-token-bucket` / `--domain-min-interval` flags, and surface the active `rate_policies` + `rate_backend` on `ResolvedConfig`. `DownloadRun.run()` logs the policy table on startup.
* Telemetry surfaces limiter fields end-to-end: `telemetry.RunTelemetry` and `AttemptRecord` include `rate_limiter_*` metadata, `summary.emit_console_summary()` and `statistics.DownloadStatistics` aggregate acquire/wait/block counters, and manifests (`manifest.metrics.json`) expose backend/role snapshots.
* Tests covering the contract live in `tests/content_download/test_httpx_networking.py` (cache hits), `test_rate_control.py` (multi-window, role delays, backend smoke), and `test_args_config.py` (CLI overrides, legacy flag mapping, startup logging). Use them as regression guides when tuning policies.
* Operations runbooks highlight backend selection and migration toggles (`--rate-disable` / `DOCSTOKG_RATE_DISABLED`). Default backend remains in-memory; use SQLite for shared runners, `multiprocess` for forked workers, or Redis/Postgres when a distributed store is required.

---

## J. Initial policy (safe defaults you can ship today)

* **metadata** (APIs like OpenAlex/Crossref/etc.): two windows, e.g., `Rate(10, SECOND) + Rate(5_000, HOUR)`; `max_delay=100–250 ms`, `count_head=False`.
* **landing** (publisher HTML): conservative per-second rate; short `max_delay` (pages can change; cache will do most of the work).
* **artifact** (PDFs): stricter per-second rate, optional per-minute cap; `max_delay=1–3 s`, `count_head=False`.

*(Treat this as operational starting points—real numbers live in config and evolve as you observe 429s and latency.)*

---

## Why this approach

* It uses the **Limiter façade** and **bucket backends** as intended (no re-invented wheels), supports **multi-window** limits, **blocking** vs **raise** modes, and **per-call** overrides—all recommended primitives from the library.
* The **transport placement** ensures you pace **only** network calls (cache hits don’t waste tokens), which is crucial now that you’ve standardized on **Hishel** for HTTP caching.
* The **hub-only** implementation keeps resolvers/pipeline clean and aligns with your “max consolidation” rule across the codebase.

If you want, I can follow up with a one-pager of **policy strings → Rate lists** (parser rules) and the **exact locations** in `networking.py` to insert the transport & clients so the agent can cut a tidy PR in one pass.
