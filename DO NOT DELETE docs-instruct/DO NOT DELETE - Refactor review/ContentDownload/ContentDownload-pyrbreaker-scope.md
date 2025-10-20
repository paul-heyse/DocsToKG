Absolutely—here’s a **repo-shaped, best-in-class plan** to introduce **pybreaker** as the **per-host (and optional per-resolver) circuit breaker** at the networking layer of `src/DocsToKG/ContentDownload`. I’ll lay out (1) where it lives, (2) how it cooperates with your HTTPX+Hishel+Tenacity+limiter stack, (3) config & telemetry, (4) the migration away from the in-house breaker, and (5) advanced options.

Where I reference current code, I cite your repo so a junior dev can locate the exact places to touch.

---

# 0) Why we’re doing this (and what we replace)

You currently ship a **home-grown `CircuitBreaker`** plus ad-hoc per-host breaker wiring in the pipeline (e.g., `_ensure_host_breaker`, `_host_breaker_allows`, `_update_breakers`). We’ll **remove those** and consolidate behavior in the networking hub so **every HTTP call** passes through **one** breaker layer before Tenacity tries the request.

You’ve also finished a Tenacity migration that centralizes retries in `networking`—perfect place to seat the breaker.

---

# 1) Where pybreaker lives (single choke-point)

**Authoritative home:** `src/DocsToKG/ContentDownload/networking.py` (your HTTP hub with head precheck, retries, conditional requests). All HTTP already flows through here (e.g., `head_precheck()`/`request_with_retries()`), and this module is where callers should *stop* thinking about backoff/health.

> Goal: **Network flow = URL normalize → (breaker pre-check) → rate limiter → Tenacity → HTTPX/Hishel → breaker post-update.**
> (The pipeline/resolvers should not implement breaker logic themselves.)

---

# 2) Breaker topology (what we track)

* **Per-host breaker** (required): key = lowercased punycoded host (after your canonicalization step).
* **Per-resolver breaker** (optional): key = resolver name (useful when a specific resolver’s logic fails systematically while others work).

You already have the notion of host & resolver breakers; we’ll **move them to networking** and back them with pybreaker.

---

# 3) BreakerRegistry (the one object you add)

Add a tiny `BreakerRegistry` in `networking.py` that:

* Holds **pybreaker breakers** keyed by `(host)` and `(resolver)`; exposes `get_host_breaker(host)`, `get_resolver_breaker(name)`.
* Encodes **policy** (fail thresholds, reset timeouts) from a single mapping (see §6).
* Implements **pre-flight**: `allow(host, resolver, role) -> (allowed: bool, cooldown_remaining: float)`. If not allowed, raise a small `BreakerOpenError` (custom).
* Implements **post-update**:

  * `on_success(host, resolver)` → breaker success bookkeeping.
  * `on_failure(host, resolver, *, status:int|None, exception:Exception|None, retry_after_s:float|None)` → breaker failure bookkeeping (see §4.2 for classification).

> This **replaces** the pipeline helpers `_host_breaker_allows`, `_ensure_host_breaker`, `_update_breakers` (delete them there).

---

# 4) How breakers cooperate with Tenacity, Hishel, and the limiter

## 4.1 Order of operations (every request)

1. **URL canonicalize** (you planned this in `urls.py`).
2. **Breaker pre-check** (host and resolver): if **open**, raise `BreakerOpenError` immediately.
3. **Rate limiter** acquire (only if breaker allowed).
4. **Hishel cache**: if it returns from cache, **skip** breaker accounting entirely (no network happened).
5. **Tenacity** sends via HTTPX.
6. **Breaker post-update** (see below), then return the response.

> Result: the breaker **prevents** wasted retries when an origin is unhealthy, and Hishel ensures we don’t “spend” breaker or limiter capacity on cache hits.

## 4.2 What counts as **failure** vs **success**

Define a **failure classifier** in `networking.py`:

* **Failures** increment breakers:

  * HTTP: **429, 500, 502, 503, 504** (transient classes), and optionally **408**.
  * Exceptions: HTTPX connect/read/write/timeout errors.
* **Successes** reset breakers:

  * 2xx/3xx responses and your **“safe terminal results”** (`SKIPPED`, `CACHED`, etc.) when they represent *healthy* origin behavior. (You already handle classification; reuse it to avoid jitter.)

**Edge decisions** (do *not* count as breaker failures): 401/403 (auth), 404 (not found), 410 (gone), 451 (legal), robots/blocked. Those are not “origin unhealthy”.

## 4.3 “Retry-After aware” cooldown (best practice)

When the response is **429 or 503** and carries a `Retry-After`, open the **host breaker** for **that duration** (bounded by a policy ceiling). If `Retry-After` is absent, rely on the breaker’s configured `reset_timeout`. (You already parse `Retry-After` for Tenacity; reuse it here.)

> Because pybreaker’s `reset_timeout` is static, keep a tiny **cooldown override** map `{host -> monotonic_until}` inside `BreakerRegistry`. Pre-flight checks the override *before* asking pybreaker. If override is in the future, we treat it as “open”.

## 4.4 Tenacity integration (don’t thrash)

* **Do not retry** on `BreakerOpenError` (configure your Tenacity predicate to exclude it).
* Keep retry for **network exceptions** and **retryable statuses** (your Tenacity is already set up for both).

---

# 5) Wiring at the networking callsite (what changes in practice)

* In the **HTTP hub** wrapper that issues requests (your Tenacity-backed `request_with_retries()` / transport call), insert:

  * `registry.allow(host, resolver, role)` before limiter acquire.
  * After the Tenacity attempt returns:

    * If **exception path**: call `registry.on_failure(host, resolver, exception=e, status=None, retry_after_s=None)` and re-raise.
    * If **response path**:

      * classify success/failure as in §4.2,
      * parse `Retry-After` when `status in {429,503}`,
      * call `registry.on_success(...)` or `registry.on_failure(..., status=…, retry_after_s=…)` accordingly,
      * then return the response.

This keeps **all breaker decisions in one place** and leaves resolvers/pipeline oblivious.

---

# 6) Configuration & policy (one map, clear knobs)

Add a **`CircuitPolicy` map** in `networking.py` (or a small `breakers.py`) with per-host (and default) settings:

* `fail_max` (aka threshold)
* `reset_timeout_s` (static fallback)
* `retry_after_cap_s` (upper bound when honoring Retry-After)
* `window` (optional: consecutive vs rolling—see Advanced §A)
* `roles`: optional overrides per role (`metadata`, `landing`, `artifact`)
* `resolver_overrides`: optional per-resolver thresholds

Populate **known origins** (Crossref, OpenAlex, arXiv, PMC/EuropePMC, Wayback) with conservative defaults, then allow CLI/env overrides:

* `--breaker api.crossref.org=fail_max:5,reset:60,retry_after_cap:900`
* `--breaker export.arxiv.org=fail_max:3,reset:120`
* `--breaker-resolver landing_page=fail_max:4,reset:45`

You already carry breaker thresholds/cooldown in config near token buckets; move those keys under this new policy surface so everything reads from one place.

---

# 7) Telemetry (what to record, where)

Extend your existing telemetry (or Wayback’s pattern) with **breaker lifecycle**:

* **Per request (enriched on the response event you already emit)**:

  * `breaker_host_state` (`closed|open|half_open`), `breaker_host_cooldown_ms`
  * `breaker_resolver_state` (same)
  * `breaker_recorded` (`success|failure|none`)
* **State transitions** (via pybreaker **listener**):

  * `breaker_transition` rows: `{scope: "host|resolver", key, from_state, to_state, fail_count, reset_timeout_s, ts}`

You already record run summaries and per-attempt logs; add these fields alongside the networking counters so we can compute “breaker opens per host”, “time saved by break”, etc. (Your telemetry module is already in active use and recently improved.)

---

# 8) Tests (must-have cases)

1. **Open on consecutive 5xx/429**: send N failing responses and assert (a) breaker opens, (b) `allow()` blocks with non-zero remaining. (You already had property tests for the home-grown breaker; reuse the structure.)
2. **Honor Retry-After**: send 429 with `Retry-After=4`; assert host breaker blocks for ~4s (bounded by cap).
3. **Success closes**: once open period elapses, a success should reset to closed.
4. **Exclusions**: 401/403/404 should not flip the breaker.
5. **Cache bypass**: when Hishel serves from cache, breaker counters **do not** change.
6. **Tenacity**: ensure `BreakerOpenError` is **not retried**.
7. **Role separation**: artifact failures do not affect metadata breaker policy when configured per role.

---

# 9) What to delete / migrate

* Remove the **custom `CircuitBreaker` class** and its tests after the pybreaker path is stable (keep the TokenBucket work until your limiter land).
* Delete pipeline breaker helpers (`_host_breaker_allows`, `_ensure_host_breaker`, `_update_breakers`) and replace with networking calls.
* Scrub any **inline sleeps** or “cooldown” hacks in resolvers/pipeline—**breaker + limiter** now own timing.

---

# 10) Operational posture (SLOs & dashboards)

* **SLOs**: p95 “**breaker block time**” < reset_timeout (obvious), “**breaker open rate**” by host should trend down after rate-limiter rollout; “**success-after-open**” should be > 50% (useful signal the cooldown worked).
* **Dashboards**: “opens by host/hour”, “time in open by host”, “requests avoided while open”, “top reasons for failure (status vs exception)”.

---

# 11) Advanced / “best possible” enhancements

**A) Rolling window breaker (beyond consecutive)**
pybreaker trips on **consecutive** failures. For origins that fail intermittently, add a **rolling-window wrapper**: keep a small deque of timestamps/flags per host and flip the host into “manual open” (your cooldown override) if **X failures within Y seconds**. (Pre-flight checks this before pybreaker.)

**B) Half-open sampling & jitter**
On close expiry, allow **K trial calls** (e.g., K=1 for metadata, K=2 for artifact) and **stagger** first attempts with a small jitter to avoid herd effects at second boundaries.

**C) Distributed state (multi-process or multi-host)**
If you plan many worker processes or multiple hosts:

* Keep **pybreaker** for logic, but store state in a **shared backend** (Redis/SQLite). If you don’t adopt a pybreaker storage, mirror the **cooldown override** map in SQLite (one row per host) with a very short TTL; pre-flight reads it to honor opens across processes.

**D) Dynamic reset_timeout**
If `Retry-After` regularly exceeds the static `reset_timeout`, surface a metric and **auto-bump** the configured reset for that host (bounded by a ceiling) to reduce flapping.

**E) Breaker-aware routing**
If you host multiple resolvers that can fetch the same content from different origins, a **breaker-open** on one host can proactively trigger alternate resolver priority (outside scope of networking, but your resolver registry supports ordering).

**F) CLI controls**
Add `docstokg breaker open/close/show` commands for ops: force-open a host for N seconds, close immediately, list state with cooldown remaining, etc.

---

# 12) Step-by-step task list (for the PR)

1. **Add** `pybreaker` dependency.
2. **Create** `BreakerRegistry` in `networking.py`; add `BreakerOpenError`.
3. **Insert** breaker **pre-check** & **post-update** in the HTTP hub wrapper (the same place you added Tenacity).
4. **Wire** roles: pass `role="metadata|landing|artifact"` in request `extensions` (same mechanism you used for limiter).
5. **Move/translate policy** (thresholds/cooldowns) from pipeline config to the new circuit policy map in networking.
6. **Delete** pipeline breaker helpers; update imports.
7. **Telemetry**: add breaker fields to the per-request telemetry and listener-based transition events. (You already have telemetry scaffolding and sinks.)
8. **Tests**: add the cases in §8; keep your existing head_precheck tests green.
9. **Docs**: update the API doc page for networking to list the new breaker behavior and config knobs.

---

## TL;DR

* **One** pybreaker-backed **BreakerRegistry** inside `networking.py`, keyed by host (+ optional resolver), with **Retry-After-aware** cooldowns.
* **Before send**: check breaker; **after send**: update breaker based on status/exception.
* **Don’t retry** `BreakerOpenError`; do keep Tenacity for network faults & retryable statuses.
* **Delete** custom breaker & pipeline breaker code; keep all logic in the hub.
* Ship with **telemetry & ops controls** so you can see opens, time saved, and tune policies quickly.

If you want, I can follow up with a 1-page config schema (YAML/CLI mapping) for breaker policies and a tiny `BreakerListener` skeleton that emits the state transitions to your telemetry sink.
