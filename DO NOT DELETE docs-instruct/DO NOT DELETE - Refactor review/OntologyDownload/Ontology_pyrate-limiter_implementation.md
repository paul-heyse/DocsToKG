Awesome — here’s a **surgical, implementation-ready plan** to replace your home-grown buckets/registry with **pyrate-limiter** across `src/DocsToKG/OntologyDownload`, keeping behavior stable for callers and minimizing bespoke code. I’ll anchor the steps to the exact places in your tree where the changes land.

---

# North-star (what “done” means)

* **All throttling goes through pyrate-limiter** `Limiter` objects and library buckets (in-memory, SQLite, or multiprocess). No custom bucket math, no bespoke file-locked JSON state.
* A single, **central limiter manager** serves per-key quotas where the key = `"{service or _}:{host or default}"`, so your existing per-host/per-service semantics remain intact. Call sites continue to “just call a throttle” before network I/O.
* The public surface exposed by `OntologyDownload` remains stable (same CLI and `download_stream(...)` entry) — only the **internal rate-limit layer** changes in `ratelimit.py`, plus small call-site adjustments in `net.py` and `resolvers.py`.

Why pyrate-limiter: it provides Rates, Buckets (in-memory, Redis, SQLite, Postgres, multiprocess), a Limiter façade with sync/async acquire, decorators, background leaking, delay/exception semantics, and proper multi-window limits — all out of the box.

---

# Inventory: what you have today (and where we’ll touch it)

* `net.py` currently **aliases** your local bucket type and consumes it inside `download_stream(...)`:

  * acquires a bucket keyed by `(service, host)` and calls `bucket.consume()` before the GET.
* `resolvers.py` similarly **pulls a bucket** for resolver API calls (license lookups, OLS/BioPortal, etc.) and uses it inside the retry wrapper.
* You ship a dedicated **`ratelimit.py` module** with `TokenBucket`, `SharedTokenBucket` (filesystem-backed via `fcntl/msvcrt`), and a `RateLimiterRegistry` that returns buckets keyed by `(service, host)`, plus `apply_retry_after(...)`.
* Config already parses human strings like `"5/second"` → **RPS** via `parse_rate_limit_to_rps`, and supports per-service overrides — we’ll **reuse this**.

---

# Design choices (explicit)

1. **Backend selection (no new knobs required)**

   * Default: **InMemoryBucket** (single process).
   * If `DownloadConfiguration.shared_rate_limit_dir` is set today (used by your `SharedTokenBucket`), we create a **SQLiteBucket** at `<shared_rate_limit_dir>/ratelimit.sqlite` for cross-process persistence. This is the clean, library-supported analog of your file-locked JSON state.
   * If you later run true multi-process throttling without persistence, swap to **MultiprocessBucket**; the manager interface below won’t change.
2. **Keys and windows**

   * Key format: `service_key = service or "_"`, `host_key = host or "default"`, combined as `f"{service_key}:{host_key}"`. Matches your current registry qualification.
   * **Rates**: derive from config exactly as today:

     * Base per-host limit → fallback Rate(s).
     * Per-service override (if present) wins.
   * We keep a **single window** that mirrors the user’s unit (second | minute | hour), and optionally add a second guard window if you’re worried about bursts (e.g., `N/sec` + `60N/min`). Both are native in pyrate-limiter.
3. **Acquire semantics**

   * Maintain current behavior — **block until capacity** (your `consume()` slept). We’ll configure the limiter with **blocking acquires** and no global `max_delay`, so it waits as needed; callers don’t change their control flow. (If you want hard caps later, we’ll add `max_delay` in config.)
4. **Server back-pressure (`Retry-After`)**

   * You currently nudge bucket state via `apply_retry_after(...)`; we will **retire** this hook (less bespoke) and instead **defer to the HTTP retry policy** to honor Retry-After sleeps between attempts. If you want to keep a hook, we’ll convert it to “sleep before next acquire” rather than manipulating buckets directly.

---

# Step-by-step plan (small, safe PRs)

## PR-1 — Add dependency and feature flag (no runtime change)

1. Add `pyrate-limiter` to project dependencies (runtime).
2. Introduce a config toggle in code (not CLI) `rate_limiter = "pyrate"` (default), so you can roll back to the legacy path for a release if needed.

**Acceptance:** build green; tests untouched.

---

## PR-2 — Replace `ratelimit.py` internals with a **thin manager on pyrate-limiter** (keep module name/exports)

Goal: keep import surfaces stable while deleting the home-grown bucket logic.

1. **Keep the module name** `DocsToKG.OntologyDownload.ratelimit` and its public exports (`get_bucket`, optionally `apply_retry_after`, `reset`), but **remove**:

   * `TokenBucket`, `SharedTokenBucket` implementations,
   * bespoke `RateLimiterRegistry` state & JSON file locking.
2. Implement a **LimiterManager** inside `ratelimit.py` (pyrate-limiter based):

   * **Keying**: compute `service_key, host_key` exactly like today.
   * **Rate resolution**: reuse your `DownloadConfiguration.rate_limit_per_second()` and `parse_service_rate_limit(service)` to compute the **effective** RPS for a given `(service, host)`.
   * **Rates → pyrate objects**: translate the user’s text unit (“per second”, “per minute”, “per hour”) into a **`Rate(limit, Duration.UNIT)`**. If the string is already normalized to RPS only, emit `Rate(ceil(rps), Duration.SECOND)`. (Multi-window optional.)
   * **Backend**:

     * If `shared_rate_limit_dir` is set, **use SQLiteBucket** at `<dir>/ratelimit.sqlite`; else **use InMemoryBucket**. (This preserves your current “shared across processes via filesystem” intent without any custom file locks.)
   * **Limiter construction**: for each unique key, create (or reuse) a **`Limiter(bucket)`**. Keep them in a small internal cache; expose a `dispose_all()` used by tests to clear state.
3. **Define the new public API in `ratelimit.py`** (matching existing call patterns):

   * `get_bucket(http_config, service, host)` → returns a **simple gate object** with a single method `consume(tokens: float=1.0)`. Internally, it delegates to `Limiter.try_acquire(name=f"{service_key}:{host_key}", weight=ceil(tokens))` with **blocking acquires**. (This preserves your call sites that currently do `bucket = get_bucket(...); bucket.consume()` without changing their control flow.)
   * `apply_retry_after(...)`: **no-op** or deprecated log; we’ll remove call sites in PR-4.
   * `reset()` → clears the limiter cache; used in tests today.

**Acceptance:**

* `grep` shows **no** remaining definitions of custom `TokenBucket`/`SharedTokenBucket` in `ratelimit.py`.
* Unit tests compile with the same import surfaces.

---

## PR-3 — Switch **call sites** to the new ratelimit façade (small diffs)

1. **`net.py` → `download_stream(...)`**

   * Keep the existing lines that compute `service` and `host`.
   * Continue to call `bucket = ratelimit.get_bucket(http_config=..., host=..., service=...)` and then `bucket.consume()`. No further changes are needed — it now uses pyrate-limiter under the hood.
2. **`resolvers.py` (retry wrapper)**

   * Keep resolving a `bucket` from `get_bucket(...)` for `service` API calls and continue consuming before the network attempt. Your current injection already routes through `ratelimit.get_bucket(...)`.

**Acceptance:** all tests green; no behavior regressions; throttling still happens at the same call boundaries.

---

## PR-4 — Remove legacy hooks and dead code (clean sweep)

1. **Delete / deprecate** `apply_retry_after(...)` call sites in `net.py`/`resolvers.py` (if any remain) and rely on your HTTP retry policy to honor `Retry-After` sleeps. Remove the function from `ratelimit.py` in the next release after a deprecation window.
2. Purge stale imports/aliases, e.g., `TokenBucket = ratelimit.TokenBucket` in `net.py`. (It’s now an internal adapter, not a type callers need.)

**Acceptance:** no references to `TokenBucket`/`SharedTokenBucket`/JSON state remain anywhere in the package.

---

## PR-5 — Tests & observability

1. **Unit tests**:

   * Reuse existing `parse_rate_limit_to_rps` tests (keep as is).
   * Add tests for **per-service overrides** and **per-host defaults** resolving to different keys (`"_:host"` vs `"service:host"`) and thus different buckets/limiters.
   * Add tests for **blocking acquire** (the gate waits rather than raises) using a tiny rate and measuring elapsed time.
   * If `shared_rate_limit_dir` is configured in tests, assert that **two processes** (or two Python interpreters) respect the shared limit via SQLite.
2. **Metrics/logging**:

   * At the ratelimit façade, log the **key**, **rate window** chosen, and whether an acquire **blocked** (with the delay). This mirrors what your buckets implicitly did by sleeping.

**Acceptance:** tests cover success path, full bucket path, and multi-key routing; logs show key/rate decisions.

---

# Migration details & exact mappings (for the agent)

**Where to delete bespoke logic**

* `DocsToKG/OntologyDownload/ratelimit.py`: remove classes `TokenBucket`, `SharedTokenBucket`, and the JSON-file locking code; replace the registry with a pyrate-limiter powered manager as above.
* Ensure `REGISTRY`-style global state goes away; pyrate-limiter limiters sit in a **small in-module cache** keyed exactly as your `_qualify(...)` does today.

**Where to keep current behavior**

* `net.py` acquisition point in `download_stream(...)`: **unchanged** call-site (`get_bucket(...).consume()`), now backed by pyrate-limiter.
* `resolvers.py` retry path: **unchanged** semantics — acquire token(s) **before** invoking each resolver HTTP call.

**How to calculate rates**

* Use `DownloadConfiguration.rate_limit_per_second()` and `parse_service_rate_limit(service)` (already in your config) to compute RPS; convert to `Rate(...)` with the **unit the user specified** (sec/min/hour). Keep the text parser you just consolidated.

**Backend defaults**

* If `shared_rate_limit_dir` is set: create an **SQLiteBucket** under that directory (instead of your JSON files). This yields persistence and safe multi-process coordination **without** custom file locks.
* Else: **InMemoryBucket**.

**Delays vs. exceptions**

* Your current `consume()` **sleeps**; we therefore configure the limiter for **blocking** acquires and omit `max_delay`, preserving the “wait rather than fail” behavior. (If later you want 429-style behavior, set `raise_when_fail=True` and/or a finite `max_delay` and translate overflows to `DownloadFailure`.)

**429 Retry-After**

* Remove `ratelimit.apply_retry_after(...)` plumbing. Instead, let your **retry/backoff** layer incorporate `Retry-After` delays between attempts. This reduces custom state mutation against the limiter and aligns with HTTP semantics.

---

# Definition of Done (quick checks)

* [ ] `grep -R "class TokenBucket" src/DocsToKG/OntologyDownload` → **0 matches** (only a shim adapter may remain inside `ratelimit.py`).
* [ ] `grep -R "SharedTokenBucket" ...` → **0 matches**.
* [ ] `grep -R "apply_retry_after(" ...` → **0 matches** (or only a deprecated wrapper for one release).
* [ ] `download_stream(...)` still calls a bucket/gate’s `consume()` before network I/O, but that gate is powered by **pyrate-limiter**.
* [ ] `resolvers` still throttles resolver HTTP calls through the same façade.
* [ ] If `shared_rate_limit_dir` is configured, a SQLite database file exists and shared limits are honored across processes.

---

# Rollout guidance

* Ship PR-2 and PR-3 behind the internal `rate_limiter="pyrate"` switch for one release; dogfood under CI and a full `plan → pull → validate` run. Then delete the switch and legacy code in PR-4.
* Keep your `parse_rate_limit_to_rps` docs/tests visible so users continue to configure `"N/second"`, `"M/minute"`, or `"K/hour"` strings — your CLI and doctor output already exercise these helpers.

---

If you’d like, I can turn this into PR-ready diffs next (ratelimit manager + small call-site edits).
