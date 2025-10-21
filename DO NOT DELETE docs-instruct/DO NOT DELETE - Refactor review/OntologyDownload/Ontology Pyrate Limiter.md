Absolutely — here’s a **best-in-class, code-free, agent-ready implementation plan** to roll out **pyrate-limiter** as the authoritative rate-limit layer in `src/DocsToKG/OntologyDownload`, replace the legacy TokenBucket code, and integrate it cleanly with HTTPX/Tenacity and your observability.

---

# 0) North-star & Definition of Done

**Goal:** All throttling in OntologyDownload is enforced by **pyrate-limiter** with **per-(service,host) keys**, sane defaults, multi-window limits, and **multi-process safety** (SQLite). Call-sites acquire a token **once per attempt** with a single façade; no custom bucket math remains.

**Done when:**

* No references to `TokenBucket`, `SharedTokenBucket`, or `apply_retry_after` remain.
* A single façade `ratelimit.acquire(service, host, mode="block|fail", weight=1)` is used by downloader & resolvers.
* Defaults: **8/sec** **and** **300/min** (two windows) if no per-service override.
* When `settings.ratelimit.shared_dir` is set, processes coordinate via **SQLiteBucket**; otherwise **InMemoryBucket**.
* 429 handling is done by Tenacity (`Retry-After` sleeps) — the limiter **is not** mutated ad-hoc.
* Structured telemetry `ratelimit.acquire` (blocked_ms, outcome) is emitted per attempt.
* Unit, component, and **multi-process** tests verify correctness and timing (no overshoot).

---

# 1) Files & Surfaces

```
src/DocsToKG/OntologyDownload/
  ratelimit/
    __init__.py
    config.py             # parse & normalize RateSpecs (default + per-service)
    manager.py            # registry, acquire() façade, pid-bind, cooldown
    instrumentation.py    # emit ratelimit.acquire / cooldown events
  settings.py             # add RateLimitSettings (strict Pydantic)
  net/ or downloader/     # call acquire() at attempt boundaries
tests/ontology_download/ratelimit/
  test_parse_rates.py
  test_acquire_singleproc.py
  test_acquire_block_vs_fail.py
  test_acquire_multiwindow.py
  test_multiprocess_sqlite.py
  test_retry_after_interplay.py
  test_prewarm_and_lazy_keys.py
```

**Removal targets** (delete or deprecate):

```
src/DocsToKG/OntologyDownload/**/ratelimit.py   # legacy TokenBucket/SharedTokenBucket/apply_retry_after
```

---

# 2) Settings (Pydantic v2; strict types)

In `settings.py` add:

```text
RateLimitSettings:
  default: str | None           # e.g., "8/second" (None means unlimited)
  per_service: dict[str, str]   # e.g., {"ols":"4/second","bioportal":"2/second"}
  shared_dir: Path | None       # if set => SQLiteBucket file under this dir
  engine: Literal["pyrate"] = "pyrate"
```

**Normalization rules:**

* Accept strings **or** maps; allow semicolon shorthand `"ols:4/second;bioportal:2/second"`.
* Parse `"N/second|minute|hour"` to a canonical `RateSpec(limit:int, unit:Enum)` and compute **RPS** for diagnostics.
* If `default is None`: limiter returns **no throttle** unless a per-service override exists.
* If `default is set`: **always** add a second minute-window guard (300/min default) unless overridden.

---

# 3) RateSpec parsing & canonicalization (`ratelimit/config.py`)

APIs:

* `parse_rate(s: str) -> RateSpec`           # raises with friendly message on bad format
* `parse_map(s_or_map) -> dict[str, RateSpec]`
* `default_windows(default_spec: RateSpec | None) -> list[RateSpec]`

  * If given `"8/second"`, return `[Rate(8, SECOND), Rate(300, MINUTE)]` unless settings explicitly add minute window.

**TypeAdapter hints** (Pydantic v2): support env → dict and semicolon string → dict; lower-case units; validate >0.

---

# 4) Manager design (`ratelimit/manager.py`)

### 4.1 Registry & keying

* **Key**: `f"{service or '_'}:{host or 'default'}"`.
* **Registry** (per process): `dict[key, Entry]` where `Entry = { limiter, bucket, rates, pid }`.

**PID binding**: Store `pid` in each `Entry`. On first `acquire()` after a fork, if `os.getpid() != entry.pid`, **rebuild** the entry (new limiter/bucket) to avoid sharing sockets/handles across processes.

### 4.2 Bucket selection

* If `settings.ratelimit.shared_dir` is set:

  * `db = shared_dir / "ratelimit.sqlite"` → **SQLiteBucket** (global across processes).
* Else:

  * **InMemoryBucket** (fast; single process).

### 4.3 Limiter creation & windows

* For a given key, assemble rates:

  * If service has an override → start with its spec.
  * Else, use `settings.ratelimit.default` spec (if any).
  * Always expand to **multi-window**: e.g., `[(8, SECOND), (300, MINUTE)]` unless explicitly configured.
* Create `Limiter(bucket, [Rate(...)...])`.

### 4.4 Acquire façade

```text
def acquire(service: str | None, host: str | None,
            mode: Literal["block","fail"]="block",
            weight: int = 1) -> None
```

* Build key; **get or create** Entry (pre-warm known services at init; see §4.6).
* If `mode=="block"`:

  * `try_acquire(name=key, weight=weight, max_delay=None)` → may **sleep** until allowed.
* If `mode=="fail"`:

  * `try_acquire(name=key, weight=weight, max_delay=0, raise_when_fail=True)` → raise `RateLimitExceeded` immediately if not allowed.

Emit one `ratelimit.acquire` event with `{key, rates:["8/s","300/min"], weight, mode, blocked_ms, outcome:"ok|exceeded|error"}`.

### 4.5 Retry-After “cooldown”

* **Do not** modify the bucket on 429.
* Instead, expose a **helper**:

```text
def cooldown_for(key: str, seconds: float) -> None
```

* Store a `cooldown_until[key] = monotonic()+seconds`.
* At next `acquire()`:

  * If `monotonic() < cooldown_until[key]`:

    * `block` mode: sleep `(until-now)` once **before** trying to acquire (so the limiter window doesn't accumulate virtual waits).
    * `fail` mode: raise `RateLimitExceeded` immediately with hint “cooldown”.

Emit a `ratelimit.cooldown` event when a cooldown is registered.

**Integration**: Tenacity status policy (429) calls `cooldown_for(key, retry_after_seconds)` then sleeps; subsequent `acquire()` should **not double-sleep** (blocked_ms ≈ 0).

### 4.6 Pre-warm & lazy creation

* On first `init_manager(settings)` call:

  * For every `service` in `per_service` keys, create an Entry with `host="default"`.
* For new hosts (e.g., CDNs), entries are created **lazily** on first `acquire()`.

---

# 5) Observability (`ratelimit/instrumentation.py`)

* `emit_acquire(key, rates, weight, mode, blocked_ms, outcome, error=None)`
* `emit_cooldown(key, seconds)`
* All events include `run_id` and `config_hash` via your existing emitter context.

---

# 6) Call-site integration (downloader, resolvers)

**Downloader (`download_stream` or equivalent):**

* Before each HTTP attempt:

  * Derive `service` (resolver or “_” if unknown) and `host` from the normalized URL.
  * `ratelimit.acquire(service, host, mode="block", weight=1)`.
* Let **Tenacity** handle 429/5xx:

  * On 429, parse `Retry-After`, call `ratelimit.cooldown_for(key, seconds)`, and **sleep**; next `acquire()` should not block again.

**Resolvers / Metadata probes:**

* Use `mode="fail"` if you’d rather **bail quickly** than block.
* Map a `RateLimitExceeded` to a clean `DownloadFailure`/retry (or straight fail) depending on your policy.

**Important:** remove all references to `apply_retry_after(...)`; replace with `cooldown_for` + Tenacity sleep.

---

# 7) Deprecation & deletion of legacy code

* **Delete** `TokenBucket`, `SharedTokenBucket`, any registry logic, JSON file manipulation, and `apply_retry_after`.
* Provide a **compat shim** only if other packages in the monorepo import the old module; otherwise delete outright.
* Grep guard in CI:

  ```bash
  grep -R "TokenBucket\|SharedTokenBucket\|apply_retry_after" src/DocsToKG/OntologyDownload && exit 1 || true
  ```

---

# 8) Tests

## 8.1 Unit (parsing & config)

* `test_parse_rates.py`:

  * `"8/second"`, `"2/minute"`, `"120/hour"` → canonical RateSpec.
  * Semicolon map `"ols:4/second;bioportal:2/second"` → dict.
  * Invalid formats give friendly error messages.
  * Default expands to `[8/s, 300/min]` when minute window unspecified.

## 8.2 Component (single process)

* `test_acquire_singleproc.py`:

  * N+1 acquires in a second → last blocks; measure `blocked_ms ~ 1000/N_limit`.
  * `fail` mode raises immediately.
* `test_acquire_multiwindow.py`:

  * Configure `[8/s, 12/min]`; drive >12 total requests in a minute → latter ones blocked/fail while 1-sec window might allow bursts earlier.

## 8.3 Multi-process (SQLiteBucket)

* `test_multiprocess_sqlite.py`:

  * With `shared_dir` set, spawn 2–4 processes; each issues requests against same key, verify the **combined** rate ≈ specified windows (within jitter tolerance).
  * Ensure no overshoot; blocked_ms accumulate across processes.

## 8.4 Retry-After interplay

* `test_retry_after_interplay.py`:

  * Simulate 429 with `Retry-After: 1`. Tenacity sleeps 1s; next `acquire(block)` has `blocked_ms ≈ 0`; total wall-time ~1s, not ~2s (no double wait).

## 8.5 Pre-warm & lazy keys

* Assert pre-warmed services exist on init; new hosts create entries lazily at first use.

**Performance sanity**: micro-bench acquire fail-fast path < 0.1 ms; block-path overhead negligible beyond sleep time.

---

# 9) CI & Rollout

* **PR 1 — Scaffolding**

  * Add `settings.RateLimitSettings` and `ratelimit/config.py|manager.py|instrumentation.py`.
  * Keep legacy code present but unused; add the new façade; write unit tests for parsing.

* **PR 2 — Call-site integration**

  * Replace downloader & resolver calls: `ratelimit.acquire(...)`.
  * Remove `apply_retry_after` references; add Tenacity handler to call `cooldown_for`.
  * Emit `ratelimit.acquire` events; component tests pass.

* **PR 3 — Legacy removal**

  * Delete `TokenBucket`/`SharedTokenBucket` modules; add CI grep guard.
  * Multi-process SQLite tests pass.

* **PR 4 — Docs & CLI doctor** (optional, but useful)

  * Add `ontofetch rl doctor` to display current keys → windows and recent blocked_ms tallies from events.
  * Update README/Settings doc with examples.

---

# 10) Acceptance checklist (copy into PR)

* [ ] `ratelimit.acquire(service, host, mode, weight)` façade used by all call-sites; **no** legacy buckets remain.
* [ ] Defaults **8/s** and **300/min** active when not overridden; per-service overrides parsed & enforced.
* [ ] `shared_dir` enables SQLiteBucket; multi-process test shows combined rate obeyed (no overshoot).
* [ ] `Retry-After` handled by Tenacity + `cooldown_for`; **no** double waits; limiter state not mutated.
* [ ] Telemetry: one `ratelimit.acquire` event per attempt with `{key, rates, blocked_ms, outcome}`; optional `ratelimit.cooldown`.
* [ ] Performance sanity: acquire fail-fast path < 0.1 ms; blocked_ms ≈ expected sleeps.
* [ ] Docs/settings updated; examples for per-service overrides.
* [ ] CI grep guard ensures TokenBucket/SharedTokenBucket/apply_retry_after are gone.

---

## Quick architecture sketch (how this slots in)

```
Settings (RateLimitSettings)
   │  default + per_service + shared_dir
   ▼
[Limiter Manager]
   • key = "service:host"
   • default windows: [8/s, 300/min]
   • per-service overrides
   • bucket: InMemory or SQLite (shared_dir)
   • pid-bind & fork-aware rebuild
   • cooldown_for(key, seconds)
   ▼
acquire(service,host,mode,weight)
   │ emits ratelimit.acquire {blocked_ms,outcome}
   ▼
HTTP attempt (HTTPX/Tenacity)
   • on 429: cooldown_for(key, Retry-After) + sleep
   • retry → acquire (blocked_ms ~ 0)
```

This gives you a **robust, observable, multi-process-safe** rate-limit layer with minimal code in call-sites and zero bespoke math — fully aligned with the rest of the “best-in-class” stack you’re building.
