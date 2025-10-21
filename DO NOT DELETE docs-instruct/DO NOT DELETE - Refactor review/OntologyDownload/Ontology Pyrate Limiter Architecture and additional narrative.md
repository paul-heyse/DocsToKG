Absolutely — here’s a **complete architecture + narrative** for the new **pyrate-limiter** layer. It’s written to be friendly for newer engineers while still precise enough for senior reviewers.

---

# 1) The Big Picture (what and why)

Our downloader and resolvers make lots of HTTP calls. If we hit a provider too fast, we get **429 (Too Many Requests)** or risk being blocked. We need a central “traffic light” that:

* knows each **service** (e.g., OLS, BioPortal) and **host** (e.g., `ebi.ac.uk`),
* enforces limits (e.g., **8 per second** and **300 per minute**),
* coordinates **multiple processes** (so 4 workers don’t accidentally do 4× the allowed rate),
* plays nicely with our **retry policy** (Tenacity, `Retry-After`), and
* emits **structured telemetry** so we can see what’s happening.

**pyrate-limiter** gives us a tiny, battle-tested engine for exactly this. We’ll wrap it with a small façade so the rest of our code just calls `acquire()`.

---

# 2) Architecture (components and interactions)

```
┌──────────────────────────────────────────────────────────────┐
│                  Traced Settings (Pydantic)                  │
│  RateLimitSettings:                                          │
│    default="8/second" + implicit "300/minute"                │
│    per_service={"ols":"4/second", ...}                       │
│    shared_dir="/tmp/ontofetch_rl" (→ SQLiteBucket)           │
└───────────────┬──────────────────────────────────────────────┘
                │ normalized RateSpecs (+ RPS for logs)
                ▼
┌──────────────────────────────────────────────────────────────┐
│                 Limiter Manager (our module)                 │
│  - Registry: { "service:host" → Limiter+Bucket+PID }         │
│  - Keying: "ols:ebi.ac.uk", "_:default" for unknowns         │
│  - Bucket: InMemory (single proc) OR SQLite (multi-proc)     │
│  - Windows: [8/s, 300/min] unless overridden                 │
│  - cooldown_for(key, seconds) (from 429 Retry-After)         │
│  - acquire(service,host, mode="block|fail", weight=1)        │
│    → emits ratelimit.acquire {blocked_ms, outcome}           │
└───────────────┬──────────────────────────────────────────────┘
                │ acquire once per HTTP attempt
                ▼
┌──────────────────────────────────────────────────────────────┐
│           HTTP Attempt (HTTPX + Tenacity policy)             │
│  - request()                                                 │
│  - if 429:                                                    │
│      parse Retry-After → cooldown_for(key, seconds)          │
│      Tenacity sleeps (1s etc), retries                       │
│  - no double-wait: next acquire() should block ~0ms          │
└──────────────────────────────────────────────────────────────┘
```

**Key points:**

* **Settings** define who gets what quota.
* The **manager** is the only place that knows about buckets and windows.
* Call-sites **do not** know anything except “call `acquire()` before I/O”.
* **Tenacity** owns waiting for `Retry-After` (not the limiter).
* **Telemetry** tells us which keys blocked, for how long, and why.

---

# 3) Roles & responsibilities (one-liners)

* **Settings**: truth for your limits (default + per service + shared_dir).
* **Limiter Manager**: registry, keying, PID-aware rebuilding (after forks), cooldown handling, event emission.
* **Downloader/Resolvers**: call `acquire()` before each attempt; on 429, call `cooldown_for()` and let Tenacity sleep.
* **Tenacity**: reads `Retry-After` and sleeps; does not mutate limiter state.
* **Events**: `ratelimit.acquire` per attempt; optional `ratelimit.cooldown` when registered.

---

# 4) Data contracts (so everyone speaks the same language)

**RateSpec**

* Text form: `"N/second" | "N/minute" | "N/hour"`
* Canonical: `{limit: int, unit: SECOND|MINUTE|HOUR, rps: float}`

**Key**

* String: `"{service or '_'}:{host or 'default'}"`

  * e.g., `"ols:ebi.ac.uk"`, `"_:default"`

**Events**

* `ratelimit.acquire`
  `{ key, rates: ["8/s", "300/min"], weight, mode: "block|fail", blocked_ms, outcome: "ok|exceeded|error" }`
* `ratelimit.cooldown`
  `{ key, seconds }`

All events also include our standard envelope `{ts, run_id, config_hash, context{…}}`.

---

# 5) Lifecycle (step-by-step)

## 5.1 Initialization

1. **Load Settings** → `RateLimitSettings`.
2. **Pre-warm** known services (`per_service` keys) as entries with host=`"default"`.
3. If `shared_dir` is present, create `ratelimit.sqlite` there and use **SQLiteBucket**; otherwise **InMemoryBucket**.

> Why pre-warm? It avoids an initial “first acquire builds limiter” delay in hot paths. Unknown hosts (e.g., CDN) are **lazy-created** on the first `acquire()`.

## 5.2 Acquire before a request

* Call-site determines `service` and `host` (from the URL we are about to fetch).
* It calls:

```text
acquire(service="ols", host="ebi.ac.uk", mode="block", weight=1)
```

* Manager:

  * Computes key `"ols:ebi.ac.uk"`.
  * Ensures entry exists, **rebuilding** if PID changed (fork safety).
  * Calls pyrate-limiter:

    * `mode="block"` → waits as needed; `blocked_ms ~ 0` if allowed immediately.
    * `mode="fail"` → raises `RateLimitExceeded` immediately if not allowed.
  * Emits `ratelimit.acquire`.

## 5.3 When 429 happens (Retry-After)

* HTTP handler parses `Retry-After: 2` and calls:

```text
cooldown_for(key="ols:ebi.ac.uk", seconds=2.0)
```

* Tenacity sleeps ~2s and retries.
* Next `acquire()` sees the **cooldown** and returns almost immediately (no double waiting) because the sleep already happened.
* If we had multiple workers, **all** see the cooldown behavior in their own flow (Tenacity governs each attempt process).

---

# 6) Defaults, overrides, and weights

* **Default windows:**
  If no per-service override is given, we enforce both **8/sec** **and** **300/min**. This keeps bursts reasonable **and** prevents minute-level drifts.

* **Per-service override:**
  `per_service = {"ols":"4/second","bioportal":"2/second"}`
  We still recommend adding a minute window unless you’re certain the provider doesn’t require it.

* **Weights:**
  `weight=1` per normal request. You can choose to make very heavy operations `weight=2` later (e.g., extremely large downloads) — most teams start with `1`.

---

# 7) Modes — when to block vs fail

* **Downloader**: `mode="block"` — we prefer to wait a little than error a long download.
* **Resolvers / metadata lookups**: `mode="fail"` — fail fast; the pipeline can skip and move on.

---

# 8) Multi-process safety (SQLiteBucket)

**Why we need it:** If we run 4 worker processes, each process’ in-memory bucket won’t know about the others — they would collectively make **4×** the allowed requests.

**Solution:** set `settings.ratelimit.shared_dir` → all processes use **one** SQLite file (`ratelimit.sqlite`). That database tracks the shared windows correctly.

**How it looks:**

* Manager opens a SQLite bucket pointing at that file.
* All `acquire()` calls in all processes consult the same database.
* The combined rate across all workers respects the windows.

---

# 9) Interaction with retry policy (Tenacity)

**Do this:**

* When you see 429, **call** `cooldown_for(key, seconds)` **then sleep** via Tenacity, then retry.

**Do not do this:**

* Do **not** mutate bucket state directly for Retry-After.
* Do **not** sleep in both the limiter and Tenacity (that’s a **double wait**).

**Why:** Limiter enforces a steady rhythm; Tenacity handles provider hints about when to try again. Separation keeps logic simple and testable.

---

# 10) Examples (concrete flows)

### Example A — Single process, OLS, block mode

1. First 8 requests in 1s: all pass, `blocked_ms≈0`.
2. The 9th in the same second: `blocked_ms≈125ms` (until the next second).
3. Over a minute, if we exceed 300 total, we start blocking until the minute rolls over.

### Example B — Multi-process (4 workers), shared_dir set

* 4 processes each making requests to `"bioportal:data.bioontology.org"`.
* Combined rate ≤ the windows (e.g., 2/s and 120/min if configured), regardless of process count.

### Example C — 429 with Retry-After: 2

* Attempt #1 → 429
  `cooldown_for("ols:ebi.ac.uk", 2)`; Tenacity sleeps 2 seconds.
* Attempt #2 → `acquire()` returns quickly; new request → success (200).
* Events show: two `ratelimit.acquire` (first ok, second blocked_ms≈0), one `ratelimit.cooldown`.

---

# 11) Troubleshooting (common pitfalls, quick fixes)

* **“We still double-wait on 429.”**
  You’re probably sleeping in Tenacity and the limiter is blocking too. Ensure the limiter’s `acquire()` checks cooldown and returns quickly after Tenacity’s sleep.

* **“We still got blocked by the provider.”**
  Add/adjust a **minute window** (e.g., 120/min) — many providers throttle longer horizons. Also confirm your **key** is correct (service + **host**).

* **“My test runs overshoot the rate with 3 processes.”**
  Confirm `shared_dir` is configured and **all** processes point to the same SQLite file. Ensure you’re not forking after manager init without PID-aware rebuild (the manager should rebuild entries if PID changed).

* **“Acquire is slow even when allowed.”**
  In fail-fast mode it should be sub-millisecond. If not, you might be repeatedly constructing limiters. Ensure you **cache** them in the registry and pre-warm services on init.

---

# 12) Testing strategy (how we prove it works)

* **Single process**: N+1 acquires in 1s → last blocks ~expected ms; fail mode raises instantly.
* **Multi-window**: configure 8/s + 12/min; show that hitting >12 total in a minute blocks, even if per-second allows bursts earlier.
* **Multi-process (SQLite)**: 2–4 processes together don’t exceed the total windows.
* **429 interplay**: after `Retry-After`, the next `acquire()` should show `blocked_ms≈0` (no double sleep).
* **Pre-warm & lazy**: known services present at init; new `host`s created lazily on first acquire.

---

# 13) Anti-patterns (what not to do)

* **Per-request new limiter**: creates overhead and defeats shared state; always use the manager’s registry.
* **Mutating limiter on Retry-After**: confuses semantics; keep timing sleeps in Tenacity.
* **Blocking resolvers**: prefer `mode="fail"` so you don’t stall an entire resolver queue.
* **Relying on second-only window**: add a minute window to keep longer-term burstiness under control.

---

# 14) “How we ship this” (PR narrative your juniors can follow)

1. **PR-1: Add new module + settings.**

   * `RateLimitSettings`, parse helpers, `manager.acquire()` and `cooldown_for()`; unit tests for parsing.

2. **PR-2: Wire call-sites.**

   * Downloader and resolvers call `acquire()` before attempts.
   * Tenacity 429 handler calls `cooldown_for()` + sleeps.
   * Emit `ratelimit.acquire` events.

3. **PR-3: Remove legacy** (`TokenBucket`, `SharedTokenBucket`, `apply_retry_after`).

   * Multi-process tests pass (SQLiteBucket).

4. **PR-4: Docs & runbook.**

   * Configuration examples, “block vs fail”, “429 flow”, “multi-process setup”, and the troubleshooting above.

---

# 15) One-page mental model (for quick onboarding)

> **“Before I hit the network, I must ask the traffic light.”**

* **Who am I?** (service)
  **Where am I going?** (host)
  → **Key** = `service:host`

* **Ask**: `acquire(key)`

  * Green: go now (blocked_ms ≈ 0)
  * Yellow: wait (blocked_ms > 0)
  * Red (fail mode): raise immediately → handle gracefully

* **If I get a 429**: the provider tells me when to try again (Retry-After).

  * I **tell** the light: `cooldown_for(key, seconds)`
  * I **sleep** for that time (Tenacity)
  * I ask the light again — it should let me through without another wait.

* **If we have multiple cars (processes)**: put the light control box in one place (SQLite). Everyone shares the same schedule.

---

If you’d like, I can also drop in **ready-to-paste doc pages** (e.g., `docs/architecture/rate-limits.md`) that reuse this narrative and include copy-ready examples for `.env` and CLI snippets.
