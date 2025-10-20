Awesome — here’s a **repo-shaped, junior-dev-friendly** plan to make your **pybreaker circuit breaker** truly “best in class,” with exact file touch-points, function signatures to fill, and the order to implement. It assumes you already added the skeletons we discussed:

* `breakers.py` (Registry + policies + in-memory cooldowns),
* `breakers_loader.py` (YAML/env/CLI → `BreakerConfig`),
* `networking_breaker_listener.py` (telemetry listener),
* HTTP hub calls `registry.allow(...)` pre-flight and `registry.on_success/on_failure(...)` post-response.

Below we’ll (1) finish `breakers.py`, (2) add cross-process cooldown stores, (3) wire the listener & networking, (4) add CLI controls, (5) telemetry, (6) tests, and (7) rollout & acceptance.

---

## 0) Overview (what you’ll have when done)

* A **single** `BreakerRegistry` in the networking hub that:

  * **Pre-flight**: `allow(host, role, resolver)` denies immediately if open.
  * **Post-update**: `on_success` / `on_failure(status|exc, retry_after_s?)` keeps state in sync.
  * **Retry-After aware**: a `429/503` with header opens a **cooldown override** until that time (capped).
  * **Rolling-window manual open**: if ≥ N failures occurred within W seconds, set an override cooldown (cooldown store).
  * **Half-open sampling**: allow only `trial_calls` per role after cooldown (with a tiny jitter budget).
  * **Cross-process sharing**: optional SQLite/Redis-backed **cooldown store** so all workers respect opens.
* A **BreakerListener** that emits `state_change / success / failure / before_call` to telemetry.
* **CLI**: `docstokg breaker show|open|close` for ops.
* **Telemetry & summary**: opens/hour by host; success-after-open; time saved by break; % opens due to Retry-After vs rolling window.

---

## 1) Finish `breakers.py` (Registry internals)

Open `src/DocsToKG/ContentDownload/breakers.py` and fill the TODOs. Key decisions:

### 1.1 Pre-flight `allow()`

**Algorithm (pseudocode):**

```python
now = self._now()
host = host.lower()

# 1) Cooldown overrides (shared store)
until = self.cooldowns.get_until(host)          # returns monotonic deadline if using in-proc; see §2 for cross-proc conversion
if until and until > now:
    raise BreakerOpenError(f"host={host} cooldown_remaining_ms={int((until-now)*1000)}")

# 2) Rolling window manual open (best effort)
if self._should_manual_open(host, now):
    # _should_manual_open sets cooldown if threshold met
    until2 = self.cooldowns.get_until(host)
    if until2 and until2 > now:
        raise BreakerOpenError(...)

# 3) Host breaker state
h_cb = self._get_or_create_host_breaker(host)
if h_cb.current_state == pybreaker.STATE_OPEN:
    raise BreakerOpenError(...)

# 4) Resolver breaker (optional)
if resolver:
    r_cb = self._get_or_create_resolver_breaker(resolver)
    if r_cb.current_state == pybreaker.STATE_OPEN:
        raise BreakerOpenError(...)

# 5) Half-open probes per role
self._enforce_half_open_probe_limit(host, role, h_cb, now)
```

**Half-open jitter:** you don’t sleep in the registry. Instead:

* If `h_cb.current_state == HALF_OPEN`, compute a **suggested** jitter `0..jitter_ms` and attach it to a tiny return object or to `request.extensions["breaker_jitter_ms"]`. The networking hub can `sleep(jitter/1000)` just before sending. *If you want to keep `allow()` void, skip the return and let the hub ask `registry.current_state(...)` and randomize within `jitter_ms` when half-open.*

### 1.2 Post-update `on_failure()` / `on_success()`

* **Failure path**:

  * `h_cb.call_failed()`, and if resolver provided, `r_cb.call_failed()`.
  * `self._record_failure_for_rolling(host, now)` (adds timestamp to deque, prunes by window).
  * **Retry-After aware**:

    * If `status in {429,503}` and `retry_after_s > 0`:

      * duration = `min(retry_after_s, policy.retry_after_cap_s)`,
      * `self.cooldowns.set_until(host, now + duration, reason="retry-after")`.
* **Success path**:

  * `h_cb.call_success()` (and resolver cb success),
  * `self.cooldowns.clear(host)`,
  * reset half-open probe counter for `(host, role)`.

### 1.3 Rolling window manual open

* Done in `self._record_failure_for_rolling(host, now)` and checked in `_should_manual_open`.
* If deque length ≥ `threshold_failures` inside `window_s`, set `cooldowns.set_until(host, now + cooldown_s, reason="rolling-window")`.

### 1.4 Helper: `_remaining_cooldown_ms(cb, now)`

* Approximate using `cb._state_storage.opened_at` (if present) and `cb.reset_timeout`. If not available, 0. (You already sketched this.)

### 1.5 Host/role policies

* `self._policy_for_host(host)` → base `BreakerPolicy` (host override or defaults).
* `self._role_policy(base, role)` → `BreakerRolePolicy` with `trial_calls` and optional overrides of `fail_max/reset_timeout_s`.

> **Important:** the host breaker itself remains host-wide (one instance); role overrides are enforced in **half-open probe accounting**, not by creating per-role breakers.

---

## 2) Cross-process cooldown stores (shared state)

Add two backends in `breakers.py` (or in `breakers_store.py` if you prefer):

### 2.1 `SQLiteCooldownStore`

**Use case:** multiple local processes share a POSIX filesystem.

* **Schema (create on first use):**

  ```sql
  CREATE TABLE IF NOT EXISTS breaker_cooldowns (
      host TEXT PRIMARY KEY,
      until_wall REAL NOT NULL,   -- UTC epoch seconds
      reason TEXT,
      updated_at REAL NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_cd_until ON breaker_cooldowns(until_wall);
  ```

* **API:**

  * `get_until(host) -> Optional[float_monotonic]`

    * Read `until_wall`; convert to **monotonic**:
      `until_monotonic = now_monotonic + max(0, until_wall - now_wall)`
  * `set_until(host, until_monotonic, reason)`

    * Convert back to wall: `until_wall = now_wall + max(0, until_monotonic - now_monotonic)`
    * `INSERT OR REPLACE` row.
  * `clear(host)`
* **Locking**: Use a short `BEGIN IMMEDIATE; ... COMMIT` around writes; reads can be `DEFERRED`. You’ve already added a SQLite file lock utility — reuse it to serialize writers.
* **GC**: periodically delete rows where `until_wall < (now_wall - 1)`.

### 2.2 `RedisCooldownStore` (optional)

* Keys: `breaker:cooldown:<host>`
* Value: JSON `{"until_wall": <float>, "reason": "retry-after|rolling-window"}`
* Set **TTL** to `ceil(until_wall - now_wall)` seconds for automatic expiry.
* Retrieval uses the same **wall→monotonic** conversion as SQLite.

> Both backends use **wall clock** in storage and convert to **monotonic** at runtime so you don’t depend on synchronized monotonic clocks across processes.

Wire one of these into the registry at startup by reading `cooldown_store` from your breaker YAML/env (you already drafted those keys in the config schema earlier).

---

## 3) Listener → telemetry wiring

In `networking_breaker_listener.py` you already have `NetworkBreakerListener`. Do this in `BreakerRegistry._get_or_create_host_breaker()` and `_get_or_create_resolver_breaker()`:

```python
listeners = []
if self.listener_factory:
    l = self.listener_factory(host, "host", None)  # or ("resolver", resolver)
    if l: listeners.append(l)

cb = pybreaker.CircuitBreaker(
    fail_max=base.fail_max,
    reset_timeout=base.reset_timeout_s,
    state_storage=None,
    listeners=listeners,
)
```

**Telemetry events produced:**

* `breaker_before_call`
* `breaker_success`
* `breaker_failure` (includes exception class)
* `breaker_state_change` (includes from→to, reset_timeout)

> Recommend: store these into your **SQLite sink** (`breaker_transitions` table) or JSONL—mirror your Wayback sink pattern.

---

## 4) Networking hub integration (precise order)

In your HTTP send wrapper (the one Tenacity wraps):

1. **Canonicalize** URL; derive `host` (punycoded lowercase).
2. **Breaker pre-flight:** `registry.allow(host, role, resolver)`

   * If `BreakerOpenError`, **do not** enter Tenacity; short-circuit with a precise reason (`breaker_open`, include `cooldown_remaining_ms`).
   * Optionally apply **half-open jitter** suggested by registry (see §1.1 note).
3. **Limiter acquire** (bounded wait).
4. **Hishel** (cache); if `from_cache=True`: **skip** breaker updates (no network happened).
5. **Tenacity loop** → send via HTTPX.
6. **Post-update**:

   * On **exception** → `on_failure(..., exception=e)`.
   * On **response**:

     * If status in `{429,500,502,503,504}` → `on_failure(status=..., retry_after_s=parsed_retry_after_if_429_or_503)`.
     * If neutral (e.g., 401/403/404/410/451) → **do not** update breaker (neither success nor failure).
     * Else (2xx/3xx) → `on_success(...)`.

**Tenacity predicate** must **exclude** `BreakerOpenError` so we never spin while a breaker is open.

---

## 5) CLI controls (quick win)

Add subcommands to your CLI (e.g., `src/DocsToKG/ContentDownload/cli.py`):

### 5.1 `docstokg breaker show`

* List all known hosts (from config) with:

  * host, pybreaker `state`, `reset_timeout_s`,
  * cooldown override in shared store: `open_until` (UTC) & `remaining_s` (computed),
  * rolling window counters (failures in current window),
  * half-open probe budget per role (if any active).
* Optionally filter by `--open-only`.

### 5.2 `docstokg breaker open <host> --seconds <N> [--reason <str>]`

* Write `cooldowns.set_until(host, now + N, reason="cli-open:<reason>")`.
* Do **not** force open pybreaker; the **override** is what blocks pre-flight.

### 5.3 `docstokg breaker close <host>`

* `cooldowns.clear(host)` and call `h_cb.call_success()` to reset counts.

> These commands talk to the same `BreakerRegistry` instance (or instantiate a lightweight “tooling” registry bound to the same shared cooldown store).

---

## 6) Run summary & metrics (what to measure)

Emit and summarize per run:

* **Opens/hour** per host:

  * Count `breaker_state_change` events to `OPEN`.
* **Time saved by break** (estimate):

  * For each request blocked by `allow()`, add `min(remaining_ms, expected_latency_ms)`; report per host.
* **Success-after-open rate**:

  * Count successes immediately after half-open sampling / cooldown expiry divided by trials.
* **Open reason mix**:

  * For each open period, capture whether it was triggered by `retry-after` or `rolling-window` (parse `reason` from cooldown store write).
* **Breaker denials** vs **limiter blocks**:

  * Helps tune—ideally limiter avoids most 429s so breaker rarely opens.

---

## 7) Tests (add to `tests/networking/test_breakers.py`)

1. **Consecutive failures open**

   * Drive `fail_max` 503s; assert `allow()` raises `BreakerOpenError` until `reset_timeout` elapses.
2. **Retry-After honored**

   * Return 429 with `Retry-After: 4`; assert immediate cooldown override for ~4s (capped).
3. **Rolling window manual open**

   * `threshold_failures=3, window_s=10`: fire 3 failures in <10s; assert manual open.
4. **Half-open probes per role**

   * After open elapses, assert only `trial_calls[metadata]=1` passes; second concurrent attempt triggers `BreakerOpenError`.
5. **Neutral statuses don’t count**

   * 404/403 do not increment failure count; breaker remains closed.
6. **Cache bypass**

   * When Hishel serves from cache, no breaker updates occur.
7. **Cross-process cooldown** (SQLite or Redis)

   * Process A: set open for 3s; Process B: `allow()` blocks; after 3s, B proceeds.
8. **CLI**

   * `breaker open/close/show` behave as expected.

---

## 8) Migration (delete legacy code)

* Remove the pipeline’s `_ensure_host_breaker`, `_host_breaker_allows`, `_update_breakers`, and any ad-hoc cooldown sleeps.
* All breaker logic must live in **networking** via `BreakerRegistry`.

---

## 9) Rollout plan

1. Land **in-proc** overrides (Retry-After + rolling window), **no cross-proc** yet.
2. Enable **listener**; collect a week of telemetry.
3. Add **SQLiteCooldownStore** (safe on one host) and flip registry to use it.
4. Optional: enable **AIMD** on the rate limiter first; breaker opens should drop.
5. Add **CLI**; document ops runbooks for “show/open/close”.

---

## 10) Acceptance criteria (DoD)

* Breaker pre-flight denies unhealthy hosts deterministically; Tenacity never retries `BreakerOpenError`.
* Retry-After opens are honored (capped); rolling window manual opens work.
* Half-open sampling limits trial calls per role; optional jitter applied.
* Cross-process cooldowns are respected (SQLite/Redis backend).
* Telemetry shows opens/hour, time saved, success-after-open, and reason mix.
* CLI can show/open/close breakers.
* Legacy breaker logic in pipeline is removed.

---

### Implementation notes & tips

* **Keys**: always normalize hosts to lowercase **punycode** (same helper as your URL/loader) before looking up policies.
* **Jitter**: keep it tiny (≤150 ms) and **only** in half-open to desynchronize probes. If you don’t want to change return types of `allow()`, let the hub compute a random jitter when `registry.current_state(host)` says `half_open`.
* **Ordering**: breaker check happens **before** limiter (no point rate-limiting something you won’t send). Cache hits bypass both limiter and breaker update.
* **Retry classifications**: keep them in one place (your existing `is_failure_for_breaker()` helper) so policy changes are consistent across breaker & Tenacity predicates.
* **Docs**: put a short policy block in `breakers.py` module docstring explaining what constitutes a failure, how Retry-After is respected, and how cooldown stores work (wall→monotonic conversion).

If you want, I can also provide a minimal `SQLiteCooldownStore` implementation snippet and a tiny `argparse` mixin for the three CLI subcommands so your agent can paste and wire them in one pass.
