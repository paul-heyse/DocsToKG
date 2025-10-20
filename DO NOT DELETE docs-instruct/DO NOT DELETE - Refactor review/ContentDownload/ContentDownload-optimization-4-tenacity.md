Perfect — here’s a **repo-shaped, junior-dev-friendly implementation plan** to make **Tenacity retries & backoff** precise, centralized, and spec-aware in `src/DocsToKG/ContentDownload`. It assumes your HTTP hub already uses **HTTPX + Hishel + limiter + breaker + url-normalize**.

---

# Goals (what “good” looks like)

* **One** place defines retry behavior (the networking hub).
* **Retry-After aware** waits that override exponential jitter when a server tells us how long to wait.
* **Tight predicate**: retry only on **HTTPX connect/read/write/timeout** and **HTTP {429,500,502,503,504}** (optionally 408).
* **Ceilings**: both **attempt count** and **wall-clock time**.
* **Patchable sleep** for tests (use the module’s `time.sleep`).
* **Classification helper** so resolvers never guess what’s retryable.
* **Structured logs & telemetry** for every retry (cause, wait, attempt X/Y).

---

# Config surface (new)

Add to your networking config (YAML/env/CLI):

```yaml
retries:
  max_attempts: 7                 # total tries = initial + 6 retries
  max_total_s: 120                # OR ceiling; set 0/None to disable
  backoff:
    multiplier: 0.75              # base for wait_random_exponential
    max_s: 60
  statuses: [429,500,502,503,504] # add 408 if desired
  methods: ["GET","HEAD"]         # retryable HTTP methods by default
  retry_408: false
  retry_on_timeout: true          # httpx.TimeoutException
  retry_on_remote_protocol: true  # httpx.RemoteProtocolError
  # honor Retry-After when present (cap in seconds)
  retry_after_cap_s: 900
  # POST/GraphQL: allow only if caller marks request idempotent
  allow_post_if_idempotent: true
```

Env/CLI overlays:

* `DOCSTOKG_RETRIES_MAX_ATTEMPTS=...`, `DOCSTOKG_RETRIES_MAX_TOTAL_S=...`
* `DOCSTOKG_RETRIES_STATUSES=429,500,502,503,504`
* `DOCSTOKG_RETRIES_METHODS=GET,HEAD`
* `--retry-statuses 429,500,502,503,504`, `--retry-max-attempts 7`, etc.

---

# File changes (high-level)

* **`networking.py`** (the hub): everything lives here.

  * `is_retryable(status, exception, method, *, offline, breaker_open) -> bool`
  * `_build_tenacity_retrying(cfg) -> tenacity.Retrying`
  * `_wait_retry_after(fallback_wait)`: Tenacity wait strategy that prefers `Retry-After`
  * **Send wrapper**: uses the above; logs a line before each sleep; updates telemetry.
* **(optional)** `retries_config_loader.py`: parse YAML/env/CLI into a small dataclass if you want parity with your other loaders.

---

# Step 1 — Classification helper (single source of truth)

Add to `networking.py`:

**Signature**

```python
def is_retryable(
    *,
    method: str,
    status: int | None,
    exception: BaseException | None,
    offline: bool,
    breaker_open: bool,
    cfg: RetriesConfig,
) -> bool:
    ...
```

**Rules**

* **Return False immediately** if:

  * `offline` mode is on (we used `only-if-cached`): synthetic 504s are not retryable.
  * a **breaker** denied the call (`BreakerOpenError`) — never loop on our own open circuit.
* **HTTP methods**: allow only from `cfg.methods` (default `GET`,`HEAD`).

  * If method is `POST`: only retry when caller marks request idempotent, e.g. `request.extensions["idempotent"]=True`.
* **Network exceptions** (retry): `httpx.ConnectError`, `httpx.ReadError`, `httpx.WriteError`, `httpx.TimeoutException`, and (config-gated) `httpx.RemoteProtocolError`.

  * **Do not** retry `httpx.LocalProtocolError` (usually client bug) or `BreakerOpenError`.
* **HTTP statuses**: retry **only** if `status ∈ cfg.statuses` (default `{429,500,502,503,504}`; add 408 if you opt in).

  * Never retry 4xx (except 429), `401/403/404/410/451`, or semantic errors (e.g. 422).

Keep this helper **pure** so you can unit-test it in isolation.

---

# Step 2 — Retry-After aware wait

Add a tiny Tenacity `wait` that prefers `Retry-After`:

**Signature**

```python
class _WaitRetryAfter(tenacity.wait.wait_base):
    def __init__(self, fallback: tenacity.wait.wait_base, cap_s: float): ...
    def __call__(self, retry_state) -> float: ...
```

**Behavior**

* Inspect the last outcome:

  * If it’s a **response** and the status is `429` or `503`, parse `Retry-After` using your existing header parser (supports **seconds** or **HTTP-date**).
  * If parsed and > 0, return `min(parsed, cap_s)`.
* Else, return the **fallback**: `wait_random_exponential(multiplier=cfg.backoff.multiplier, max=cfg.backoff.max_s)`.

This guarantees spec-aware pacing when servers send backpressure signals.

---

# Step 3 — Tenacity Retrying builder

**Signature**

```python
def _build_tenacity_retrying(cfg: RetriesConfig) -> tenacity.Retrying:
    ...
```

**Ingredients**

* **Retry predicate**:

  * `retry_if_exception(lambda e: is_retryable(method=..., status=None, exception=e, ...))`
  * `| retry_if_result(lambda resp: is_retryable(method=..., status=resp.status_code, exception=None, ...))`
  * You’ll bind `method`, `offline`, `breaker_open` via closure or small shim.
* **Stop policy**:

  * `stop_after_attempt(cfg.max_attempts)` **|** (`stop_after_delay(cfg.max_total_s)` if provided).
* **Wait policy**:

  * `_WaitRetryAfter(fallback=wait_random_exponential(...), cap_s=cfg.retry_after_cap_s)`
* **Sleep function**:

  * `sleep=time.sleep` (imported from this module) so tests can monkeypatch it.
* **before_sleep hook (structured log)**:

  * A function that logs **one line per retry** with:

    * `host`, `role`,
    * `attempt n/N`, `next_wait_ms`,
    * `cause=HTTP <code>` or `cause=<Exception>`,
    * `retry_after_s` when available,
    * request `method`, `url` (redacted if sensitive).
  * Also increment telemetry counters (see Step 5).

**Notes**

* Keep **HTTPX transport** `retries` on connect at **0–1** (not higher) to avoid “double stacking” with Tenacity. One connect retry at transport level is fine to paper over SYN race; everything else is Tenacity’s job.

---

# Step 4 — Wire in the hub send wrapper

In the **single** function that all resolvers call to perform HTTP, insert:

1. **Role & host**: extract from `request.extensions["role"]` and normalized URL (`urls.canonical_for_request`).
2. **Offline flag**: from CLI/env; if on, add header `Cache-Control: only-if-cached` on cached client; **do not** retry synthetic 504s.
3. **Breaker pre-flight**: `registry.allow(...)` — if it raises, **don’t** enter Tenacity.
4. **Pick client**: cached vs raw based on your cache policy and role.
5. **Create Retrying**: `_build_tenacity_retrying(cfg)` with the **bound context**:

   * method, offline, breaker_open=False (we’re past pre-flight), role, host.
6. **Attempt loop**:

   * With Tenacity’s `for attempt in retrying:` guard/body pattern:

     * Send request (HTTPX client); if `is_retryable(status, None, ...)` returns **True**, close the response and let Tenacity retry.
     * If **429**/**503**, parse `Retry-After` for logging (wait engine already picks it up).
     * On **exception**, raise to Tenacity; your retry predicate decides.
7. **Post-response integration**:

   * Update **breaker** accounting: success vs failure (aligned with its classifier).
   * Update **telemetry** (Step 5).
   * Return response.

**Special cases**

* **Method gating**: before even building `Retrying`, if the method is not in `cfg.methods` and not explicitly marked idempotent, set `max_attempts=1`.
* **Offline 504** from Hishel: treat as non-retryable; emit `blocked_offline` reason.

---

# Step 5 — Telemetry & metrics

Per request, capture:

* `retry_attempts_total` (count of retries performed)
* `retry_total_backoff_ms` (sum of waits)
* `retry_last_cause` (`HTTP 503`, `ConnectError`, etc.)
* `retry_success_after_n` (if succeeded after >0 retries)
* `nonretry_errors_total` (errors deliberately not retried: 404, BreakerOpenError, offline 504, etc.)

**Run summary**

* **Retries per request** (mean, p95)
* **Average backoff** (ms)
* **% retried that eventually succeeded**
* **% non-retried errors** (sanity check for predicate tightness)
* **Top hosts by retry volume** (guide for rate/breaker tuning)

---

# Step 6 — Logging (quick win)

Add a compact, structured **one-liner** for every retry (in `before_sleep`):

```
retry host=api.crossref.org role=metadata method=GET code=503 attempt=3/7 wait_ms=4200 retry_after_s=4 url=https://...
```

If the cause is an exception:

```
retry host=... role=... method=GET exc=httpx.ConnectError attempt=1/7 wait_ms=800 url=https://...
```

**Redact** query strings for signed artifact URLs.

---

# Step 7 — Tests (must add)

1. **Retry-After precedence**

   * 503 with `Retry-After: 4` → next wait ~4s (capped), not exponential fallback.

2. **Status predicate**

   * 500/502/503/504 retry; 404/401/403 do **not** retry.
   * Optional: 408 obeys `retry_408` toggle.

3. **Exception predicate**

   * `httpx.ConnectError` / `TimeoutException` retry;
   * `httpx.LocalProtocolError` does **not**.

4. **Methods**

   * POST without `idempotent=True` does **not** retry; with it, retries.

5. **Ceilings**

   * `stop_after_attempt`: hit attempt cap exactly; `stop_after_delay`: stop by wall-clock.

6. **Sleep patchability**

   * Monkeypatch `DocsToKG.ContentDownload.networking.time.sleep`; assert Tenacity calls the patched function (no real sleeps in unit tests).

7. **Offline**

   * With `only-if-cached`, on miss return synthetic 504; **no** retry; log `blocked_offline`.

8. **Breaker interplay**

   * When `BreakerOpenError`, verify we **don’t** enter Tenacity loop.

9. **Log line emitted**

   * Spy logger to verify one structured line per retry with correct fields.

---

# Step 8 — Operational guidance & defaults

* Start with: `max_attempts=7`, `max_total_s=120`, `multiplier=0.75`, `max_s=60`, `statuses={429,500,502,503,504}`.
* For **high-latency** origins (e.g., arXiv/PMCs), you can bump `max_total_s` slightly (not attempts).
* Re-evaluate after a week using telemetry: if many retries end in failure, tighten predicates or increase **rate-limit** smoothing before touching retries.

---

# Step 9 — Guardrails & pitfalls

* Avoid transport **connect retries >1**; Tenacity owns backoff.
* Don’t retry **client bugs** (`LocalProtocolError`) or **auth/semantic failures**.
* Don’t double-sleep: let Tenacity control **all** waits; rate limiter already paced you **before** the call.
* Respect **Retry-After** strictly; log when you honor it (seconds used).

---

# Definition of Done

* One Tenacity policy in the hub; resolvers have **zero** sleeps/loops.
* `is_retryable()` used consistently by both Tenacity predicates and post-response breaker updates.
* Retry-After aware wait **in effect** (validated by tests).
* Structured per-retry log lines and counters in telemetry.
* Ceilings enforced (attempts **or** wall clock), and sleep function patchable in tests.

If you want, I can also draft tiny function signatures (docstrings + parameter lists) for `_build_tenacity_retrying`, `_WaitRetryAfter`, and `is_retryable` exactly as they should appear in `networking.py`, so your agent can paste them in and fill bodies quickly.
