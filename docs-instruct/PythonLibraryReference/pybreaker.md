Below is an engineering‑refactor guide to **pybreaker**—a small, threadsafe Circuit Breaker library for Python. It’s organized so an AI agent can replace custom “retry/fail‑fast” logic with clean, observable breakers, and decide when to pick pybreaker vs. alternatives.

---

## 0) Snapshot (what, version, Python support)

* **What:** `pybreaker` implements the classic **Circuit Breaker** pattern (Closed → Open → Half‑Open) from *Release It!*, with configurable thresholds, listeners (hooks), and pluggable state storage (in‑memory or Redis). It can be used as a decorator, a direct `.call()` wrapper, or as a context manager; there’s also optional support for **Tornado** async coroutines. ([GitHub][1])
* **Latest release:** **1.4.1** on PyPI (released **Sep 21, 2025**). License: BSD. PyPI metadata lists **Python ≥ 3.9**. (The README in the repo shows “3.10+”; treat the PyPI classifiers/metadata as authoritative for installation.) ([PyPI][2])

---

## 1) Install & basic shape

```bash
pip install pybreaker
```

Create one breaker **per integration point** and keep it **global** (shared across calls/requests) so it can accumulate failures/successes over time:

```python
import pybreaker

# e.g., DB calls
db_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)
```

The README recommends keeping breakers in application scope. ([GitHub][1])

---

## 2) Mental model (how the breaker behaves)

* **Closed:** calls pass through; consecutive failures are counted.
* **Open:** calls are **short‑circuited** and raise `CircuitBreakerError` immediately.
* **Half‑Open:** after `reset_timeout` seconds, **one trial call** is permitted; on success, the breaker **closes**, on failure, it **re‑opens**.
* **Defaults:** with default parameters the breaker opens **after 5 consecutive failures** and tries again **after 60 seconds**. ([GitHub][1])

> Pybreaker implements these states and exposes them as strings: `'closed'`, `'open'`, `'half-open'`. Constants are exported (e.g., `STATE_CLOSED`) and also used by storage backends. ([infuse.readthedocs.io][3])

---

## 3) The core API (three ways to use it)

### A) Decorator (most common)

```python
@db_breaker
def update_customer(cust):
    return call_database(cust)
```

### B) Direct call

```python
result = db_breaker.call(update_customer, cust)
```

### C) Context manager

```python
with db_breaker.calling():
    update_customer(cust)
```

All three are documented; they trigger the same state machine and error behavior. ([GitHub][1])

---

## 4) Configuration knobs that replace home‑grown logic

```python
pybreaker.CircuitBreaker(
    fail_max=5,              # how many consecutive failures trip the breaker
    reset_timeout=60,        # seconds to wait before moving from OPEN -> HALF-OPEN
    success_threshold=None,  # optional: require N consecutive successes to close
    exclude=None,            # list[Exception | (Exception) -> bool]
    listeners=None,          # list[CircuitBreakerListener]
    state_storage=None,      # CircuitMemoryStorage (default) or CircuitRedisStorage
    name=None,               # optional identifier for logs/metrics
    # throw_new_error_on_trip is also supported; see below
)
```

* **`success_threshold`**: if set (e.g., `3`), the breaker requires **N consecutive successes** (in Half‑Open) before closing; useful when recovery is flaky. ([GitHub][1])
* **`exclude`**: you can pass **exception classes** and/or **predicates** that, when matched, do **not** count as failures (e.g., a 4xx business error). ([GitHub][1])
* **`throw_new_error_on_trip`**: by default, once open, pybreaker raises its own `CircuitBreakerError`. Set `throw_new_error_on_trip=False` to **re‑raise the original exception** when the breaker trips. ([GitHub][1])

---

## 5) Observability & hooks (listeners)

Attach **listeners** for logging, metrics, tracing:

```python
import logging, pybreaker

class LogListener(pybreaker.CircuitBreakerListener):
    def state_change(self, cb, old_state, new_state):
        logging.info("CB %s: %s → %s", cb.name, old_state, new_state)

    def failure(self, cb, exc):  # every counted failure
        logging.warning("CB %s failure: %r", cb.name, exc)

    def success(self, cb):       # every success
        logging.debug("CB %s success", cb.name)

db_breaker = pybreaker.CircuitBreaker(listeners=[LogListener()], name="db")
```

Listener hooks include `before_call`, `failure`, `success`, and `state_change`. You can add/remove listeners dynamically as well. ([GitHub][1])

**Runtime controls & counters** you can query/alter:

```python
db_breaker.fail_counter          # consecutive failures
db_breaker.success_counter       # consecutive successes
db_breaker.fail_max = 10
db_breaker.success_threshold = 3
db_breaker.reset_timeout = 120
db_breaker.current_state         # 'open' | 'half-open' | 'closed'
db_breaker.open(); db_breaker.half_open(); db_breaker.close()   # force state
```

These are explicitly listed in the README’s “Monitoring and Management” section. ([GitHub][1])

---

## 6) Storage backends (single‑process vs. shared)

### In‑process (default)

If you pass nothing, pybreaker uses **`CircuitMemoryStorage`** (per‑process memory). Good for single‑process services. ([infuse.readthedocs.io][3])

### Redis (multi‑process / multi‑host)

To share state across workers/hosts:

```python
import redis, pybreaker
r = redis.StrictRedis()  # IMPORTANT: don't set decode_responses=True
db_breaker = pybreaker.CircuitBreaker(
    fail_max=5, reset_timeout=60,
    state_storage=pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, r, namespace="my-service-db")
)
```

Notes from the README:

* **`decode_responses=True`** is **not supported** (you’ll get `'str' object has no attribute "decode"'` in Python 3+).
* When you create multiple breakers that share Redis, **set a unique `namespace`** for each to avoid interference. ([GitHub][1])

> If you’re curious about the Redis storage implementation: the class holds `state`, `counter`, and `opened_at`, and uses a namespace under which keys are stored; the docs show the `namespace` and fallback fields. ([infuse.readthedocs.io][3])

---

## 7) Optional Tornado support (async coroutines)

Pybreaker can call **Tornado** coroutines either via a special decorator kwarg or via `.call_async()`:

```python
from tornado import gen

@db_breaker(__pybreaker_call_async=True)
@gen.coroutine
def async_update(cust):
    ...
# or
yield db_breaker.call_async(async_update, cust)
```

If you are using **asyncio/HTTPX/async frameworks**, prefer **aiobreaker** (separate project) rather than Tornado hooks. ([GitHub][1])

---

## 8) Refactor mapping (custom code → pybreaker)

| If your code currently…                                    | Replace it with…                                                                  |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Counts consecutive failures and “fails fast” for N seconds | `CircuitBreaker(fail_max=N, reset_timeout=T)`; wrap the risky function            |
| Special‑cases business errors (e.g., 4xx)                  | `exclude=[YourError, lambda e: isinstance(e, HTTPError) and e.status_code < 500]` |
| Requires several OKs before declaring recovery             | `success_threshold=M`                                                             |
| Logs state changes / success / failure                     | `CircuitBreakerListener` with `state_change`, `success`, `failure`                |
| Manually toggles a “maintenance switch”                    | Call `breaker.open()` / `close()` programmatically                                |
| Shares health across workers                               | Use `CircuitRedisStorage(..., namespace="...")`                                   |
| Needs original exception semantics when tripped            | `throw_new_error_on_trip=False`                                                   |

All features above are in the project README, with code snippets and notes. ([GitHub][1])

---

## 9) Example patterns you’ll most likely need

### 9.1 Wrap an HTTP call (requests/HTTPX)

```python
import httpx, pybreaker

api_breaker = pybreaker.CircuitBreaker(
    name="billing-api",
    fail_max=10,
    reset_timeout=30,
    exclude=[lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500],
)

@api_breaker
def charge(customer_id, amount):
    with httpx.Client(timeout=5) as client:
        r = client.post("https://billing/api/charge", json={"id": customer_id, "amount": amount})
        r.raise_for_status()
        return r.json()
```

Note: pybreaker won’t add I/O timeouts—**you** should set them on the client (best practice in the README). ([GitHub][1])

### 9.2 Provide a safe fallback

```python
try:
    invoice = charge(cust, amount)
except pybreaker.CircuitBreakerError:
    invoice = {"status": "pending", "source": "cache"}  # app-specific fallback
```

When the circuit is **Open**, calls fail fast with `CircuitBreakerError` unless `throw_new_error_on_trip=False`. ([GitHub][1])

### 9.3 Promote observability with metrics

```python
from prometheus_client import Counter
cb_state_changes = Counter("cb_state_change_total", "CB state changes", ["name", "new_state"])

class MetricsListener(pybreaker.CircuitBreakerListener):
    def state_change(self, cb, old, new):
        cb_state_changes.labels(cb.name or "default", new).inc()
```

The listener API is designed exactly for this kind of integration. ([GitHub][1])

---

## 10) Tuning advice (pragmatic rules)

* Start with **`fail_max`** near your **SLO window**: e.g., if a dependency is flaky bursts of ≤3 calls, choose `fail_max=3–5`.
* Keep **`reset_timeout`** short enough to probe recovery (e.g., 15–60s), but long enough to stop hammering a dying dependency.
* Use **`success_threshold`** when the dependency sometimes “ghosts” back—require 2–3 clean calls before closing.
* Exclude **business‑level exceptions** (like 4xx validations) so they **don’t trip** the breaker. All of these are first‑class options in pybreaker. ([GitHub][1])

---

## 11) Operational controls (from code/console)

You can **force a state** during maintenance or feature flags:

```python
if draining_db:
    db_breaker.open()
# Or reopen after maintenance
db_breaker.close()
```

You can also inspect counters and expose them in a health endpoint or admin UI. The API for `fail_counter`, `success_counter`, `current_state`, and state methods is documented. ([GitHub][1])

---

## 12) Testing your breaker

* Unit‑test behavior by making a wrapped function fail N times in a row and assert that the N+1 call raises `CircuitBreakerError`.
* Simulate the **Half‑Open** probe by advancing time or temporarily setting a short `reset_timeout`.
* Attach a **test listener** and assert that `state_change` fired with `closed → open`, etc. The usage patterns are shown in README; Q&A threads also discuss test strategies. ([GitHub][1])

---

## 13) Async notes (choosing the right tool)

* Pybreaker supports **Tornado** (`@gen.coroutine`) via `__pybreaker_call_async=True` or `.call_async()`. ([GitHub][1])
* For **asyncio** codebases, prefer **aiobreaker** (same idea, asyncio‑native, with listeners and Redis support). Keep pybreaker for sync code or when you already depend on Tornado. ([aiobreaker.netlify.app][4])

---

## 14) Common pitfalls & guardrails

* **No timeouts inside pybreaker** → always set client timeouts; breakers + timeouts go together. ([GitHub][1])
* **Redis storage collisions** → set a **unique `namespace`** per breaker when using `CircuitRedisStorage`. Don’t enable Redis `decode_responses=True`. ([GitHub][1])
* **Re‑raising behavior** → if your logs/alerts depend on original exceptions, set `throw_new_error_on_trip=False`. ([GitHub][1])
* **Process model** → memory storage is **per‑process**; if you scale with multiple workers and need a shared view, use Redis storage. (The project’s issue tracker also discusses behavior with shared storage and namespaces.) ([GitHub][5])

---

## 15) Minimal “drop‑in” recipes for an AI agent

### Replace your custom “N strikes then sleep” wrapper

```python
breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)
@breaker
def risky_call(...): ...
```

Defaults match the typical “5 failures / 60s cooldown” example; tune as needed. ([GitHub][1])

### Ignore business errors; only count 5xx/timeouts

```python
import httpx
breaker = pybreaker.CircuitBreaker(
    exclude=[lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500]
)
```

Exclusion takes classes or predicates. ([GitHub][1])

### Require 3 successes before closing

```python
breaker = pybreaker.CircuitBreaker(success_threshold=3)
```

Closes only after 3 consecutive success calls when Half‑Open. ([GitHub][1])

### Share breaker state across workers

```python
storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis_client, namespace="svc-A-db")
breaker = pybreaker.CircuitBreaker(state_storage=storage)
```

Use `namespace`; avoid `decode_responses=True` on the Redis client. ([GitHub][1])

---

## 16) When to consider alternatives

* **Asyncio first:** choose **aiobreaker** (async/await, optional Redis backing). ([aiobreaker.netlify.app][4])
* **All‑in‑one decorator style with built‑in retries/timeouts:** libraries like `circuitbreaker` offer different ergonomics; compare feature sets (but pybreaker’s listeners and storages are strong). ([PyPI][6])

---

## 17) Quick reference (cheat sheet)

* **Constructor:** `CircuitBreaker(fail_max, reset_timeout, success_threshold=None, exclude=None, listeners=None, state_storage=None, name=None)`
* **Use:** `@breaker`, `breaker.call(fn, *a, **kw)`, `with breaker.calling(): ...`
* **States:** `current_state` ∈ `{ 'closed','open','half-open' }`; force with `.open() / .half_open() / .close()`
* **Counters:** `fail_counter`, `success_counter` (read‑only); `fail_max`, `success_threshold`, `reset_timeout` (settable)
* **Errors:** open circuit → `CircuitBreakerError` unless `throw_new_error_on_trip=False`
* **Storage:** default `CircuitMemoryStorage`; shared `CircuitRedisStorage(state, redis, namespace=...)`
* **Listeners:** subclass `CircuitBreakerListener` and pass in `listeners=[...]`
  All of these items are explicitly shown in the README and project metadata. ([GitHub][1])

---

### Sources (project “source of truth”)

* **PyPI project page / metadata / release history** (version, Python requirement, description). ([PyPI][2])
* **GitHub README** (features, usage modes, Tornado support, listeners, Redis notes, counters & state methods). ([GitHub][1])
* **Module constants & storage details** (state names; memory storage default constructor logic). ([infuse.readthedocs.io][3])

If you share a snippet of your current fail‑fast/retry wrapper (including what you count as a “failure”), I can translate it 1:1 into a `CircuitBreaker(...)` with the right `exclude`, `success_threshold`, listeners, and state storage, plus a minimal test to lock the behavior in.

[1]: https://github.com/danielfm/pybreaker "GitHub - danielfm/pybreaker: Python implementation of the Circuit Breaker pattern."
[2]: https://pypi.org/project/pybreaker/ "pybreaker · PyPI"
[3]: https://infuse.readthedocs.io/en/latest/_modules/pybreaker/?utm_source=chatgpt.com "pybreaker — Infuse 0.4.1.dev0 documentation - Read the Docs"
[4]: https://aiobreaker.netlify.app/?utm_source=chatgpt.com "aiobreaker — aiobreaker 1.1.0 documentation"
[5]: https://github.com/danielfm/pybreaker/issues/26?utm_source=chatgpt.com "Circuit erroneously closed when using a shared storage #26"
[6]: https://pypi.org/project/circuitbreaker/?utm_source=chatgpt.com "circuitbreaker"
