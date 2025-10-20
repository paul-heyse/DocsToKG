Amazing—here’s a **complete, production-oriented scope** for adding a **pybreaker-based per-host circuit breaker** to `src/DocsToKG/ContentDownload`, including:

1. a **config schema** (YAML + env/CLI overlay)
2. recommended **default breaker policies** for your sources
3. a **BreakerRegistry** design (what it does, how it integrates)
4. a **BreakerListener** you can drop in to emit state-transition telemetry
5. wiring, telemetry fields, tests, and rollout checklist

This is written so a junior dev can implement it end-to-end without guessing.

---

# 1) Config schema (YAML; overlayable via env/CLI)

> Put this in `configs/networking/breakers.yaml` (or your central config). You’ll load it once at startup and merge env/CLI overrides.

```yaml
version: 1
defaults:
  fail_max: 5                 # consecutive failures to trip
  reset_timeout_s: 60         # time before trying again (half-open)
  retry_after_cap_s: 900      # cap when honoring Retry-After
  classify:
    failure_statuses: [429, 500, 502, 503, 504, 408]
    neutral_statuses: [401, 403, 404, 410, 451]
    # anything else is treated as success for breaker accounting
  roles:                      # optional per-role overrides of fail_max/reset
    metadata:
      fail_max: 5
      reset_timeout_s: 45
    landing:
      fail_max: 4
      reset_timeout_s: 60
    artifact:
      fail_max: 3
      reset_timeout_s: 120
  half_open:
    trial_calls:              # number of "probe" calls allowed when half-open
      metadata: 1
      landing: 1
      artifact: 2
    jitter_ms: 150            # optional stagger when multiple workers resume
advanced:
  rolling_window:             # optional manual-open heuristic (best effort)
    enabled: true
    window_s: 30
    threshold_failures: 6     # if >=6 failures in 30s, force-open
    cooldown_s: 60            # manual open duration
  cooldown_store:             # optional cross-process store for cooldowns
    backend: "none"           # "none" | "sqlite" | "redis"
    dsn: ""                   # e.g. file path or redis:// URL (if used)
hosts:
  api.crossref.org:
    fail_max: 5
    reset_timeout_s: 60
  api.openalex.org:
    fail_max: 5
    reset_timeout_s: 45
  api.unpaywall.org:
    fail_max: 5
    reset_timeout_s: 120
  export.arxiv.org:
    fail_max: 3
    reset_timeout_s: 180
  api.semanticscholar.org:
    fail_max: 5
    reset_timeout_s: 60
  eutils.ncbi.nlm.nih.gov:
    fail_max: 5
    reset_timeout_s: 120
  europepmc.org:
    fail_max: 5
    reset_timeout_s: 60
  api.osf.io:
    fail_max: 5
    reset_timeout_s: 90
  zenodo.org:
    fail_max: 5
    reset_timeout_s: 90
  api.figshare.com:
    fail_max: 5
    reset_timeout_s: 90
  api.core.ac.uk:
    fail_max: 5
    reset_timeout_s: 90
  doaj.org:
    fail_max: 5
    reset_timeout_s: 90
  api.openaire.eu:
    fail_max: 5
    reset_timeout_s: 120
  hal.science:
    fail_max: 4
    reset_timeout_s: 120
  web.archive.org:
    roles:
      metadata: { fail_max: 3, reset_timeout_s: 60 }
      artifact: { fail_max: 2, reset_timeout_s: 120 }
resolvers:                    # optional overrides for specific resolvers
  landing_page:
    fail_max: 4
    reset_timeout_s: 45
```

### Env overlays (read once at startup)

* `DOCSTOKG_BREAKERS_YAML=/path/to/breakers.yaml`
* `DOCSTOKG_BREAKER__api.crossref.org=fail_max:5,reset:60,retry_after_cap:900`
* `DOCSTOKG_BREAKER_ROLE__web.archive.org__artifact=fail_max:2,reset:120`
* `DOCSTOKG_BREAKER_RESOLVER__landing_page=fail_max:4,reset:45`
* `DOCSTOKG_BREAKER_ADV__ROLLING=enabled:true,window:30,thresh:6,cooldown:60`
* `DOCSTOKG_BREAKER_COOLDOWN_BACKEND=sqlite` / `DOCSTOKG_BREAKER_COOLDOWN_DSN=/run/locks/breakers.sqlite`

### CLI overrides (examples)

```
docstokg run ... \
  --breaker api.crossref.org=fail_max:5,reset:60 \
  --breaker-role web.archive.org:artifact=fail_max:2,reset:120 \
  --breaker-resolver landing_page=fail_max:4,reset:45 \
  --breaker-rolling enabled=true,window=30,thresh=6,cooldown=60
```

Overlay order: **YAML → env → CLI** (last writer wins).

---

# 2) Breaker policies (what we recommend out of the box)

* **Crossref** `api.crossref.org`: `fail_max=5`, `reset=60s`
* **OpenAlex** `api.openalex.org`: `5 / 45s`
* **Unpaywall** `api.unpaywall.org`: `5 / 120s`
* **arXiv (export)** `export.arxiv.org`: `3 / 180s`
* **Semantic Scholar** `api.semanticscholar.org`: `5 / 60s`
* **NCBI eUtils** `eutils.ncbi.nlm.nih.gov`: `5 / 120s`
* **Europe PMC** `europepmc.org`: `5 / 60s`
* **OSF** `api.osf.io`: `5 / 90s`
* **Zenodo** `zenodo.org`: `5 / 90s`
* **Figshare** `api.figshare.com`: `5 / 90s`
* **CORE** `api.core.ac.uk`: `5 / 90s`
* **DOAJ** `doaj.org`: `5 / 90s`
* **OpenAIRE** `api.openaire.eu`: `5 / 120s`
* **HAL** `hal.science`: `4 / 120s`
* **Wayback** `web.archive.org`:

  * `metadata`: `3 / 60s`
  * `artifact`: `2 / 120s`
* **Defaults for unknown hosts**: use top-level defaults, then tighten based on telemetry.

**Roles:** stricter for `artifact` (PDF loads), looser for `metadata` to keep discovery snappy.

---

# 3) BreakerRegistry (shape & responsibilities)

> Lives in `src/DocsToKG/ContentDownload/networking.py` (or a small `breakers.py` that networking imports). One instance per process.

**Responsibilities**

* **Keying:** `(host)` and optional `(resolver)`, with canonicalized host (your `urls.canonical_for_request` already ensures stable host).

* **Pre-flight `allow()`**

  * Check **cooldown overrides** first (Retry-After or rolling-window manual open).
  * Check pybreaker’s state; if open, compute remaining cooldown and **raise `BreakerOpenError(host)`**.

* **Post-update**

  * On **success**: `cb.call_success()` (pybreaker) and clear cooldown override for host.
  * On **failure**: increment breaker (pybreaker failure), and if status is 429/503 with `Retry-After`, set the **cooldown override**: `open_until = now + min(retry_after, retry_after_cap_s)`.

* **Half-open sampling**

  * pybreaker allows 1 trial call after timeout. To support *K* trial calls, keep a tiny counter per `(host, role)` while state is half-open and deny further `allow()` after K “probe” calls until a success closes the breaker (or a failure re-opens).

* **Rolling-window manual open (optional)**

  * Keep a small deque of timestamps for **failures** per host. If there are ≥ `threshold_failures` within `window_s`, set a cooldown override of `cooldown_s`. This doesn’t change pybreaker state; it just makes `allow()` raise `BreakerOpenError` until the manual open expires.

* **Shared cooldown store (optional)**

  * If configured, persist `{host, open_until, reason}` in SQLite/Redis so **other processes** see the open period. Networking reads this at `allow()` and writes it at post-failure. The breaker’s actual state can remain process-local; the **cooldown** is what matters for cross-process politeness.

---

# 4) BreakerListener (telemetry for state transitions)

> Drop this into `src/DocsToKG/ContentDownload/networking_breaker_listener.py`. It plugs into every pybreaker instance via the `listeners=[...]` arg.

```python
# networking_breaker_listener.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Mapping, Any, Optional
import time

try:
    import pybreaker
except ImportError:  # pragma: no cover
    pybreaker = None  # type: ignore

class BreakerTelemetrySink(Protocol):
    def emit(self, event: Mapping[str, Any]) -> None: ...

@dataclass
class BreakerListenerConfig:
    run_id: str
    host: str                   # breaker key (host)
    scope: str = "host"         # "host" | "resolver"
    resolver: Optional[str] = None

class NetworkBreakerListener(pybreaker.CircuitBreakerListener):  # type: ignore[misc]
    """Emits state transitions & per-call signals for a single breaker."""

    def __init__(self, sink: BreakerTelemetrySink, cfg: BreakerListenerConfig):
        self.sink = sink
        self.cfg = cfg

    def _emit(self, event_type: str, **body: Any) -> None:
        payload = {
            "event_type": f"breaker_{event_type}",
            "ts": time.time(),
            "run_id": self.cfg.run_id,
            "host": self.cfg.host,
            "scope": self.cfg.scope,
            "resolver": self.cfg.resolver,
        }
        payload.update(body)
        self.sink.emit(payload)

    # Called right before the protected call executes
    def before_call(self, cb, func, *args, **kwargs):
        self._emit("before_call", state=str(cb.current_state))

    # Called when the protected call succeeds
    def success(self, cb):
        self._emit("success", state=str(cb.current_state), fail_counter=getattr(cb, "fail_counter", None))

    # Called when the protected call fails (exception raised by the call)
    def failure(self, cb, exc):
        self._emit("failure", state=str(cb.current_state), exc_type=type(exc).__name__, msg=str(exc)[:300],
                   fail_counter=getattr(cb, "fail_counter", None))

    # Called when circuit breaker state changes
    def state_change(self, cb, old_state, new_state):
        # States are classes; stringify for logs
        self._emit("state_change", old=str(old_state), new=str(new_state),
                   reset_timeout_s=getattr(cb, "reset_timeout", None))
```

**Where to wire it**

When you construct each pybreaker instance in `BreakerRegistry`, do:

```python
cb = pybreaker.CircuitBreaker(
    fail_max=policy.fail_max,
    reset_timeout=policy.reset_timeout_s,
    state_storage=None,  # in-proc; cooldown override is handled outside
    listeners=[NetworkBreakerListener(telemetry_sink, BreakerListenerConfig(run_id, host))]
)
```

If you also create per-resolver breakers, pass `scope="resolver"` and `resolver=<name>` in the listener config.

**Telemetry events generated**

* `breaker_before_call`: `{host, scope, state}`
* `breaker_success`: `{state, fail_counter}`
* `breaker_failure`: `{state, exc_type, msg, fail_counter}`
* `breaker_state_change`: `{old, new, reset_timeout_s}`

You can write a small SQLite sink mirroring your Wayback sink (table `breaker_transitions`) or reuse your JSONL sink.

---

# 5) Integration points in networking (what to change)

> All changes are localized to the networking hub wrapper around HTTPX calls.

1. **Create BreakerRegistry** once at startup (inject run_id & telemetry sink).

2. **On every request:**

   * Determine **host** (canonicalized) and **role** (`metadata|landing|artifact`).
   * **Pre-flight**: `registry.allow(host, resolver, role)` (raise `BreakerOpenError` if not allowed).
   * **Limiter** acquire → **Hishel** check → **HTTPX/Tenacity** attempt.
   * **Post-update**:

     * If HTTPX exception or status in `{429,500,502,503,504,408}` → `registry.on_failure(host, resolver, status=..., exception=..., retry_after_s=...)`
     * Else if status in `{401,403,404,410,451}` → **neutral** (do nothing)
     * Else → `registry.on_success(host, resolver)`

3. **Tenacity**

   * Ensure your retry predicate **excludes** `BreakerOpenError` so Tenacity doesn’t busy-loop while the breaker is open.
   * Keep retries for network exceptions & retryable statuses.

---

# 6) Minimal policy/loader types (to steer the implementation)

* **`BreakerPolicy`**: `fail_max:int`, `reset_timeout_s:int`, optional `retry_after_cap_s:int`, and optional per-role overrides.
* **`BreakerClassification`**: sets for failure/neutral statuses; exceptions you treat as failure.
* **`BreakerConfig`**: `defaults`, `advanced`, `hosts`, `resolvers`.

Loader order:

1. Parse YAML → Pydantic/dataclasses
2. Apply env overlays (parse `fail_max`, `reset`, `retry_after_cap`)
3. Apply CLI overrides
4. Freeze into an immutable registry object

Validate:

* `fail_max >= 1`, `reset_timeout_s > 0`
* Role overrides: if present, they override only listed fields
* Hosts must be FQDNs; normalize to lower-case punycode for the key

---

# 7) Telemetry fields to add per HTTP request

Augment your existing per-request telemetry with:

* `breaker_host_state`: `closed|open|half_open` (from pybreaker)
* `breaker_resolver_state`: same (if you enable resolver breakers)
* `breaker_open_remaining_ms`: if blocked by cooldown override, how long remains
* `breaker_recorded`: `success|failure|none` (what we posted back to registry)
* `retry_after_s`: value honored for this request (if any)

This lets you build dashboards: opens/hour by host, time avoided, and whether Retry-After drives most opens.

---

# 8) Tests you must add

* **Open on consecutive failures**: drive 5× 503; assert subsequent `allow()` raises `BreakerOpenError` until `reset_timeout` elapses.
* **Honor Retry-After**: return 429 with `Retry-After: 4`; assert `allow()` is blocked ~4s (bounded by `retry_after_cap_s`).
* **Half-open probes**: after timeout, allow only `trial_calls[role]` concurrent probes; extra calls should be blocked until a success closes or a failure re-opens.
* **Neutral statuses**: 404/403 do not increment failure count.
* **Cache bypass**: requests served by Hishel do not change breaker counters.
* **Rolling window** (if enabled): inject failures spaced inside `window_s`; assert manual open via cooldown override.
* **Cross-process cooldown** (if using SQLite/Redis backend): open in one process; assert another process sees `allow()` blocked.

---

# 9) Rollout checklist

1. Add `pybreaker` to your environment.
2. Land `breakers.yaml`; add loader & validation.
3. Implement `BreakerRegistry` + optional `CooldownStore` (none/sqlite/redis).
4. Wire **pre-flight** and **post-update** in networking’s HTTP wrapper.
5. Add **NetworkBreakerListener**; register it on each breaker.
6. Add per-request telemetry fields; add `breaker_state_change` stream sink.
7. Delete legacy breaker code in pipeline; update imports.
8. Ship tests; run canary (enable on a subset of hosts first).
9. Expose ops controls:

   * `docstokg breaker show` (list state & cooldowns),
   * `docstokg breaker open <host> --seconds 300`,
   * `docstokg breaker close <host>`.

---

## “Best possible” extras (optional but powerful)

* **Adaptive reset**: if you consistently see `Retry-After` > `reset_timeout`, auto-raise the host’s reset timeout for the session (bounded by `retry_after_cap_s`), and log the change.
* **Per-resolver fallbacks**: when a host breaker is open, raise a soft signal to the resolver orchestration to **reorder resolvers** (e.g., try proxy/alternate source earlier).
* **Backoff décor**: annotate state transitions with the **limiter’s** last `rate_delay_ms` so you can tell when rate limiting already smoothed bursts (breaker opens should trend down after limiter launch).
* **Breaker transition sampling**: if you have many hosts, sample `breaker_before_call` events and always keep `state_change`, `failure`, `success`.

---

## TL;DR

* Centralize breaker policy in one YAML, overlay with env/CLI.
* One **BreakerRegistry** in networking: **pre-flight** deny, **post-update** on response/exception, with **Retry-After-aware** cooldowns.
* A **NetworkBreakerListener** emits clean transition telemetry.
* Delete pipeline breaker code; keep per-request telemetry fields so you can see opens, time saved, and tune quickly.

If you want, I can follow up with a tiny `breakers.py` skeleton (dataclasses for policy + registry API signatures) to jump-start the PR.
