Awesome — here’s a **repo-shaped, junior-dev-friendly** implementation plan to make **pyrate-limiter** first-class in `src/DocsToKG/ContentDownload`, aligned with your HTTPX + Hishel + Tenacity + pybreaker + url-normalize stack.

---

# Goals (what “good” looks like)

* **One** limiter layer in the **networking hub**; no sleeps or ad-hoc throttles anywhere else.
* **Multi-window**, **per-host** **and** **per-role** (`metadata`, `landing`, `artifact`) policies.
* **HEAD discount** (usually don’t count HEAD at all).
* **Bounded wait**: tiny for metadata (100–250 ms), larger for artifact (1–3 s); otherwise fail fast.
* **AIMD** (stretch): adapt rates down when 429s spike, recover slowly when stable.
* **Cache-smart placement**: tokens consumed **only** on real network calls (cache misses / revalidations), not on cache hits.
* **Ops-friendly**: YAML policy + env/CLI overrides; warm “effective policy” table at startup.
* **Telemetry**: rate-delay p95 per host+role; tokens acquired vs blocked; 429/503 trend dropping after rollout.
* **Global ceiling** to cap total in-flight requests.

---

# Config surface

Create `configs/networking/ratelimits.yaml`:

```yaml
version: 1
defaults:
  # two windows: per-second smoothing + per-hour politeness
  metadata:
    rates: ["10/SECOND", "5000/HOUR"]
    max_delay_ms: 200
    count_head: false
    max_concurrent: 100        # optional per-host per-role concurrency cap
  landing:
    rates: ["5/SECOND", "2000/HOUR"]
    max_delay_ms: 250
    count_head: false
    max_concurrent: 50
  artifact:
    rates: ["2/SECOND", "500/HOUR"]
    max_delay_ms: 2000
    count_head: false
    max_concurrent: 10

hosts:
  api.crossref.org:
    metadata: { rates: ["25/SECOND","10000/HOUR"], max_delay_ms: 150 }
  api.openalex.org:
    metadata: { rates: ["20/SECOND","8000/HOUR"], max_delay_ms: 150 }
  api.unpaywall.org: {}
  export.arxiv.org:
    metadata: { rates: ["1/3SECOND","1000/DAY"], max_delay_ms: 150 }
  web.archive.org:
    metadata: { rates: ["5/SECOND","300/MINUTE"], max_delay_ms: 200 }
    artifact: { rates: ["2/SECOND","120/MINUTE"], max_delay_ms: 3000 }

backend:
  kind: "memory"              # memory | multiprocess | sqlite | redis
  # used when kind != memory
  dsn: ""                     # e.g., /run/ratelimits.sqlite or redis://host:6379/3

aimd:                          # stretch, can start disabled
  enabled: false
  window_s: 60
  high_429_ratio: 0.05         # if 5%+ of requests are 429 in the last minute → decrease
  increase_step_pct: 5         # every window with low 429s, +5%
  decrease_step_pct: 20        # when high 429s, -20%
  min_multiplier: 0.3
  max_multiplier: 1.0

global:
  max_inflight: 500            # overall in-flight cap (quick win)
```

**Env/CLI overlays (examples)**

* `DOCSTOKG_RLIMITS_YAML=/path/to/ratelimits.yaml`
* Host/role override:
  `DOCSTOKG_RLIMIT__api.crossref.org__metadata=rates:30/SECOND+12000/HOUR,max_delay_ms:150,count_head:false`
* Backend: `DOCSTOKG_RLIMIT_BACKEND=sqlite`, `DOCSTOKG_RLIMIT_DSN=/run/ratelimits.sqlite`
* Global ceiling: `--max-inflight 500`
* AIMD: `--aimd enabled=true,high_429_ratio=0.08,decrease=25,increase=5`

---

# Placement in the stack (critical)

**Transport chain (cached client):**

```
HTTPX Client
  └─ Hishel CacheTransport         (RFC cache / revalidate / offline)
      └─ RateLimitedTransport (*)  ← acquire tokens ONLY on cache miss/revalidate
          └─ HTTPTransport
```

**Transport chain (raw client, PDFs):**

```
HTTPX Client
  └─ RateLimitedTransport (*)
      └─ HTTPTransport
```

> This guarantees cache hits **don’t burn tokens**, and only real network sends are throttled.

---

# Networking changes (what to add)

## 1) Limiter registry (one per process)

Create a `RateLimitRegistry` inside `networking.py` (or `ratelimit.py` imported by the hub):

* **Keying**: `(host: str, role: Literal["metadata","landing","artifact"])`
* **Policy**: loaded once from YAML/env/CLI → immutable map with:

  * `rates: List[RequestRate]`
  * `mode: "wait"|"raise"` (we’ll always “wait” but with **bounded** `max_delay_ms`)
  * `max_delay_ms: int`
  * `count_head: bool`
  * `max_concurrent: Optional[int]` (optional per host+role concurrency cap)
* **Limiter cache**: `Dict[(host, role), Limiter]` created with chosen **bucket backend**:

  * `InMemoryBucket` (now), or `MultiprocessBucket`/`SQLiteBucket`/`RedisBucket` when scaling beyond a single process/host.
* **AIMD multiplier** per (host, role) (stretch):

  * `eff_multiplier ∈ [min_multiplier, 1.0]` multiplies all configured rates; updated each window from telemetry (see AIMD section).

**Public API (used by transport wrapper)**

```python
class RateLimitRegistry:
    def acquire(self, host: str, role: str, method: str) -> RateAcquisition:
        """Acquire with bounded wait; may return (acquired=True, delay_ms=...) or raise RateLimitExceeded."""
    def record_429(self, host: str, role: str): ...
    def record_success(self, host: str, role: str): ...
    def tick_aimd(self): ...   # runs every window_s if enabled
```

`RateAcquisition` carries:

* `acquired: bool`
* `delay_ms: int` (actual wait time)
* `policy_max_delay_ms: int`

> **HEAD discount**: in `acquire()`, if `method=="HEAD"` and `count_head=False` → **skip** limiter; otherwise acquire normally.

**Per-role concurrency cap (optional)**:

* Maintain an internal `BoundedSemaphore` per (host, role). Acquire before calling the limiter; release after request completes. This prevents large batches of PDFs from saturating a host.

---

## 2) RateLimitedTransport (HTTPX transport wrapper)

Wrap the inner transport used by the clients; implement `handle_request(request)`:

1. Extract **host** from the canonical URL (use your `urls.canonical_for_request`/`canonical_host`).
2. Determine **role** from `request.extensions["role"]` (default `metadata`).
3. If `method=="HEAD"` and `count_head=False` → optionally **skip** limiter (depending on policy).
4. Call `registry.acquire(host, role, method)`. Two modes:

   * **Bounded wait**: block up to `max_delay_ms`. If granted → proceed.
   * If not granted within window → raise `RateLimitExceeded` (custom).
5. Forward to inner transport; on return:

   * If the response is **429/503**: call `registry.record_429(host, role)` (AIMD input).
   * Else: `registry.record_success(host, role)`.

**Tenacity integration**:

* Ensure your **retry predicate** does **not** retry `RateLimitExceeded` (it’s your own pacing decision). Real 429/503 **responses** remain retryable under Tenacity with `Retry-After` respected.

---

## 3) Global in-flight ceiling (quick win)

In the hub, add a process-wide `BoundedSemaphore(max_inflight)`:

* Acquire before any network call (both cached and raw paths).
* Release immediately after the call completes.
* Expose CLI/env `--max-inflight` / `DOCSTOKG_MAX_INFLIGHT`.

This prevents CPU/memory spikes and keeps the OS socket table sane.

---

# AIMD (stretch, but powerful)

**Goal**: When a host returns too many 429s in the last minute, **reduce** effective rates (Additive Decrease). As things stabilize (low 429 ratio), **increase** slowly (Additive Increase).

**How**

* For each (host, role), keep counters in a 60s sliding window (or EMA):

  * `calls_total`, `status_429_total`.
* Every `aimd.window_s` seconds (`tick_aimd()`), compute `ratio = 429 / max(1, calls_total)`.

  * If `ratio >= high_429_ratio` → `eff_multiplier = max(min_multiplier, eff_multiplier * (1 - decrease_step_pct/100.0))`
  * Else → `eff_multiplier = min(max_multiplier, eff_multiplier * (1 + increase_step_pct/100.0))`
* **Apply**: reconstruct the Limiter with scaled `RequestRate`s (e.g., `int(base_rps * eff_multiplier)`), swap it atomically in the registry. (Keep a small lock; the wrapper reads the current Limiter ref per call.)
* Telemetry: log multiplier changes as events: `aimd_adjust host=... role=... old=0.8 new=0.64 reason=high_429_ratio`.

**Notes**

* Start **disabled**; turn on per host after observing stable behavior.
* Keep multipliers **sticky** only for the run; don’t persist cross-run without strong justification.

---

# Telemetry (measure & tune)

For every **network** request (post-cache):

* `rate_delay_ms` (actual wait incurred before sending)
* `rate_acquired_total{host,role}`
* `rate_blocked_total{host,role}` (exceeded `max_delay_ms`)
* `rate_head_skipped_total{host}` (if HEAD bypassed)
* `rate_concurrency_wait_ms{host,role}` (if using per-role semaphores)
* 429/503 counters (already collected by the hub) → feed AIMD.

**Run summary**

* p50/p95 `rate_delay_ms` by host+role
* tokens acquired vs blocked
* 429/503 rate before vs after rollout
* top hosts by delay and by blocks
* AIMD multipliers (if enabled)

---

# Tests (must-add)

1. **Cache bypass correctness** (cached client):

   * First request (miss) → limiter acquires; second request (hit) → **no acquire** (delay=0).
2. **Multi-window enforcement**:

   * Configure `["2/SECOND","10/MINUTE"]`. Hammer with a loop; assert per-second smoothing and minute cap both applied (blocked counts grow when exceeded).
3. **Per-role separation**:

   * `metadata` and `artifact` have different policies; assert artifact waits up to 2–3 s, metadata caps ~200 ms.
4. **HEAD discount**:

   * With `count_head=false`, issue HEAD bursts; assert no rate delay; then GET bursts → delays appear.
5. **Bounded wait**:

   * Set `max_delay_ms=100`. Generate load; assert we raise `RateLimitExceeded` instead of waiting longer.
6. **429 feedback**:

   * Simulate server returning 429 repeatedly; ensure `record_429()` increments; if AIMD enabled, multipliers decrease on tick.
7. **Global ceiling**:

   * Set `max_inflight=3`; fire 10 parallel requests; assert at most 3 run concurrently (e.g., measure start/finish timing).
8. **Backend swap** (light):

   * Memory → SQLite/Redis only if you enable those; verify two processes share buckets (optional if you stay single-proc now).

---

# Implementation tasks (PR checklist)

1. **Loader** `ratelimits_loader.py`

   * Parse YAML/env/CLI into `RatePolicy` objects:

     * `RatePolicy(host, role): rates[List[str]], max_delay_ms, count_head, max_concurrent`
   * Normalize host keys (lowercase punycode, same helper as breakers).
2. **Registry** `RateLimitRegistry` (networking or `ratelimit.py`)

   * Build Limiter per (host, role) using chosen backend bucket.
   * Provide `acquire()/record_429()/record_success()/tick_aimd()`.
   * Optional semaphore map for `max_concurrent`.
3. **Transport wrapper** `RateLimitedTransport`

   * Insert under CacheTransport for cached client; directly above HTTPTransport for raw client.
   * Convert policy to behavior (HEAD skip, bounded wait, raise on exceed).
4. **Global ceiling** (hub)

   * Add `BoundedSemaphore`; acquire/release around every network send.
5. **Tenacity predicate**

   * Exclude `RateLimitExceeded` from retries; keep real 429/503 retryable.
6. **Telemetry**

   * Emit `rate_delay_ms`, acquired/blocked counters, 429/503; integrate with your run summary.
7. **Startup log**

   * Print a table: host, role, rates, max_delay_ms, count_head, backend, (aimd on/off).
8. **Tests**

   * Add the cases above.

---

# Operational defaults (starting point)

* **metadata**: `10/s + 5000/h`, `max_delay_ms=200`, `count_head=false`
* **landing**: `5/s + 2000/h`, `max_delay_ms=250`
* **artifact**: `2/s + 500/h`, `max_delay_ms=2000`
* **global max_inflight**: `500` (tune down if machine is smaller; you can also shape by CPU cores × 8–16).

---

# Guardrails & footguns

* Don’t place the limiter **above** the cache; you’ll burn tokens on cache hits.
* Don’t set transport‐level retries high; Tenacity handles backoff.
* Keep `max_delay_ms` **small** for metadata; if you find yourself waiting >250 ms often, your base rates are too low or your concurrency is too high — tune rates or global ceiling.
* For AIMD, prefer **hour-scoped** runs; reset multipliers each run to avoid stale throttles.

---

## Definition of Done

* All HTTP passes through **one** `RateLimitedTransport`.
* Tokens are only spent on cache misses/revalidations (not hits).
* HEAD is discounted according to policy.
* Per-host+role limits, bounded waits, and (optional) per-role concurrency caps are enforced.
* 429/503 rate **drops**; p95 `rate_delay_ms` is low and stable.
* Telemetry and startup logs clearly show effective policies.
* Global in-flight ceiling prevents runaway concurrency.

# High level architecture #

Absolutely—here’s a clean, drop-in **skeleton** you can paste into `src/DocsToKG/ContentDownload/ratelimit.py`. It contains:

* Compact **dataclasses** for policies (host/role) and runtime config
* A small **exception** type
* The **registry API** (constructor + `acquire/record_429/record_success/tick_aimd`)
* A **RateLimitedTransport** wrapper signature compatible with your HTTPX hub
* Protocols/hooks for telemetry and backends (placeholders)
* Plenty of TODO markers so a junior dev (or agent) can fill in the bodies safely

> This is intentionally light on implementation to keep the *surface* stable while you wire loaders and telemetry around it.

---

```python
# File: src/DocsToKG/ContentDownload/ratelimit.py
"""
Per-host *and* per-role rate limiting built on pyrate-limiter, with:
- multi-window limits (e.g., 10/SECOND + 5000/HOUR),
- HEAD discount (usually don't count HEAD),
- bounded wait per role (small for metadata, larger for artifact),
- optional per-role concurrency caps,
- optional AIMD dynamic tuning (stretch),
- registry + HTTPX transport wrapper (so cache hits don't burn tokens).

This module provides a stable API that the networking hub can rely on.
Actual policy values are loaded elsewhere (ratelimits_loader.py) and injected.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Tuple
from typing_extensions import Literal

import httpx

# Optional import: keep import-safe when running unit tests that don't install it yet.
try:
    from pyrate_limiter import Limiter, Rate, Duration  # type: ignore
except Exception:  # pragma: no cover
    Limiter = object  # type: ignore
    class Rate:  # type: ignore
        def __init__(self, limit: int, duration): ...
    class Duration:  # type: ignore
        SECOND = 1
        MINUTE = 60
        HOUR = 3600
        DAY = 86400


# ────────────────────────────────────────────────────────────────────────────────
# Public types & small helpers
# ────────────────────────────────────────────────────────────────────────────────

Role = Literal["metadata", "landing", "artifact"]


class RateLimitExceeded(RuntimeError):
    """Raised when a bounded wait elapses without acquiring capacity for (host, role)."""


# Telemetry hook used by both the registry and the transport
class RateTelemetrySink(Protocol):
    def emit_acquire(self, *, host: str, role: Role, delay_ms: int) -> None: ...
    def emit_block(self, *, host: str, role: Role, waited_ms: int, max_delay_ms: int) -> None: ...
    def emit_head_skipped(self, *, host: str) -> None: ...
    def emit_429(self, *, host: str, role: Role) -> None: ...
    def emit_success(self, *, host: str, role: Role) -> None: ...
    def emit_aimd_adjust(self, *, host: str, role: Role, old: float, new: float, reason: str) -> None: ...


# ────────────────────────────────────────────────────────────────────────────────
# Dataclasses: policy & runtime config
# ────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoleRates:
    """
    Role-specific rate policy:
    - 'rates' are human-readable strings like "10/SECOND", "5000/HOUR",
      which the loader will convert to pyrate-limiter Rate objects.
    - 'max_delay_ms' is the *bounded* wait per acquisition attempt.
    - 'count_head' toggles whether HEAD requests consume tokens.
    - 'max_concurrent' (optional) caps concurrent in-flight requests for (host, role).
    """
    rates: List[str] = field(default_factory=list)
    max_delay_ms: int = 200
    count_head: bool = False
    max_concurrent: Optional[int] = None


@dataclass(frozen=True)
class HostPolicy:
    """
    Per-host policy grouping role policies. Missing roles inherit from defaults.
    """
    metadata: Optional[RoleRates] = None
    landing: Optional[RoleRates] = None
    artifact: Optional[RoleRates] = None


@dataclass(frozen=True)
class BackendConfig:
    """
    Which pyrate-limiter bucket backend to use.
    kind: "memory" | "multiprocess" | "sqlite" | "redis"
    dsn: path/URL used when kind != memory (e.g., '/run/ratelimits.sqlite', 'redis://host:6379/3')
    """
    kind: str = "memory"
    dsn: str = ""


@dataclass(frozen=True)
class AIMDConfig:
    """
    Optional AIMD dynamic tuning. Start disabled, enable per run if desired.
    """
    enabled: bool = False
    window_s: int = 60
    high_429_ratio: float = 0.05
    increase_step_pct: int = 5
    decrease_step_pct: int = 20
    min_multiplier: float = 0.3
    max_multiplier: float = 1.0


@dataclass(frozen=True)
class RateConfig:
    """
    Fully-resolved runtime configuration:
    - defaults: role policies applied to any host that doesn't override them,
    - hosts: map 'host -> HostPolicy' (host keys must be lowercase punycode),
    - backend: bucket backend,
    - aimd: optional dynamic tuning,
    - global_max_inflight: overall in-flight ceiling (None = no ceiling).
    """
    defaults: Mapping[Role, RoleRates]
    hosts: Mapping[str, HostPolicy]
    backend: BackendConfig = BackendConfig()
    aimd: AIMDConfig = AIMDConfig()
    global_max_inflight: Optional[int] = 500


# ────────────────────────────────────────────────────────────────────────────────
# Registry (the single surface the networking hub calls)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class RateAcquisition:
    acquired: bool
    delay_ms: int
    policy_max_delay_ms: int


class RateLimitRegistry:
    """
    One process-global registry. The networking hub and transports use:
      - acquire(host, role, method)
      - record_429 / record_success
      - tick_aimd() (if enabled)

    Acquisitions are *bounded wait*: if capacity isn't available within 'max_delay_ms',
    a RateLimitExceeded is raised (so the caller can fail fast or reschedule).
    """

    def __init__(
        self,
        cfg: RateConfig,
        *,
        telemetry: Optional[RateTelemetrySink] = None,
        now: callable = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._tele = telemetry
        self._now = now

        self._lock = threading.RLock()
        self._limiters: Dict[Tuple[str, Role], Limiter] = {}        # (host, role) -> Limiter
        self._rates: Dict[Tuple[str, Role], List[Rate]] = {}         # parsed Rate objects
        self._sem: Dict[Tuple[str, Role], threading.BoundedSemaphore] = {}  # per-role concurrency caps
        self._aimd_mult: Dict[Tuple[str, Role], float] = {}          # effective multiplier ∈ [min,max]
        self._counters: Dict[Tuple[str, Role], Dict[str, int]] = {}  # rolling counters for AIMD

        # Global in-flight ceiling
        self._global_sem = (
            threading.BoundedSemaphore(cfg.global_max_inflight)
            if cfg.global_max_inflight and cfg.global_max_inflight > 0 else None
        )

        # TODO: parse cfg.defaults/cfg.hosts into _rates & initialize _limiters/_sem/_aimd_mult.
        # TODO: initialize rolling window counters for AIMD if enabled.

    # ----------------------- Public API -----------------------

    def acquire(self, *, host: str, role: Role, method: str) -> RateAcquisition:
        """
        Acquire capacity for (host, role). Applies:
          - HEAD discount (skip or count based on policy),
          - per-role bounded wait (policy.max_delay_ms),
          - optional per-role concurrency cap,
          - optional global in-flight ceiling.

        Returns: RateAcquisition(acquired=True, delay_ms=actual_wait, policy_max_delay_ms=cap)
        Raises: RateLimitExceeded if not acquired within cap.
        """
        # TODO: compute policy for (host, role), HEAD discount, bounded wait against the Limiter.
        # TODO: acquire per-role semaphore if configured, and global semaphore if configured.
        # TODO: emit telemetry via self._tele.emit_acquire or emit_block.
        raise NotImplementedError

    def record_429(self, *, host: str, role: Role) -> None:
        """
        Record a server 429 for AIMD and diagnostics.
        """
        # TODO: increment rolling counters for AIMD; noop if aimd.disabled.
        # TODO: emit telemetry via self._tele.emit_429.
        raise NotImplementedError

    def record_success(self, *, host: str, role: Role) -> None:
        """
        Record a non-429/503 success for AIMD recovery & diagnostics.
        """
        # TODO: increment success counter in rolling window (AIMD input); telemetry.
        raise NotImplementedError

    def tick_aimd(self) -> None:
        """
        Periodic adjustment (call every 'aimd.window_s' seconds when enabled).
        Computes 429 ratio per (host, role) and adjusts _aimd_mult, then rebuilds
        the Limiter with scaled rates atomically.
        """
        # TODO: implement AIMD window evaluation and limiter rebuild (with locks).
        # TODO: emit telemetry via self._tele.emit_aimd_adjust for each changed multiplier.
        raise NotImplementedError

    # ----------------------- Internals -----------------------

    def _effective_policy(self, host: str, role: Role) -> RoleRates:
        """
        Resolve RoleRates for (host, role): host override -> defaults.
        Host key must be lowercase punycode (loader enforces this).
        """
        # TODO: resolve HostPolicy from self._cfg.hosts.get(host), else defaults.
        raise NotImplementedError

    def _get_or_create_limiter(self, host: str, role: Role) -> Limiter:
        """
        Build or fetch a Limiter for (host, role), using the selected backend
        and current AIMD multiplier on the parsed base rates.
        """
        # TODO: lazy-create Limiter with Rate list derived from _parse_rates(...) × eff_multiplier.
        raise NotImplementedError

    def _get_or_create_semaphore(self, host: str, role: Role) -> Optional[threading.BoundedSemaphore]:
        """
        Return a per-(host, role) semaphore if 'max_concurrent' is configured; else None.
        """
        # TODO: instantiate BoundedSemaphore(policy.max_concurrent) if not present.
        raise NotImplementedError

    def _parse_rates(self, rates: List[str]) -> List[Rate]:
        """
        Convert ["10/SECOND","5000/HOUR","1/3SECOND"] → [Rate(10,SECOND), Rate(5000,HOUR), Rate(1, 3*SECOND)].
        Loader can pre-parse; this is a safety net if raw strings arrive here.
        """
        # TODO: implement parser; accept case-insensitive units: SECOND, MINUTE, HOUR, DAY, "<n>/SECOND", "1/3SECOND".
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────────────
# HTTPX transport wrapper
# ────────────────────────────────────────────────────────────────────────────────

class RateLimitedTransport(httpx.BaseTransport):
    """
    HTTPX transport that applies per-(host, role) rate limiting with bounded waits.
    Place this *under* Hishel's CacheTransport so cache hits don't consume tokens.

    Expected request extensions:
      - 'role': Literal["metadata","landing","artifact"] (default: "metadata")
      - optional 'canonical_host': lowercased punycode host (if the hub already derived it)

    Usage:
      inner = httpx.HTTPTransport(...)
      rlt = RateLimitedTransport(inner, registry=RateLimitRegistry(...))
      client = httpx.Client(transport=rlt, ...)
    """

    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        registry: RateLimitRegistry,
        telemetry: Optional[RateTelemetrySink] = None,
        now: callable = time.monotonic,
    ) -> None:
        self._inner = inner
        self._reg = registry
        self._tele = telemetry
        self._now = now

    # NOTE: httpx.BaseTransport in sync mode needs 'handle_request' implemented.
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """
        Apply rate limiting using (host, role, method), then delegate to the inner transport.
        Ensure per-role/global semaphores are released even on exceptions.
        """
        # TODO: derive role (default "metadata") from request.extensions.
        # TODO: derive host: use 'canonical_host' from extensions if present, else parse request.url.host.lower().
        # TODO: call self._reg.acquire(host=..., role=..., method=request.method)
        # TODO: call self._inner.handle_request(request) and capture response.
        # TODO: if response.status_code is 429 (and maybe 503), call self._reg.record_429(...); else record_success.
        # TODO: return response.
        raise NotImplementedError

    # Optionally, also implement 'close' and 'handle_async_request' (if you later add async clients):
    def close(self) -> None:  # pragma: no cover
        try:
            self._inner.close()
        except Exception:
            pass
```

---

## Notes on filling in the TODOs

* **Parsing rates**
  You can accept either `"10/SECOND"` or `"1/3SECOND"` (the latter means once every 3 seconds). A tiny parser that splits on `/` and maps unit tokens to `Duration` is enough.

* **Backends**
  Keep `memory` now; later switch to `multiprocess`, `sqlite`, or `redis` by building the matching `Bucket` and passing it to the `Limiter`. Wrap this in a tiny factory in the registry.

* **Bounded wait**
  `pyrate-limiter` supports both blocking and non-blocking acquisition. Implement *bounded* waits by looping with short sleeps (e.g., 25–50 ms) until either a token is granted or `max_delay_ms` elapses; then raise `RateLimitExceeded`. Keep the actual sleep configurable if you like.

* **HEAD discount**
  In `acquire()`: if `method == "HEAD"` and policy says `count_head=False`, skip acquisition **and** emit `emit_head_skipped(host=...)` for visibility.

* **Global ceiling**
  In `acquire()`, if `_global_sem` exists, acquire it before proceeding; release it in the transport after the inner call returns (and in an `except/finally` path).

* **AIMD**
  Use a small rolling window (simple ring buffer or counters with last-window rollover timestamp). On each `tick_aimd()`, compute the 429 ratio and scale the base rates by `eff_multiplier`. Rebuild the `(host, role)` Limiter atomically and emit `emit_aimd_adjust(...)`.

* **Telemetry**
  Emit `emit_acquire` with actual `delay_ms` for every grant; `emit_block` when you raise `RateLimitExceeded`; `emit_429/emit_success` on responses; `emit_head_skipped` when you bypass for HEAD.

---

## Minimal function signatures for loaders (optional)

If you want your loader to return strongly-typed policies that plug straight into the registry, aim for:

```python
# ratelimits_loader.py (not included here, just target signatures)

from DocsToKG.ContentDownload.ratelimit import RateConfig, RoleRates, HostPolicy, BackendConfig, AIMDConfig

def load_rate_config(
    yaml_path: str | None,
    *,
    env: Mapping[str, str],
    cli_host_role_overrides: Iterable[str] | None,  # e.g., ["api.crossref.org:metadata=rates:25/SECOND+10000/HOUR,max_delay_ms:150"]
    cli_backend: str | None,
    cli_backend_dsn: str | None,
    cli_global_max_inflight: int | None,
    cli_aimd_override: str | None,                  # "enabled:true,window:60,high_429_ratio:0.05,decrease:20,increase:5"
) -> RateConfig:
    ...
```

---

With these **dataclasses** and **method signatures** in place, your agent can implement the internals incrementally and wire the registry + transport into the networking hub without changing call sites later.
