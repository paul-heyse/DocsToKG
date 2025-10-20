Absolutely—here’s a complete, **agent-ready** plan (with drop-in code) to:

1. add a **RedisCooldownStore** for cross-host sharing,
2. ship an **Advisor + Auto-Tuner** that detects noisy hosts and recommends (or applies) breaker/rate changes,
3. wire a **CLI “advise/tune”** flow, and
4. highlight extra “best in class” optimizations inspired by pybreaker guidance.

I’ll keep this concrete so a junior dev (or agent) can implement it in one PR per section.

---

# 1) Cross-host sharing: `RedisCooldownStore`

> Paste as `src/DocsToKG/ContentDownload/redis_cooldown_store.py`

```python
from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Optional, Callable
from urllib.parse import urlparse

try:
    import redis  # pip install redis
except Exception as e:  # pragma: no cover
    raise RuntimeError("redis-py is required for RedisCooldownStore") from e

@dataclass
class RedisCooldownStore:
    """
    Cross-host cooldown store in Redis.
    Stores wall-clock deadlines and converts to monotonic when read.

    DSN examples:
      redis://localhost:6379/3
      rediss://user:pass@host:6380/1  (TLS)
    Keys:
      breaker:cooldown:<host> -> {"until_wall": <float>, "reason": "<str>"}
      TTL = ceil(until_wall - now_wall)
    """
    dsn: str
    key_prefix: str = "breaker:cooldown:"
    now_wall: Callable[[], float] = time.time
    now_mono: Callable[[], float] = time.monotonic
    _client: redis.Redis | None = None

    def __post_init__(self) -> None:
        url = urlparse(self.dsn)
        db = int((url.path or "/0").lstrip("/"))
        # Don't enable decode_responses; we store JSON bytes.
        self._client = redis.Redis(
            host=url.hostname, port=url.port or 6379, db=db,
            username=url.username, password=url.password,
            ssl=(url.scheme == "rediss"),
            socket_timeout=2.0, socket_connect_timeout=2.0,
        )

    def _key(self, host: str) -> str:
        return f"{self.key_prefix}{host}"

    # CooldownStore API (from breakers.py)
    def get_until(self, host: str) -> Optional[float]:
        raw = self._client.get(self._key(host))
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            until_wall = float(obj.get("until_wall", 0.0))
        except Exception:
            return None
        now_w, now_m = self.now_wall(), self.now_mono()
        if until_wall <= now_w:
            self.clear(host)
            return None
        return now_m + max(0.0, until_wall - now_w)

    def set_until(self, host: str, until_monotonic: float, reason: str) -> None:
        now_w, now_m = self.now_wall(), self.now_mono()
        until_wall = now_w + max(0.0, until_monotonic - now_m)
        ttl = max(1, int(round(until_wall - now_w)))
        obj = json.dumps({"until_wall": until_wall, "reason": str(reason)[:128]}).encode("utf-8")
        self._client.set(self._key(host), obj, ex=ttl)

    def clear(self, host: str) -> None:
        self._client.delete(self._key(host))
```

**When to use which store**

* Single machine / multiprocess → **SQLiteCooldownStore**
* Multiple machines (shared breaker) → **RedisCooldownStore**
* You can select the store from breakers.yaml/env (e.g., `cooldown_store.backend: sqlite|redis`).

---

# 2) “Noisy host” **Advisor & Auto-Tuner**

We’ll build a small analytics engine that:

* ingests your **telemetry SQLite** (per-request logs + breaker transitions),
* computes **windowed metrics** (429 ratio, 5xx bursts, half-open outcomes, open durations, Retry-After distribution),
* produces **HostAdvice** (suggested `fail_max`, `reset_timeout_s`, `success_threshold`, `trial_calls`, and rate-limit multipliers),
* optionally **applies** those changes (safe, bounded deltas) via the in-memory registries (BreakerRegistry, RateLimitRegistry) or writes a patch YAML for the next run.

> Paste as `src/DocsToKG/ContentDownload/breaker_advisor.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Tuple
import sqlite3, statistics, time

# --- Inputs: table names assume you use the sinks we discussed earlier ---
# per-request table: http_events (example schema—adapt to your actual table)
#   run_id, ts, host, role, status, from_cache, revalidated, rate_delay_ms, retry_after_s,
#   breaker_host_state, breaker_recorded ("success"|"failure"|"none"), ...
# breaker transitions table: breaker_transitions
#   ts, host, scope, old_state, new_state, reset_timeout_s

@dataclass
class HostMetrics:
    host: str
    window_s: int
    calls_total: int
    calls_cache_hits: int
    calls_net: int              # non-cache
    e429: int
    e5xx: int                   # 500+504+… (exclude 503 only if you want separate)
    e503: int
    timeouts: int               # network exceptions mapped to pseudo-code or stored elsewhere
    retry_after_samples: list[float]
    open_events: int
    open_durations_s: list[float]
    half_open_success_trials: int
    half_open_fail_trials: int
    max_consecutive_failures: int

@dataclass
class HostAdvice:
    host: str
    # Circuit breaker knobs
    suggest_fail_max: Optional[int] = None
    suggest_reset_timeout_s: Optional[int] = None
    suggest_success_threshold: Optional[int] = None
    suggest_trial_calls_metadata: Optional[int] = None
    suggest_trial_calls_artifact: Optional[int] = None
    # Rate limiting knobs (optional)
    suggest_metadata_rps_multiplier: Optional[float] = None
    suggest_artifact_rps_multiplier: Optional[float] = None
    # Reasoning snippets (short)
    reasons: list[str] = None

DEFAULTS = dict(
    min_fail_max=2,
    max_fail_max=10,
    min_reset_timeout=15,
    max_reset_timeout=600,
    min_success_threshold=1,
    max_success_threshold=3,
)

class BreakerAdvisor:
    """
    Reads telemetry SQLite, computes metrics in a sliding window, and proposes breaker/rate tuning.
    """
    def __init__(self, db_path: str, window_s: int = 600) -> None:
        self.db_path = db_path
        self.window_s = window_s

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def read_metrics(self, now: float | None = None) -> dict[str, HostMetrics]:
        now = now or time.time()
        since = now - self.window_s
        m: dict[str, HostMetrics] = {}

        with self._conn() as cx:
            # Aggregate per-request
            rows = cx.execute("""
                SELECT host,
                       COUNT(*) FILTER (WHERE from_cache=1) as cache_hits,
                       COUNT(*) as calls_total,
                       COUNT(*) FILTER (WHERE status=429) as e429,
                       COUNT(*) FILTER (WHERE status BETWEEN 500 AND 599) as e5xx,
                       COUNT(*) FILTER (WHERE status=503) as e503,
                       AVG(retry_after_s) FILTER (WHERE retry_after_s IS NOT NULL) as ra_avg
                FROM http_events
                WHERE ts >= ? AND (role='metadata' OR role='landing' OR role='artifact')
                GROUP BY host
            """, (since,)).fetchall()

            # Breaker transitions and open durations (approx)
            trans = cx.execute("""
                SELECT host, ts, old_state, new_state, reset_timeout_s
                FROM breaker_transitions WHERE ts >= ?""", (since,)).fetchall()

            # Half-open trials (approx: count events immediately after open)
            half = cx.execute("""
                SELECT host,
                       SUM(CASE WHEN breaker_recorded='success' THEN 1 ELSE 0 END) as ok,
                       SUM(CASE WHEN breaker_recorded='failure' THEN 1 ELSE 0 END) as ko
                  FROM http_events
                 WHERE ts >= ? AND breaker_host_state like '%half_open%'
              GROUP BY host
            """, (since,)).fetchall()

        # Seed metrics object
        hx: dict[str, HostMetrics] = {}
        for host, cache_hits, calls_total, e429, e5xx, e503, ra_avg in rows:
            hx[host] = HostMetrics(
                host=host, window_s=self.window_s,
                calls_total=calls_total or 0,
                calls_cache_hits=cache_hits or 0,
                calls_net=(calls_total or 0) - (cache_hits or 0),
                e429=e429 or 0, e5xx=e5xx or 0, e503=e503 or 0, timeouts=0,
                retry_after_samples=[] if ra_avg is None else [float(ra_avg)],
                open_events=0, open_durations_s=[],
                half_open_success_trials=0, half_open_fail_trials=0,
                max_consecutive_failures=0
            )
        # Transitions → open durations
        open_started: dict[str, float] = {}
        for host, ts, old_s, new_s, reset_s in trans:
            if new_s.endswith("OPEN"):
                open_started[host] = ts
                hx.setdefault(host, HostMetrics(host, self.window_s, 0,0,0,0,0,0,0,[],0,[],0,0,0))
                hx[host].open_events += 1
            elif old_s.endswith("OPEN"):  # state_change OPEN -> HALF_OPEN/CLOSED
                start = open_started.pop(host, None)
                if start:
                    hx[host].open_durations_s.append(max(0.0, ts - start))
        # Half-open trial outcomes
        for host, ok, ko in half:
            H = hx.setdefault(host, HostMetrics(host, self.window_s, 0,0,0,0,0,0,0,[],0,[],0,0,0))
            H.half_open_success_trials += ok or 0
            H.half_open_fail_trials    += ko or 0

        return hx

    def advise(self, metrics: dict[str, HostMetrics]) -> dict[str, HostAdvice]:
        advice: dict[str, HostAdvice] = {}
        for host, H in metrics.items():
            A = HostAdvice(host=host, reasons=[])
            if H.calls_net <= 0:
                advice[host] = A
                continue

            # --- (1) 429 handling: prefer rate-limiter changes over breaker tuning
            r429 = H.e429 / max(1, H.calls_net)
            if r429 >= 0.05:   # 5%+ 429s in window
                A.suggest_metadata_rps_multiplier = 0.8   # -20% to start
                A.reasons.append(f"High 429 ratio {r429:.1%}: reduce metadata RPS 20%")

            # --- (2) Estimate unhealthy period and tune reset_timeout
            # Prefer Retry-After samples; else median open duration; else leave None.
            est_cool_s = None
            if H.retry_after_samples:
                est_cool_s = min(900, max(15, int(statistics.median(H.retry_after_samples))))
            elif H.open_durations_s:
                est_cool_s = min(600, max(15, int(statistics.median(H.open_durations_s))))
            if est_cool_s:
                A.suggest_reset_timeout_s = est_cool_s
                A.reasons.append(f"Reset timeout → ~{est_cool_s}s (based on Retry-After/open durations)")

            # --- (3) fail_max based on burstiness of 5xx/timeout
            # Heuristic: trip sooner than typical max run length.
            # In absence of explicit run-length logs, approximate from open events: if many opens, reduce fail_max.
            if H.open_events >= 3:   # multiple opens in window
                A.suggest_fail_max = 3
                A.reasons.append("Multiple breaker opens observed: suggest fail_max=3")

            # --- (4) half-open outcomes → success_threshold & trial_calls
            total_trials = H.half_open_success_trials + H.half_open_fail_trials
            if total_trials >= 2 and H.half_open_fail_trials / total_trials >= 0.5:
                # Many failures in half-open → require more successes to close, and fewer probes at once
                A.suggest_success_threshold = 2
                A.suggest_trial_calls_metadata = 1
                A.reasons.append("Half-open failures ≥50%: raise success_threshold to 2, trial_calls(metadata)=1")

            advice[host] = A
        return advice
```

### Applying advice (“observe”, “suggest”, “enforce”)

> Paste as `src/DocsToKG/ContentDownload/breaker_autotune.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple

from DocsToKG.ContentDownload.breaker_advisor import BreakerAdvisor
from DocsToKG.ContentDownload.breakers import BreakerRegistry, BreakerPolicy, BreakerRolePolicy, RequestRole

@dataclass
class AutoTunePlan:
    host: str
    changes: list[str]  # human-readable changes

class BreakerAutoTuner:
    """
    Applies safe, bounded adjustments to live registries or outputs a plan.
    Modes:
      - observe: compute metrics only
      - suggest: produce a plan (no changes)
      - enforce: apply in-memory changes (rebuild breaker objects with new policy)
    Safety:
      - Clamp fail_max ∈ [2, 10]
      - Clamp reset_timeout_s ∈ [15, 600]
      - Clamp success_threshold ∈ [1, 3]
      - No more than ±25% rate multiplier change per tick
    """
    def __init__(self, registry: BreakerRegistry, rate_registry=None, clamp=True) -> None:
        self._br = registry
        self._rr = rate_registry
        self._clamp = clamp

    def suggest(self, advisor: BreakerAdvisor, run_id: str | None = None) -> list[AutoTunePlan]:
        M = advisor.read_metrics()
        A = advisor.advise(M)
        plans: list[AutoTunePlan] = []
        for host, adv in A.items():
            changes = []
            if adv.suggest_fail_max:
                changes.append(f"fail_max → {adv.suggest_fail_max}")
            if adv.suggest_reset_timeout_s:
                changes.append(f"reset_timeout_s → {adv.suggest_reset_timeout_s}")
            if adv.suggest_success_threshold:
                changes.append(f"success_threshold → {adv.suggest_success_threshold}")
            if adv.suggest_trial_calls_metadata:
                changes.append(f"trial_calls(metadata) → {adv.suggest_trial_calls_metadata}")
            if adv.suggest_metadata_rps_multiplier:
                changes.append(f"metadata RPS × {adv.suggest_metadata_rps_multiplier:.2f}")
            if changes:
                changes += [f"reason: {r}" for r in (adv.reasons or [])]
                plans.append(AutoTunePlan(host=host, changes=changes))
        return plans

    def enforce(self, advisor: BreakerAdvisor) -> list[AutoTunePlan]:
        plans = self.suggest(advisor)
        # Apply changes in-memory: reconstruct pybreaker with new policy
        for plan in plans:
            host = plan.host
            # NOTE: For simplicity, call a (to-be-implemented) method on registry:
            #   registry.update_host_policy(host, fail_max=?, reset_timeout_s=?, success_threshold=?, trial_calls=?)
            # This method should rebuild the host breaker safely (closing old instance).
            # Similarly, for rate-limiter, scale multipliers via RateLimitRegistry.
            pass
        return plans
```

**Important**: pybreaker doesn’t expose setters for `fail_max` & `reset_timeout`; you’ll implement `BreakerRegistry.update_host_policy(...)` that **rebuilds** the `CircuitBreaker` for that host using the new `BreakerPolicy`, transfers only minimal context (e.g., keep cooldown overrides), and replaces the instance under lock. That keeps it safe and deterministic.

---

# 3) CLI: **advise** and **tune**

> Extend your CLI with a new command group

```python
# File: src/DocsToKG/ContentDownload/cli_breaker_advisor.py
from __future__ import annotations
import argparse, os
from pathlib import Path
from DocsToKG.ContentDownload.breaker_advisor import BreakerAdvisor
from DocsToKG.ContentDownload.breaker_autotune import BreakerAutoTuner
from DocsToKG.ContentDownload.breakers_loader import load_breaker_config
from DocsToKG.ContentDownload.breakers import BreakerRegistry
from DocsToKG.ContentDownload.sqlite_cooldown_store import SQLiteCooldownStore

def install_breaker_advisor_cli(subparsers, make_registry, telemetry_db_path: Path) -> None:
    p = subparsers.add_parser("breaker-advise", help="Analyze noisy hosts and propose tuning")
    p.add_argument("--window-s", type=int, default=600, help="Analysis window in seconds (default 600)")
    p.add_argument("--enforce", action="store_true", help="Apply safe adjustments in-memory")
    p.set_defaults(func=_cmd_advise, make_registry=make_registry, telemetry_db_path=telemetry_db_path)

def _cmd_advise(args: argparse.Namespace) -> int:
    reg, _ = args.make_registry()
    advisor = BreakerAdvisor(str(args.telemetry_db_path), window_s=args.window_s)
    tuner = BreakerAutoTuner(registry=reg)
    plans = tuner.enforce(advisor) if args.enforce else tuner.suggest(advisor)
    if not plans:
        print("No changes suggested.")
        return 0
    for plan in plans:
        print(f"[{plan.host}]")
        for c in plan.changes:
            print(f"  - {c}")
    return 0
```

Wire this with your existing `argparse` (similar to the earlier `breaker` CLI).

---

# 4) Heuristics you can deploy **now** (and tune later)

These are simple, explainable rules that work well in practice; all are straightforward to implement in the `BreakerAdvisor`:

1. **Retry-After-aware reset**

   * `reset_timeout_s` ← **median** of `Retry-After` (429/503) capped to `[15, 900]`.
   * If no headers, fallback to **median** breaker **open duration**.
   * Goal: half-open when the origin says it’s likely healthy.

2. **Trip threshold (fail_max)**

   * If you observe **multiple opens** within the window or high 5xx bursts, trip **earlier**:

     * `fail_max = clamp(3, q95_burst_len)` if you track run lengths; otherwise default `3`.
   * If opens are rare and most failures are isolated, you can relax to 4–5.

3. **Half-open policy**

   * If ≥50% of **half-open** trials fail, set `success_threshold=2` and `trial_calls(metadata)=1` (optionally `artifact)=1`).
   * If half-open success rate ≥80% across multiple periods, you can reduce `success_threshold` back to 1.

4. **Rate limiter over breaker**

   * High 429 ratio => nudge **rate limiter** first (AIMD decrease), not breaker.
   * Re-increase slowly once 429 ratio stays below threshold (`increase_step_pct` ≈ +5%/min).

5. **Flapping detection**

   * If you see **opens → immediate close → open** ping-pong for a host, your `reset_timeout` is too short (increase by +50% until flapping stops) **or** raise `success_threshold`.

6. **Exclude business errors**

   * Don’t count 401/403/404/410/451 as breaker failures. Keep that in your `is_failure_for_breaker` helper.

7. **Naming & listener**

   * Give breakers a friendly `name` so telemetry is human-readable. Attach listener; record transitions.

> If you later want smarter detection, you can add **EWMA** on 5xx and 429 rates, or basic **change-point** detection (e.g., with `ruptures`)—but the above heuristics will get you 90% of the value with very little code.

---

# 5) Are there libraries with “built-in intelligence”?

* **pybreaker** itself doesn’t auto-tune; it provides the state machine + listeners + storages.
* **resilience4j** (Java) and **Envoy/Istio outlier detection** (service mesh) have advanced ejection/outlier logic, but they aren’t Python libraries you can embed here.
* In Python land, there isn’t a widely adopted library that automatically tunes breakers/rates using telemetry.

**Conclusion:** the logic is **clear enough** and **practical** to implement in Python, exactly as outlined above: deterministic heuristics + optional AIMD + small SQL queries over your telemetry. Start simple and make the system self-explanatory.

---

# 6) “Best in class” (from pybreaker doc & practice)

* **Use `success_threshold`** > 1 when half-open failures are common.
* **Attach listeners** and **name** every breaker with its host to simplify ops.
* **Use a shared store** for breakers themselves **only** if you truly need shared state (Redis storage). We recommend **keeping pybreaker local** and using **cross-process cooldowns** (SQLite/Redis from this plan) for Retry-After/rolling-window open—simpler and safer.
* **Keep breakers as singletons** per host; wrap them **once** at process creation; don’t instantiate on every call.
* **Do not** count client-side or business logic errors as breaker failures; your classification helper already enforces that.
* Always **log transition events** and expose a short **`breaker show`** CLI so humans can reason about current state quickly.

---

## Acceptance checklist

* RedisCooldownStore works (multi-host) and you can switch between it and SQLite via config.
* `breaker-advise` prints actionable suggestions; `--enforce` applies safe changes (or at least prints what it would change).
* Breaker opens per hour **decrease** after rate limiter AIMD kicks in; success-after-open **increases** as reset_timeout aligns to Retry-After signals.
* No flapping: state transitions “OPEN → HALF_OPEN → CLOSED” cluster around unhealthy windows, not constantly.

If you’d like, I can also stub `BreakerRegistry.update_host_policy(...)` (safe rebuild) and a couple of SQL queries that compute **burst run lengths** exactly (instead of our approximate “many opens” heuristic), so you can refine `fail_max` based on actual consecutive failure runs.
