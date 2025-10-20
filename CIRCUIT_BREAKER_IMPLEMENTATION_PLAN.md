# Circuit Breaker Implementation Plan

## DocsToKG.ContentDownload — Pybreaker Integration (Complete Scope)

**Date:** October 20, 2025
**Status:** Planning Phase
**Scope:** 12 phases, ~1,500–2,000 LOC, 4–6 week timeline
**Risk:** LOW (modular, well-specified, all integration points identified)

---

## Overview

This plan implements a **production-ready circuit breaker system** for ContentDownload using **pybreaker 1.4.1**. The system:

- Detects unhealthy hosts/resolvers and short-circuits requests deterministically
- Honors `Retry-After` headers with server-controlled cooldowns
- Supports rolling-window failure detection for burst patterns
- Shares breaker state across processes via pluggable backends (SQLite, Redis, in-memory)
- Emits structured telemetry for observability and auto-tuning
- Provides CLI ops tools (`show`, `open`, `close`, `advise`)
- Includes automatic host advisor that recommends or applies safe tuning

All work aligns with the **no-install policy** (uses existing `.venv`).

---

## Architecture Overview

### Key Components

```
┌─────────────────┐
│   networking    │ (request_with_retries)
│     layer       │ • Pre-flight: registry.allow()
│                 │ • Post-response: registry.on_success/on_failure()
└────────┬────────┘
         │ uses
         ▼
┌─────────────────────────────────────┐
│   BreakerRegistry (breakers.py)     │
│  • Per-host & per-resolver breakers │
│  • Cooldown overrides (Retry-After) │
│  • Rolling window manual-open       │
│  • Half-open trial limiting         │
└────┬───────────────────┬────────────┘
     │ uses              │ uses
     ▼                   ▼
┌─────────────┐    ┌─────────────────────┐
│  pybreaker  │    │   CooldownStore API │
│ CircuitBrk. │    │ • InMemory (default)│
│ (per-host)  │    │ • SQLite (local)    │
│             │    │ • Redis (multi-host)│
└─────────────┘    └─────────────────────┘
     │ emits
     ▼
┌──────────────────────────────────────┐
│  NetworkBreakerListener              │
│  (telemetry events → sink)           │
│  • before_call, success, failure,    │
│    state_change → structured records │
└──────────────────────────────────────┘
```

### Integration Points

1. **Networking layer** (`request_with_retries`): Pre-flight & post-response checks
2. **Runner/Pipeline**: Registry instantiation, config loading
3. **CLI**: Show state, force open/close, advise tuning
4. **Telemetry**: Breaker transitions table, metrics aggregation

---

## Phase Details

### Phase 1: Cross-Process Cooldown Stores

#### 1a. SQLiteCooldownStore (`sqlite_cooldown_store.py`)

**File:** `src/DocsToKG/ContentDownload/sqlite_cooldown_store.py` (151 lines)

**Key Design:**

- Wall-clock storage (UTC epoch) with monotonic-time conversion at runtime
- Prevents issues with clock drift across processes
- PRAGMA journal_mode=WAL for concurrent access safety
- Uses existing `locks.sqlite_lock()` for cross-process write serialization

**API:**

```python
class SQLiteCooldownStore:
    db_path: Path
    def get_until(host: str) -> Optional[float]: ...  # returns monotonic deadline
    def set_until(host: str, until_monotonic: float, reason: str) -> None: ...
    def clear(host: str) -> None: ...
    def prune_expired() -> int: ...  # cleanup old entries
```

**Schema:**

```sql
CREATE TABLE breaker_cooldowns (
    host TEXT PRIMARY KEY,
    until_wall REAL NOT NULL,
    reason TEXT,
    updated_at REAL NOT NULL
);
```

**Test scenarios:**

- Multi-process write safety (lock contention)
- Monotonic deadline calculation (wall→monotonic conversion)
- Expiry detection & pruning

#### 1b. RedisCooldownStore (`redis_cooldown_store.py`)

**File:** `src/DocsToKG/ContentDownload/redis_cooldown_store.py` (86 lines)

**Key Design:**

- JSON serialization: `{"until_wall": float, "reason": str}`
- TTL auto-expiry: set(key, value, ex=ttl)
- Same wall↔monotonic protocol as SQLite
- Optional but recommended for multi-host deployments

**API:** Identical to SQLiteCooldownStore

**Wiring:**

- Configured via breakers.yaml: `cooldown_store.backend: redis://localhost:6379/3`
- DSN parsing with fallback ports/DBs

---

### Phase 2: Complete breakers.py

**File:** `src/DocsToKG/ContentDownload/breakers.py` (existing, needs completion)

**Remaining work:**

- ✅ `BreakerRegistry.allow()` – pre-flight checks (already complete)
- ✅ `BreakerRegistry.on_failure()` – failure recording (already complete)
- ✅ `BreakerRegistry.on_success()` – success & cleanup (already complete)
- ⚠️ Add `success_threshold` support to pybreaker usage
  - pybreaker supports `success_threshold` in `__init__`
  - May not expose counters; track manually if needed for telemetry
  - Example: allow N consecutive successes before closing

**pybreaker Usage (from library ref):**

- CircuitBreaker states: CLOSED, OPEN, HALF_OPEN
- Counters: `fail_counter`, `success_counter` (read-only)
- Methods: `call_failed()`, `call_success()`
- Listeners: inherit CircuitBreakerListener for state_change/success/failure/before_call

**TODOs to clear:**

- Line 270: "Add SQLiteCooldownStore / RedisCooldownStore skeletons" → deferred to Phase 1 (cross-file modules)
- Verify `throw_new_error_on_trip=False` behavior (keep original exceptions for proper classification)

---

### Phase 3: Networking Integration

**File:** `src/DocsToKG/ContentDownload/networking.py` (modify request_with_retries)

**Changes:**

- Import BreakerRegistry, BreakerOpenError, RequestRole
- Add optional `breaker_registry` parameter to request_with_retries
- Pre-flight: `registry.allow(host, role=RequestRole.METADATA)`
  - If raises BreakerOpenError: short-circuit with `reason='breaker_open'`, log cooldown_ms
  - Do NOT enter Tenacity; fail fast with structured error
- Post-response: classify and call `registry.on_success()` or `registry.on_failure()`
  - Success: 2xx/3xx responses
  - Failure: 429/5xx/408 responses, network timeouts/exceptions
  - Neutral: 401/403/404/410/451 (don't update breaker)
  - Cached responses: skip breaker updates (no network happened)

**Retry-After parsing:**

- Already implemented via `parse_retry_after_header()` (line 150–200)
- Pass parsed `retry_after_s` to `registry.on_failure(status=429, retry_after_s=...)`

**Error telemetry:**

- Add `breaker_meta` dict to request metadata
- Capture: state before/after, cooldown_remaining_ms, half_open status, probe count

**Integration flow (pseudocode):**

```python
def request_with_retries(..., breaker_registry=None, ...):
    # Pre-flight
    if breaker_registry:
        registry.allow(host, role=RequestRole.METADATA, resolver=resolver)

    # Build & send (via Tenacity, caching, etc.)
    try:
        response = ... (send via HTTP client with retries)
    except BreakerOpenError:
        raise  # Don't retry; fail fast
    except Exception as e:
        # Network error → on_failure
        registry.on_failure(..., exception=e, ...)
        raise

    # Post-response (HTTP 200 but check status)
    if response.status_code in {429, 500, 502, 503, 504}:
        registry.on_failure(..., status=response.status_code, retry_after_s=...)
    elif response.status_code in {401, 403, 404, 410}:
        # Neutral; don't update breaker
        pass
    else:
        registry.on_success(...)

    return response
```

**Test points:**

- Pre-flight deny (BreakerOpenError raised, request not sent)
- Retry-After honored (cooldown set to ≤ cap)
- On-success clears overrides
- Cached responses skip breaker updates
- Half-open trial limiting (only 1 per role by default)

---

### Phase 4: CLI Commands

#### 4a. `cli_breakers.py`

**File:** `src/DocsToKG/ContentDownload/cli_breakers.py` (83 lines)

**Subcommands:**

1. **`docstokg breaker show [--host HOST]`**
   - List all known hosts with state, cooldown remaining, half-open budget
   - Output: tabular (HOST | STATE | COOLDOWN_REMAIN_MS)

2. **`docstokg breaker open <host> --seconds N [--reason REASON]`**
   - Force-open host for N seconds via cooldown override
   - Sets: `cooldowns.set_until(host, now + N, reason="cli-open:...")`

3. **`docstokg breaker close <host>`**
   - Clear cooldown, reset failure counters
   - Calls: `cooldowns.clear(host)` + `registry.on_success(host, ...)`

#### 4b. `cli_breaker_advisor.py`

**File:** `src/DocsToKG/ContentDownload/cli_breaker_advisor.py` (52 lines)

**Subcommand:**

- **`docstokg breaker-advise [--window-s WINDOW] [--enforce]`**
  - Analyze telemetry window (default 600s)
  - Print suggested tuning (or apply if `--enforce`)
  - Output: per-host suggestions (fail_max, reset_timeout_s, rate multipliers)

---

### Phase 5: Breaker Advisor & Auto-Tuner

#### 5a. `breaker_advisor.py`

**File:** `src/DocsToKG/ContentDownload/breaker_advisor.py` (283 lines)

**Dataclasses:**

- `HostMetrics`: aggregated stats per host over a time window (429 rate, 5xx count, Retry-After samples, open duration stats)
- `HostAdvice`: recommendations (fail_max, reset_timeout_s, trial_calls, rate multipliers)

**Core logic:**

1. **read_metrics(now=None) → dict[str, HostMetrics]**
   - Query SQLite telemetry tables: http_events, breaker_transitions
   - Aggregate: 429 rate, 5xx bursts, half-open success/fail outcomes, open durations
   - Returns dict keyed by host

2. **advise(metrics) → dict[str, HostAdvice]**
   - Heuristics:
     - **High 429 ratio (≥5%):** suggest reducing metadata RPS by 20% (rate limiter over breaker)
     - **Retry-After median:** set reset_timeout_s ≈ Retry-After value (capped 15–900s)
     - **Multiple opens in window:** suggest lower fail_max (3–5 instead of 5)
     - **Half-open failures ≥50%:** require 2 successes to close, reduce trial_calls to 1
   - Returns dict of recommendations per host

#### 5b. `breaker_autotune.py`

**File:** `src/DocsToKG/ContentDownload/breaker_autotune.py` (52 lines)

**Classes:**

- `AutoTunePlan`: per-host plan (changes list, reasoning)
- `BreakerAutoTuner(registry, rate_registry, clamp=True)`

**Methods:**

1. **suggest(advisor) → list[AutoTunePlan]**
   - Generate plans (no changes applied)
   - Safe bounds: fail_max ∈ [2,10], reset_timeout_s ∈ [15,600], success_threshold ∈ [1,3]

2. **enforce(advisor) → list[AutoTunePlan]**
   - Apply suggestions in-memory (rebuilds pybreaker for affected hosts)
   - Calls: `registry.update_host_policy(host, fail_max=?, ...)`
   - Returns plans that were applied

---

### Phase 6: CLI Integration

**File:** `src/DocsToKG/ContentDownload/cli.py` (modify)

**Changes:**

- Import `install_breaker_cli`, `install_breaker_advisor_cli`
- Create a factory function that constructs BreakerRegistry + known hosts list
- Call `install_breaker_cli(subparsers, make_registry)` before main parser
- Call `install_breaker_advisor_cli(subparsers, make_registry, telemetry_db_path)` before main parser

**Factory signature:**

```python
def _make_registry() -> Tuple[BreakerRegistry, Iterable[str]]:
    cfg = load_breaker_config(env=os.environ, ...)
    store = SQLiteCooldownStore(Path(run_dir) / "telemetry/breakers.sqlite")
    reg = BreakerRegistry(cfg, cooldown_store=store, listener_factory=make_listener)
    known_hosts = sorted(cfg.hosts.keys())
    return reg, known_hosts
```

---

### Phase 7: Telemetry Integration

**File:** `src/DocsToKG/ContentDownload/telemetry.py` (modify)

**Changes:**

- Add `BreakerTransitionRecord` dataclass (event_type, ts, run_id, host, scope, old_state, new_state, reset_timeout_s)
- Add `breaker_transitions` table to manifest SQLite schema (SQLITE_SCHEMA_VERSION bump to 5)
- Wire NetworkBreakerListener into telemetry sink
- Aggregate breaker metrics in `summary.json`: opens/hour per host, time-saved, success-after-open rate

**Telemetry events:**

- `breaker_before_call`: called before each attempt
- `breaker_success`: called after successful response
- `breaker_failure`: called after failure (status or exception)
- `breaker_state_change`: CLOSED→OPEN, OPEN→HALF_OPEN, etc.

**Summary additions:**

```json
{
  "breakers": {
    "api.crossref.org": {
      "opens_per_hour": 2.5,
      "time_saved_ms": 15000,
      "success_after_open_rate": 0.8,
      "open_reason_mix": {
        "retry-after": 60,
        "rolling-window": 40
      }
    }
  }
}
```

---

### Phase 8: Config YAML

**File:** `src/DocsToKG/ContentDownload/config/breakers.yaml` (new)

**Template:**

```yaml
# Breaker policy defaults
defaults:
  fail_max: 5              # Trip after 5 consecutive failures
  reset_timeout_s: 60      # Try recovery after 60s
  retry_after_cap_s: 900   # Cap Retry-After at 15 min
  classify:
    failure_statuses: [429, 500, 502, 503, 504, 408]
    neutral_statuses: [401, 403, 404, 410, 451]
  roles:
    metadata:              # Per-role overrides
      fail_max: 5
      trial_calls: 1       # Only 1 probe in half-open
    artifact:
      fail_max: 4
      trial_calls: 2
  half_open:
    jitter_ms: 150

# Host-specific policies
hosts:
  api.crossref.org:
    fail_max: 3             # More sensitive
    reset_timeout_s: 45
    roles:
      metadata:
        trial_calls: 1

  export.arxiv.org:
    fail_max: 2
    reset_timeout_s: 120
    retry_after_cap_s: 600

# Resolver-specific policies
resolvers:
  crossref:
    fail_max: 4
    reset_timeout_s: 30
  unpaywall:
    fail_max: 5
    reset_timeout_s: 60

# Advanced: rolling window for burst detection
advanced:
  rolling:
    enabled: true
    window_s: 30            # 30-second window
    threshold_failures: 6   # 6+ failures → open
    cooldown_s: 60          # Stay open for 60s
```

---

### Phase 9: Comprehensive Tests

#### 9a. `test_breakers_core.py` (200–250 lines)

**Core behavior:**

- Consecutive failures trip breaker
- Retry-After respected (capped)
- Rolling window manual-open
- Half-open trial limiting per role
- Neutral statuses don't count
- Cache bypass (no breaker updates)
- Cross-process cooldown (SQLite/Redis)
- CLI show/open/close behavior

#### 9b. `test_breakers_networking.py` (150–200 lines)

**Integration:**

- Pre-flight deny raises BreakerOpenError
- Request not sent when breaker open
- On-success clears cooldown
- On-failure sets cooldown for Retry-After
- Half-open probe jitter applied (if configured)

#### 9c. `test_cli_breakers.py` (100–150 lines)

**CLI:**

- `breaker show` lists all hosts with state
- `--host` filter works
- `breaker open <host> --seconds 10` sets cooldown
- `breaker close <host>` clears state
- Invalid args rejected

#### 9d. `test_breaker_advisor.py` (100–150 lines)

**Advisor:**

- Metrics aggregation from telemetry window
- Heuristic advice generation
- Safe bounds (fail_max ∈ [2,10], etc.)
- Auto-tuner apply (safe rebuild)

---

### Phase 10: Runner Integration

**File:** `src/DocsToKG/ContentDownload/runner.py` (modify)

**Changes:**

- Import BreakerRegistry, load_breaker_config, SQLiteCooldownStore
- In `DownloadRun.__init__()` or `setup()`:

  ```python
  cfg = load_breaker_config(
      yaml_path=resolved_config.breaker_yaml,
      env=os.environ,
      cli_host_overrides=resolved_config.breaker_host_overrides,
      ...
  )
  store = SQLiteCooldownStore(self.telemetry_dir / "breakers.sqlite")
  registry = BreakerRegistry(cfg, cooldown_store=store, listener_factory=...)
  ```

- Pass registry to `pipeline` and `networking` layers via resolved config
- Expose in resolved config frozen dataclass

---

### Phase 11: Documentation

**File:** `src/DocsToKG/ContentDownload/AGENTS.md` (update existing section)

**Updates:**

- Reference breaker implementation in "Networking, Rate Limiting & Politeness"
- Link to breakers.py, breaker_advisor.py, cli_breakers.py
- Add operational playbooks:
  - "Debug a noisy host" → run `breaker-advise --window-s 300`
  - "Temporary maintenance" → `breaker open api.example.org --seconds 600`
  - "Tune rate limiter before breaker" → heuristic guidance
- Example CLI invocations with `--breaker*` flags

---

### Phase 12: Validation

**Checklist:**

- [ ] All imports resolve (no circular deps)
- [ ] `ruff check src/DocsToKG/ContentDownload tests/`
- [ ] `mypy src/DocsToKG/ContentDownload`
- [ ] `pytest -q tests/content_download/test_breakers*.py`
- [ ] `cli.py --help` shows all breaker subcommands
- [ ] `cli.py breaker show` works with test config
- [ ] `cli.py breaker-advise --enforce` applies changes
- [ ] End-to-end: small dry-run with 5 works, inspect manifest metrics

---

## Key Design Decisions

### 1. Pybreaker Usage (from library reference)

- **State machine:** Closed → Open → Half-Open (industry-standard)
- **Failure classification:** Retryable 5xx/429/408, neutral 4xx business errors
- **Half-open:** trial_calls limit per role (metadata=1, artifact=2 by default)
- **Listeners:** Attached per-breaker for telemetry (state_change, success, failure, before_call)
- **Storage:** Local in-process (memory), optional cross-process via SQLite/Redis

### 2. Cooldown Store Abstraction

**Why wall-clock in storage, monotonic at runtime:**

- Protects against clock drift when shared across processes
- All breaker timeouts computed relative to monotonic (steady)
- Storage uses wall-clock for durability & human debugging

### 3. Retry-After Awareness

- Parsed from 429/503 response headers via existing helper
- Sets cooldown override (separate from pybreaker state) with cap (900s default)
- Honored before rolling-window or pybreaker decisions

### 4. Rolling Window Manual-Open

- Detects burst patterns: N failures in W seconds → open for C seconds
- Configured via YAML (disabled by default, opt-in)
- Complements pybreaker's consecutive-failure detection

### 5. Half-Open Probe Limiting

- pybreaker allows 1 trial in half-open by default
- Per-role overrides (e.g., artifact may allow 2 probes if artifact failures are rare)
- Tiny jitter (≤150ms) to desynchronize concurrent probes (optional)

### 6. Auto-Tuning Heuristics

- **Rate limiter first:** high 429 → reduce RPS before adjusting breaker
- **Retry-After signal:** use median Retry-After as reset_timeout hint
- **Open frequency:** if >3 opens/window → lower fail_max
- **Half-open outcomes:** if >50% fail → require 2 successes to close
- **Safe bounds:** clamp all changes to conservative ranges

---

## Implementation Order

1. **Week 1:** Phases 1 (cooldown stores), 2 (breakers.py completion), 3 (networking integration)
2. **Week 2:** Phase 4 (CLI commands), Phase 5 (advisor/auto-tuner), Phase 6 (CLI integration)
3. **Week 3:** Phase 7 (telemetry), Phase 8 (config YAML), Phase 9 (tests)
4. **Week 4:** Phase 10 (runner integration), Phase 11 (documentation), Phase 12 (validation)

---

## Success Metrics

- ✅ All 40+ unit tests passing (core + networking + CLI + advisor)
- ✅ Breaker denials reduce HTTP 429 downstream (measured via manifest metrics)
- ✅ Half-open probes restore service (success-after-open ≥70%)
- ✅ Telemetry shows opens/hour, time-saved, reason mix (human-debuggable)
- ✅ CLI `breaker show/open/close` work as documented
- ✅ Auto-advisor suggests safe tuning (no flapping, no over-aggressive changes)
- ✅ Zero breaking changes to existing ContentDownload CLI/config

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| pybreaker version incompatibility | Pin to 1.4.1; verify constants (STATE_CLOSED, etc.) at import |
| Circular imports | Load breakers_loader deferred in registry methods |
| Cross-process race conditions | Use sqlite_lock context manager for atomicity |
| Wall-clock drift | Always convert to monotonic at runtime |
| Over-aggressive tuning | Clamp all auto-tune suggestions to safe ranges; require `--enforce` flag |
| Telemetry schema bloat | Keep breaker_transitions table minimal; aggregate in summary only |

---

## Files to Create / Modify

| File | Action | Lines | Purpose |
|------|--------|-------|---------|
| `sqlite_cooldown_store.py` | Create | 151 | Cross-process cooldown (SQLite backend) |
| `redis_cooldown_store.py` | Create | 86 | Cross-process cooldown (Redis backend, optional) |
| `breakers.py` | Modify | +50 | Add success_threshold, complete TODOs |
| `networking.py` | Modify | +100 | Pre-flight/post-response breaker integration |
| `cli_breakers.py` | Create | 83 | CLI: breaker show/open/close |
| `cli_breaker_advisor.py` | Create | 52 | CLI: breaker-advise |
| `breaker_advisor.py` | Create | 283 | Metrics aggregation, heuristic advice |
| `breaker_autotune.py` | Create | 52 | Auto-tuner for safe policy adjustments |
| `cli.py` | Modify | +30 | Wire new CLI commands & factory |
| `runner.py` | Modify | +40 | Instantiate registry, pass to layers |
| `telemetry.py` | Modify | +80 | Breaker transitions table, metrics aggregation |
| `config/breakers.yaml` | Create | 60 | Policy template (defaults, host-specific) |
| `test_breakers_core.py` | Create | 250 | Unit tests (state machine, policies) |
| `test_breakers_networking.py` | Create | 200 | Integration tests (networking layer) |
| `test_cli_breakers.py` | Create | 125 | CLI tests (show/open/close) |
| `test_breaker_advisor.py` | Create | 150 | Advisor tests (metrics, heuristics) |
| `AGENTS.md` | Update | +50 | Document breaker ops, playbooks |

**Total new lines:** ~1,750 (code + tests + docs)
**Total modified lines:** ~300 (existing files)
**Estimated implementation time:** 4–6 weeks (phased, testable at each step)

---

## Next Steps

1. ✅ Review this plan with user
2. **Phase 1a:** Implement SQLiteCooldownStore
3. **Phase 1b:** Implement RedisCooldownStore
4. **Phase 2:** Complete breakers.py (success_threshold, TODOs)
5. **Phase 3:** Integrate with networking.request_with_retries
6. ... (continue through Phase 12)

All phases are designed to be independently testable before integration.
