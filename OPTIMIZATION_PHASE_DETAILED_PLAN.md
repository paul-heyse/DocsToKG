# OntologyDownload Optimization Phases: Comprehensive Implementation Plan

**Date**: October 20, 2025  
**Status**: Ready for Planning & Scoping  
**Scope**: 4 Major Pillars + 11 PRs across ~4-6 months

---

## Overview: The Four Pillars

This optimization initiative modernizes OntologyDownload around **four interconnected pillars**:

1. **Configuration You Can Trust (and Audit)** - Traced, hashed, schema'd settings
2. **CLI as a First-Class Product** - Professional, discoverable, self-documenting
3. **HTTP That Never Surprises You** - Single HTTPX client, centralized retry/redirect logic
4. **Rate-Limit Correctness Without Bespoke Code** - Unified pyrate-limiter facade

---

## Pillar 1: Configuration You Can Trust & Audit

### Goals
- ✅ Strict, typed, validated settings (Pydantic v2)
- ✅ Deterministic config hash (SHA-256) for reproducibility
- ✅ Provenance trail: source_fingerprint mapping
- ✅ Stable canonical JSON Schema (for agents/CI)
- ✅ Aliasing without drift (backward compat + deprecation path)

### Files & Boundaries
```
src/DocsToKG/OntologyDownload/
  ├─ settings.py (existing, expand with Phase 5.4 work)
  ├─ settings_sources.py (NEW: traced source loading)
  ├─ settings_schema.py (NEW: schema generation & validation)
  ├─ cli/
  │  └─ settings_cmd.py (NEW: show|schema|validate commands)
  └─ ...

docs/
  └─ schemas/
     ├─ settings.schema.json (top-level)
     ├─ settings.http.subschema.json
     ├─ settings.security.subschema.json
     ├─ settings.extraction.subschema.json
     ├─ settings.ratelimit.subschema.json
     └─ ... (one per sub-model)

CI/Pre-commit:
  └─ scripts/verify_settings_schema.sh (diff schemas, fail on drift)
```

### Source Precedence (Traceable)
1. **CLI overlay** (dict from Typer flags) - highest priority
2. **Config file** (ONTOFETCH_CONFIG or --config)
3. **.env.ontofetch**
4. **.env**
5. **Environment** (ONTOFETCH_*)
6. **Defaults** (baked in models)

Each source produces `source_fingerprint[field_name] = source_name`, never logging values, only sources.

### Phase 5.4+ Tasks
- Add `TracingSettingsSource` wrapper for Pydantic v2
- Implement `settings.source_fingerprint: dict[str, str]`
- Generate canonical JSON Schema (top-level + per-submodel)
- Implement `settings show|schema|validate` CLI commands
- Add config hash determinism tests
- Create CI job to detect schema drift

### Acceptance Criteria
- [ ] `TracingSettingsSource` produces `source_fingerprint`
- [ ] `settings schema` generates stable JSON Schema
- [ ] `settings show` displays field | value(redacted) | source | notes
- [ ] `settings validate --file config.yaml` works
- [ ] Schema drift detected in CI (build fails)
- [ ] 100% test coverage for source precedence + aliasing

---

## Pillar 2: CLI as First-Class Product

### Goals
- ✅ Zero-surprise UX: typed options, safe defaults, consistent output
- ✅ Discoverable: rich help, examples, autocompletions
- ✅ Safe: destructive ops require confirmation, --dry-run, clear exit codes
- ✅ Self-documenting: generated docs auto-detect drift

### Files & Structure
```
src/DocsToKG/OntologyDownload/cli/
  ├─ __init__.py (app factory)
  ├─ main.py (Typer app, global options, context factory)
  ├─ _common.py (shared types, printer, confirmation, context)
  ├─ settings_cmd.py (show|schema|validate)
  ├─ plan_cmd.py (plan)
  ├─ pull_cmd.py (pull)
  ├─ extract_cmd.py (extract)
  ├─ validate_cmd.py (validate)
  ├─ latest_cmd.py (latest get|set)
  ├─ prune_cmd.py (prune --dry-run|--apply)
  ├─ db_cmd.py (db versions|files|stats|doctor)
  └─ delta_cmd.py (delta summary|files|renames|formats|validation)

tests/cli/
  ├─ test_settings_cmd.py
  ├─ test_plan_cmd.py
  ├─ test_pull_cmd.py
  ├─ test_extractive_cmd.py
  ├─ test_validate_cmd.py
  ├─ test_latest_cmd.py
  ├─ test_prune_cmd.py
  ├─ test_db_cmd.py
  ├─ test_delta_cmd.py
  └─ test_cli_integration.py

scripts/
  └─ gen_cli_docs.py (Typer introspection → Markdown)

docs/
  └─ cli/
     ├─ index.md
     ├─ settings.md
     ├─ plan.md
     ├─ pull.md
     ├─ extract.md
     ├─ validate.md
     ├─ latest.md
     ├─ prune.md
     ├─ db.md
     └─ delta.md
```

### Global Options (in @app.callback())
```
--config PATH               (envvar: ONTOFETCH_CONFIG) - traced loader
-v / -vv                   (maps to INFO/DEBUG log levels)
--format [table|json|yaml]  (unifies output rendering)
--dry-run                   (no writes; show planned actions)
--version (eager)           (prints: version, git_sha, build_date, py, os, libarchive_version)
```

### Exit Codes (Unified)
```
0   - Success
2   - Usage/validation error
3   - Policy gate rejection (e.g., traversal path attempted)
4   - Storage/filesystem error
5   - Network/TLS error
6   - Database/migration error
7   - Unknown error
```

### Commands Scope

**`settings show|schema|validate`** (integrates Pillar 1)
- `show`: effective config with sources
- `schema`: write JSON Schema to docs/schemas/
- `validate --file config.yaml`: validate external config file

**`plan|pull|extract|validate`**
- Consume `--config`, `-v`, `--format`
- Support `--dry-run` where applicable
- Emit `cli.command.start/done/error` events

**`latest get|set`**
- `get`: print current + DB status
- `set <version>`: requires confirmation

**`prune --dry-run|--apply`**
- Lists orphans; `--apply` requires typing `PRUNE`
- Support `--exclude-glob`

**`db versions|files|stats|doctor`**
- DuckDB integration; show metadata
- Support `--format` + autocompletions

**`delta summary|files|renames|formats|validation A B`**
- Compare two versions using DuckDB macros

### Autocompletions
- `version_id` ← DuckDB: `SELECT version_id FROM versions ORDER BY created_at DESC LIMIT 50`
- `service` ← DuckDB: distinct `artifacts.service`
- `format` ← DuckDB: distinct `extracted_files.format`

### Destructive Confirmations
- `prune --apply`: Require `--yes` OR interactive `type PRUNE to continue`
- `latest set`: Require `--yes` OR interactive confirm
- On non-TTY (CI): REQUIRE `--yes` (no interactive prompt)

### Phase 5.4+ Tasks (Sequence: PR-B → PR-C → PR-D)

**PR-B: Typer App Foundation**
- Create app skeleton (`main.py`, context factory)
- Add global options: `--config`, `-v/-vv`, `--format`, `--dry-run`, `--version`
- Implement `settings` subcommands (show/schema/validate)
- Add golden-help tests

**PR-C: Completions + Confirmations + Exit Codes**
- Add DuckDB-backed autocompletions
- Implement confirmation prompts + `--yes` bypass
- Unify exit codes; test coverage

**PR-D: CLI Docs & Examples**
- Create `scripts/gen_cli_docs.py` (Typer introspection)
- Generate Markdown per command
- Wire to CI (build fails if docs stale)
- Add 3 examples per command in help text

### Acceptance Criteria
- [ ] `ontofetch --help` responds in <100ms
- [ ] `ontofetch --version` prints full version info
- [ ] All commands honor `--format`, `--dry-run`, `-v/-vv`
- [ ] Completions work on bash/zsh/fish
- [ ] Destructive ops require confirmation
- [ ] Exit codes match spec (0/2/3/4/5/6/7)
- [ ] Generated CLI docs match code (CI fails on drift)

---

## Pillar 3: HTTP That Never Surprises You

### Goals
- ✅ Exactly one long-lived HTTPX `Client` (shared across planner/downloader/validators)
- ✅ Zero bespoke retry logic in call-sites (connect retries → transport; 429/5xx → Tenacity)
- ✅ Uniform instrumentation: every request tagged, timed, cache-status recorded
- ✅ Redirect safety: centralized validation gate
- ✅ Deterministic tests: mocked transport, no real network

### Files & Ownership
```
src/DocsToKG/OntologyDownload/net/
  ├─ client.py (Client factory, lifecycle, hook wiring)
  ├─ policy.py (HTTP policy constants: timeouts, pool, headers, SSL)
  ├─ instrumentation.py (hooks: on_request, on_response, on_error)
  ├─ errors.py (HTTPX exception → domain error mapping)
  └─ redirects.py (audit + follow with validate_url_security)

tests/network/
  ├─ test_client_lifecycle.py
  ├─ test_client_hooks.py
  ├─ test_redirects.py
  ├─ test_timeouts.py
  ├─ test_retries.py
  ├─ test_streaming.py
  └─ test_mocked_transport.py
```

### Client Lifecycle

**Construction** (once, lazy singleton or at module init):
- Timeouts: connect 5s, read 30s, write 30s, pool acquire 5s
- Pool: max_connections 64, keepalive 20, expiry 30s
- HTTP/2: enabled (better for concurrent requests to same host)
- Redirects: **disabled globally** (explicit opt-in at call-site)
- SSL: TLS 1.2+, SNI on, system trust (certifi fallback)
- User-Agent: single place, never overridden ad-hoc

**Hooks** (attached once):
- `on_request`: stamp correlation_id, run_id, service, host; record t_start
- `on_response`: compute timings (ttfb, read_total), extract cache metadata, emit structured metrics
- `on_error`: map exceptions (ConnectTimeout, ReadTimeout, SSLError, etc) to domain errors (E_NET_CONNECT, E_NET_READ, E_TLS, etc)

**Shutdown**:
- Provide `close()` guarded against double-close
- Tests ensure no resource leaks

### Policy Defaults
```
Timeouts:
  - connect: 5s (conservative)
  - read: 30s (metadata), ~120s (downloads via separate config)
  - write: 30s
  - pool_acquire: 5s

Redirect policy:
  - Off by default
  - Audit + follow helper: sanitize target, validate_url_security, record hops

Headers:
  - Accept, Accept-Encoding, User-Agent set once
  - No per-call mutation unless resolver requires
  - Never forward Authorization across hosts

Decompression:
  - Accept gzip/deflate/br; allow opt-out
```

### Retry Division

**Transport-level** (HTTPX retries):
- Small # retries for **connect errors ONLY** (TCP reset, timeouts)
- Capped backoff; never retry after bytes sent

**Application-level** (Tenacity):
- 429: honor Retry-After with jitter
- 5xx (idempotent GET/HEAD): bounded retries
- 4xx: abort (except 408/409/429 per policy)
- Record attempt #, sleep_ms in logs

### Instrumentation

Every request emits `net.request` event:
```
{
  service, host, method, url_redacted, status, attempt, 
  elapsed_ms, ttfb_ms, bytes_read, cache_status, http2, reused_conn
}
```

Downloader emits `download.fetch` (higher-level):
```
{
  file_size, speed_mbps, cache_status, elapsed_ms, retries
}
```

### Testing Plan (MockTransport)
- Happy path: 200 with headers
- Redirect chain: 301→302→200 (mixed hosts; ensure validator blocks unsafe)
- Retries: connect drop → 200; verify attempt count, backoff
- Timeouts: connect vs read; ensure error mapping
- Content-type: mismatch vs alias acceptance
- Contract tests: in-process ASGI test app (206 Range, chunked, gzip/br)
- Resource tests: ensure client closes; fd count stable

### Phase 5.5+ Tasks

**PR-H1: Client Foundation**
- Create client.py factory + lifecycle
- Implement timeouts, pool, HTTP/2, SSL
- Add on_request/on_response/on_error hooks
- Unit tests for lifecycle

**PR-H2: Redirect Audit + Follow**
- Implement audit_and_follow helper
- Integrate validate_url_security gate
- Test redirect chains + security gates

**PR-H3: Retrying & Error Mapping**
- Tenacity integration for 429/5xx
- HTTPX exception → domain error mapping
- Tests for retry behavior + timing

**PR-H4: Integration & Observability**
- Wire up planner, downloader, validators to use shared Client
- Emit net.request + download.fetch events
- Integration tests with real streaming, timeouts, retries

### Acceptance Criteria
- [ ] Exactly one shared Client; redirects off globally; HTTP/2 on
- [ ] Hook chain active; `net.request` logs include timings & cache hints
- [ ] Transport retries limited to connect; Tenacity handles 429/5xx with jitter
- [ ] Redirect audit validates each hop; unsafedirected rejected
- [ ] All downloader/probe paths use streaming (no buffer-entire-body)
- [ ] Deterministic tests cover redirects, retries, timeouts, error mapping

---

## Pillar 4: Rate-Limit Correctness Without Bespoke Code

### Goals
- ✅ One authoritative limiter façade; no custom token math in call-sites
- ✅ Keys: (service, host) to mirror external quotas
- ✅ Single-process (in-memory) and multi-process (SQLite) modes
- ✅ Two behaviors: block-until-allowed (downloader) vs fail-fast (resolvers)
- ✅ Multi-window enforcement (5/sec AND 300/min)

### Files & Ownership
```
src/DocsToKG/OntologyDownload/ratelimit/
  ├─ manager.py (façade: acquire, registry, lifecycle)
  ├─ config.py (parse & normalize rate specs from settings)
  ├─ instrumentation.py (counters, blocked_ms timing logs)
  └─ __init__.py (exports)

tests/ratelimit/
  ├─ test_manager_unit.py
  ├─ test_manager_multiprocess.py
  ├─ test_manager_integration.py
  └─ test_manager_chaos.py (429 scenarios)
```

### Configuration Model (from Phase 5 Settings)

Already in Phase 5.2 `RateLimitSettings`:
```python
default: Optional[str]  # e.g., "8/second" or None (unlimited)
per_service: Dict[str, str]  # e.g., {"ols": "4/second", "bioportal": "2/second"}
shared_dir: Optional[Path]  # enables SQLite cross-process mode
engine: str  # "pyrate" (only supported for now)
```

Normalization:
- Accept semicolon string, JSON/YAML/TOML maps
- Output canonical `dict[str, RateSpec]` where `RateSpec = (limit: int, unit: Enum, rps: float)`

### Keying & Registry
```
key = f"{service or '_'}:{host or 'default'}"

registry[key] = {
  limiter: Limiter(...),
  rates: List[Rate],
  bucket: InMemoryBucket | SQLiteBucket
}
```

Pre-warm known services (OLS, BioPortal, etc.); lazily create others.

### Acquire Semantics
```python
async_acquire(service: str|None, host: str|None, *, weight: int=1, mode: Literal["block","fail"]) -> None

# block mode: try_acquire(..., max_delay=None) — sleeps until allowed
# fail mode: try_acquire(..., max_delay=0, raise_when_fail=True) — raises immediately

# Where to call:
# - Downloader: before each GET/stream
# - Resolver: wrap Tenacity loop before attempt
```

### Multi-Window & Burst Control
```
rates[key] = [Rate(5, SECOND), Rate(300, MINUTE)]

# limiter enforces BOTH constraints simultaneously
```

### Retry-After & Cool-Down Integration
- Do NOT mutate bucket on 429
- Tenacity reads `Retry-After`, sleeps before next acquire
- Optional cool-down map: `cooldown_until[key] = monotonic() + delta` so next acquire respects it

### Cross-Process Safety (SQLite Mode)
- When `shared_dir` set: all processes point at ONE SQLite DB
- Correct global accounting (multiple workers)
- Graceful restarts without burst overshoot
- `doctor rate-limits` CLI shows configured windows & current tokens

### Instrumentation

Each `acquire()` emits `ratelimit.acquire` event:
```
{
  key, rates: ["5/s", "300/min"], weight, mode, blocked_ms, outcome: "ok|exceeded|error"
}
```

Aggregate counters: blocked_ms total per key, #exceeded, top blockers.

### Testing Matrix

**Unit** (time-controlled):
- Parse rate strings; invalid → clean errors
- Block mode: acquire N+1 → sleeps ~(1/rate) seconds
- Fail mode: acquire N+1 → raises immediately
- Multi-window: 5/s AND 300/min both enforced

**Cross-process** (SQLite):
- Two subprocesses; combined rate respects cap

**Integration**:
- Downloader under block mode respects quotas
- Resolver under fail-fast returns `RateLimitExceeded`

**Chaos**:
- Simulate 429 with `Retry-After=2s`; next acquire waits ≥2s

### Phase 5.5+ Tasks

**PR-R1: Limiter Façade & Config Wiring**
- Parse settings → canonical per-service RateSpec
- Registry + keying; in-memory bucket
- `acquire()` (block|fail) + instrumentation

**PR-R2: Multi-Window, Weights, SQLite Mode**
- List of Rate per key; `shared_dir` SQLite support
- Cool-down map (optional)
- `doctor rate-limits` CLI

**PR-R3: Call-Site Integration & Tests**
- Downloader + resolvers call `acquire()`
- Integration tests with Tenacity (429/5xx timing)

**PR-R4: Observability Polish**
- Aggregate counters; top-blockers report
- CI budget tests prevent regressions

### Acceptance Criteria
- [ ] Single façade `acquire()` used everywhere; no raw pyrate calls
- [ ] Keys are (service, host); pre-warm known services; lazily add others
- [ ] Multi-window rates enforced; weights supported
- [ ] Block vs fail-fast selectable, test-covered
- [ ] SQLite mode coords multiple processes; in-memory default
- [ ] 429 cool-down handled by Tenacity + cool-down map
- [ ] `ratelimit.acquire` events emitted with blocked_ms
- [ ] `doctor rate-limits` shows keys, windows, tokens

---

## Implementation Roadmap: 11 PRs in Phases

### Timeline: 4-6 Months (Q4 2025 → Q1 2026)

```
Phase 5.4 (Current: Oct 20 - Oct 31)
  └─ PR-A: Settings Trust & Tracing (already started with Phase 5.1-5.3)
     • Add TracingSettingsSource
     • Implement source_fingerprint
     • Generate JSON Schema
     • Add settings show|schema|validate commands

Phase 5.5 (Nov 1 - Nov 15)
  ├─ PR-B: Typer App Foundation
  │  • main.py, context factory
  │  • Global options: --config, -v/-vv, --format, --dry-run, --version
  │  • settings subcommands
  │
  └─ PR-H1: HTTP Client Foundation
     • client.py factory, lifecycle, hooks
     • Timeouts, pool, HTTP/2, SSL

Phase 5.6 (Nov 16 - Nov 30)
  ├─ PR-C: CLI Completions + Confirmations
  │  • DuckDB-backed autocompletions
  │  • Confirmation prompts, --yes bypass
  │  • Exit codes unified
  │
  ├─ PR-H2: Redirect Audit + Follow
  │  • audit_and_follow helper
  │  • validate_url_security integration
  │  • Redirect chain testing
  │
  └─ PR-R1: Rate-Limit Façade & Config
     • Parse settings → canonical RateSpec
     • Registry + in-memory bucket
     • acquire() (block|fail)

Phase 5.7 (Dec 1 - Dec 15)
  ├─ PR-D: CLI Docs & Examples
  │  • gen_cli_docs.py (Typer introspection)
  │  • Generated Markdown per command
  │  • CI drift detection
  │
  ├─ PR-H3: Retrying & Error Mapping
  │  • Tenacity integration
  │  • HTTPX exception mapping
  │  • Retry behavior tests
  │
  └─ PR-R2: Multi-Window + SQLite Mode
     • List of Rate per key
     • shared_dir SQLite support
     • Cool-down map
     • doctor rate-limits CLI

Phase 5.8 (Dec 16 - Dec 31)
  ├─ PR-H4: HTTP Integration & Observability
  │  • Wire planner/downloader/validators
  │  • net.request + download.fetch events
  │  • Streaming integration tests
  │
  └─ PR-R3: Rate-Limit Call-Site Integration
     • Downloader + resolvers use acquire()
     • Integration tests + timing verification

Phase 5.9 (Jan 1 - Jan 15, 2026)
  └─ PR-R4: Rate-Limit Observability
     • Aggregate counters
     • Top-blockers report
     • CI budget tests
```

---

## Cross-Pillar Integration Points

### Configuration → CLI
- CLI parses flags → overlay dict
- Traced loader produces source_fingerprint
- Every command emits cli.command.start/config_hash

### CLI → HTTP
- CLI `--format` option flows to printer
- `--version` shows libarchive version
- `--dry-run` sets no-write flag in context

### CLI → Rate-Limit
- CLI may expose `--rate-limit-mode` (block|fail)
- `doctor rate-limits` surfaces current tokens
- Downloader honors rate-limit settings

### Configuration → HTTP
- HttpSettings drives HTTPX client timeouts/pool
- SecuritySettings drives URL validation in redirects

### Configuration → Rate-Limit
- RateLimitSettings populates registry on startup
- Per-service rates applied to (service, host) keys

### HTTP → Rate-Limit
- net.request event includes blocked_ms if rate-limited
- 429 response triggers Tenacity + cool-down

### HTTP → Observability
- Each net.request includes run_id + config_hash
- Downloader emits download.fetch referencing net.request

---

## Key Design Principles (Across All Pillars)

1. **Single Source of Truth**
   - One Settings instance (singleton)
   - One HTTPX Client (shared)
   - One RateLimit registry (shared)

2. **Immutability**
   - Settings frozen post-init
   - No per-call mutations of HTTP client
   - Rate-limit bucket updates atomic

3. **Traceability**
   - config_hash in every event
   - run_id for correlation
   - source_fingerprint for blame
   - Structured logging (not printf)

4. **Safety**
   - Destructive ops require confirmation
   - --dry-run shows planned actions
   - Validation early (at CLI parse)
   - Explicit error codes (2/3/4/5/6/7)

5. **Testability**
   - Mocked transports (no real network in tests)
   - Time-controlled rate-limit tests
   - Golden snapshots for CLI help
   - Deterministic fixtures

---

## Risks & Mitigations

### Risk: Scope Creep
**Mitigation**: Strict PR boundaries (each PR solves one thing); no merged code changes outside PR scope

### Risk: Performance Regression (Tracing Overhead)
**Mitigation**: Profile before/after each PR; keep TracingSettingsSource overhead minimal (<1ms per load)

### Risk: Backward Compatibility (Rate-Limit API)
**Mitigation**: Old config still works (aliasing); new `acquire()` only called in new code paths; old code untouched

### Risk: Test Flakiness (Rate-Limit Tests)
**Mitigation**: Use fake clock (freezegun); assert with tolerance bands; no real sleep() in unit tests

### Risk: HTTP Client Resource Leaks
**Mitigation**: Ensure `close()` called in tests; check /proc/self/fd count; resource warnings on

### Risk: CLI Drift (Docs out of sync)
**Mitigation**: gen_cli_docs.py runs in CI; diff schemas & docs; build fails on drift

---

## Questions for You

Before proceeding, I have a few clarifications:

1. **Phase 5.4 Status**: Should I assume Phase 5.4 tasks (TracingSettingsSource, JSON Schema, settings commands) are IN SCOPE for the next sprint, or should these be deferred until after we complete Pillars 3 & 4?

2. **Priority Sequencing**: Which pillar feels most urgent to you?
   - Pillar 1 (Settings): Builds foundation for everything else
   - Pillar 2 (CLI): Most visible to users/ops
   - Pillar 3 (HTTP): Fixes network reliability/observability
   - Pillar 4 (Rate-Limit): Prevents quota violations

3. **Parallelization**: Can we run PR-H1 (HTTP client) and PR-R1 (rate-limit) in parallel, or do they have dependencies? (I believe they're independent)

4. **Scope Boundaries**: Are there any existing code patterns or APIs you want preserved vs. refactored? (E.g., keep existing Resolver interface, but modernize HTTP calls underneath)

5. **Testing Infrastructure**: Do we have a test database fixture we can use for autocompletion + CLI integration tests? Or should I create one?

6. **Documentation**: Should the detailed plan be captured in this markdown file, or would you prefer separate PR checklists in GitHub issues?

---

**Status**: READY FOR APPROVAL & SCOPING  
**Next Step**: Clarifications + PR prioritization → then detailed Phase 5.4 kick-off

