# Fallback & Resiliency Strategy - Implementation Plan

**Document Version**: 1.0
**Status**: READY FOR IMPLEMENTATION
**Target**: Optimization 9 (ContentDownload)
**Scope**: Tiered, budgeted resolution with health gates, cancellation, and observability

---

## Executive Summary

This document outlines the implementation of **Fallback & Resiliency Strategy** (Optimization 9) for ContentDownload. The system provides:

- **Deterministic, tiered resolution** across 7 sources (Unpaywall → arXiv → PMC → DOI → Landing → Europe PMC → Wayback)
- **Budgeted execution** (time, attempt count, concurrency limits)
- **Health gates** (breaker state, offline mode, rate limiter awareness)
- **Cancellation & race to success** (stop as soon as valid PDF found)
- **Full observability** (per-attempt telemetry, SLO tracking)
- **Zero-configuration defaults** with YAML/env/CLI tuning

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ FallbackOrchestrator                                             │
│  - Executes plan tier-by-tier                                   │
│  - Enforces budgets (time, attempts, concurrency)               │
│  - Manages cancellation flag                                    │
│  - Emits telemetry                                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
    Adapters                               Health Gates
    (7 sources)                        - Breaker check
    - Unpaywall                        - Offline mode
    - arXiv                            - Rate limiter
    - PMC                              - Cache aware
    - DOI redirect
    - Landing scrape
    - Europe PMC
    - Wayback
        ↓                                       ↓
        └───────────────────┬───────────────────┘
                            ↓
                    AttemptResult
                    - outcome
                    - url (if found)
                    - status
                    - reason
                    - metadata
                            ↓
                      Telemetry Sink
```

---

## Phase 1: Core Types (fallback/types.py)

**File**: `src/DocsToKG/ContentDownload/fallback/types.py`
**Lines**: ~150
**Complexity**: Low (dataclasses only)

### Dataclasses to Create

1. **ResolutionOutcome** (Literal)

   ```
   "success" | "no_pdf" | "nonretryable" | "retryable" |
   "timeout" | "skipped" | "error"
   ```

2. **AttemptPolicy**
   - name: str (source name)
   - timeout_ms: int
   - retries_max: int
   - robots_respect: bool

3. **AttemptResult**
   - outcome: ResolutionOutcome
   - url: Optional[str]
   - status: Optional[int]
   - host: Optional[str]
   - reason: str (short code)
   - meta: Dict[str, Any]
   - elapsed_ms: int

4. **TierPlan**
   - name: str
   - parallel: int
   - sources: list[str]

5. **FallbackPlan**
   - budgets: dict
   - tiers: list[TierPlan]
   - policies: dict[str, AttemptPolicy]
   - gates: dict

### Checklist

- [ ] Create fallback/**init**.py (empty, allows imports)
- [ ] Create fallback/types.py with all 5 dataclasses
- [ ] Add comprehensive docstrings
- [ ] Add type hints throughout
- [ ] Run linting/formatting
- [ ] No external dependencies

---

## Phase 2: Orchestrator (fallback/orchestrator.py)

**File**: `src/DocsToKG/ContentDownload/fallback/orchestrator.py`
**Lines**: ~300-400
**Complexity**: High (threading, budgets, cancellation)

### Key Methods

1. **FallbackOrchestrator.**init****
   - Takes: plan, breaker, rate limiter, clients, telemetry, logger
   - Stores references

2. **resolve_pdf(context, adapters)**
   - Main entry point
   - Returns: AttemptResult (success or final failure)
   - Implements:
     - Tiered iteration
     - Per-tier parallelization
     - Budget enforcement
     - Cancellation flag
     - Telemetry emission

3. **_health_gate(source_name, context)**
   - Checks breaker state
   - Checks offline mode
   - Returns: AttemptResult (skipped) or None (proceed)

4. **_emit_attempt_telemetry(tier, result, context)**
   - Low-cardinality fields only
   - Sends to telemetry sink

### Threading Strategy

- Use `threading.Thread` for tier parallelization
- Use `queue.Queue` for result collection
- Use `threading.Event` for cancellation flag
- Short `thread.join(timeout=0.5)` to avoid hanging

### Budget Enforcement

- `total_timeout_ms`: hard cap from start
- `total_attempts`: count across all sources
- `max_concurrent`: limit inflight threads
- `per_source_timeout_ms`: defaults per source

### Checklist

- [ ] Create orchestrator.py
- [ ] Implement tiered scheduler
- [ ] Implement thread pooling
- [ ] Implement budget tracking
- [ ] Implement cancellation flag
- [ ] Implement health gates
- [ ] Add comprehensive docstrings
- [ ] Test threading behavior locally

---

## Phase 3: Source Adapters (fallback/adapters/)

**Files**: `src/DocsToKG/ContentDownload/fallback/adapters/`
**Total Lines**: ~600-800
**Complexity**: Medium-High (HTTP logic, URL handling, parsing)

### 7 Adapters to Create

Each adapter is a function: `(policy: AttemptPolicy, ctx: dict) → AttemptResult`

#### 3.1 Unpaywall PDF (`adapters/unpaywall.py`)

- Input: DOI from context
- Call: Unpaywall API (metadata role, cached client)
- Validate: HEAD check on PDF URL
- Output: AttemptResult

#### 3.2 arXiv PDF (`adapters/arxiv.py`)

- Input: arXiv ID from context
- Build: `https://arxiv.org/pdf/<id>.pdf`
- Validate: HEAD check
- Output: AttemptResult

#### 3.3 PMC PDF (`adapters/pmc.py`)

- Input: PMCID or PMID
- Call: E-utils or EPMC API
- Validate: HEAD check
- Output: AttemptResult

#### 3.4 DOI Redirect (`adapters/doi_redirect.py`)

- Input: DOI
- Follow: `https://doi.org/<doi>` with redirects
- Validate: HEAD check (raw client)
- Output: AttemptResult

#### 3.5 Landing Scrape (`adapters/landing_scrape.py`)

- Input: Landing page URL
- GET: HTML page (metadata role, cached)
- Parse: `<meta>`, `<link>`, `<a>` for PDF
- Validate: HEAD check
- Output: AttemptResult

#### 3.6 Europe PMC PDF (`adapters/europe_pmc.py`)

- Input: DOI/PMID
- Call: EPMC API
- Validate: HEAD check
- Output: AttemptResult

#### 3.7 Wayback (`adapters/wayback.py`)

- Input: Landing/artifact URL
- Call: WaybackResolver (existing)
- Validate: HEAD check
- Output: AttemptResult

### Shared Utilities

Create `adapters/__init__.py` with:

- `head_pdf(url, client, timeout) → (ok, status, host, reason, meta)`
- Validates content-type and sniffs `%PDF-`
- Calls breaker.allow() first
- Returns structured result

### Checklist

- [ ] Create adapters/ directory
- [ ] Create **init**.py with shared utilities
- [ ] Create unpaywall.py adapter
- [ ] Create arxiv.py adapter
- [ ] Create pmc.py adapter
- [ ] Create doi_redirect.py adapter
- [ ] Create landing_scrape.py adapter
- [ ] Create europe_pmc.py adapter
- [ ] Create wayback.py adapter
- [ ] All adapters follow same signature
- [ ] All adapters respect timeout/retry budgets
- [ ] All adapters call breaker.allow()
- [ ] All adapters canonicalize URLs

---

## Phase 4: Configuration (config/fallback.yaml)

**File**: `src/DocsToKG/ContentDownload/config/fallback.yaml`
**Lines**: ~100
**Complexity**: Low (YAML structure)

### YAML Schema

```yaml
version: 1

budgets:
  total_timeout_ms: 120000       # hard cap
  total_attempts: 20             # across all sources
  max_concurrent: 3              # parallel threads
  per_source_timeout_ms: 10000   # default

tiers:
  - name: "direct_oa"
    parallel: 2
    sources: [unpaywall_pdf, arxiv_pdf, pmc_pdf]

  - name: "doi_follow"
    parallel: 1
    sources: [doi_redirect_pdf]

  - name: "landing_scrape"
    parallel: 2
    sources: [publisher_landing_pdf, europe_pmc_pdf]

  - name: "archive"
    parallel: 1
    sources: [wayback_pdf]

sources:
  unpaywall_pdf:
    timeout_ms: 6000
    retries_max: 3

  arxiv_pdf:
    timeout_ms: 6000
    retries_max: 3

  # ... (all 7 sources)

gates:
  skip_if_breaker_open: true
  skip_if_http2_denied: false
  offline_behavior: "metadata_only"  # | "block_all" | "cache_only"
```

### Checklist

- [ ] Create config/fallback.yaml
- [ ] Define all 4 tiers with sources
- [ ] Define all 7 source policies
- [ ] Define budget defaults
- [ ] Define gates
- [ ] Validate YAML syntax

---

## Phase 5: Configuration Loader (fallback/loader.py)

**File**: `src/DocsToKG/ContentDownload/fallback/loader.py`
**Lines**: ~200-300
**Complexity**: Medium (YAML parsing, merging)

### Main Function: `load_fallback_plan()`

Inputs:

- yaml_path: Path to config/fallback.yaml
- env: dict of environment variables
- cli_overrides: dict of CLI arguments

Outputs:

- FallbackPlan (fully resolved)

### Precedence

1. Start with YAML defaults
2. Override with environment variables (`DOCSTOKG_FB_*`)
3. Override with CLI arguments

### Environment Variables

- `DOCSTOKG_FB_TOTAL_TIMEOUT_MS`
- `DOCSTOKG_FB_MAX_CONCURRENT`
- `DOCSTOKG_FB_TIER_OVERRIDE` (JSON tier config)
- `DOCSTOKG_FB_DISABLE_SOURCES` (comma-separated)

### CLI Overrides

- `--fallback-total-timeout-ms`
- `--fallback-max-concurrent`
- `--fallback-tier-override`
- `--disable-wayback`
- `--disable-landing-scrape`

### Checklist

- [ ] Create loader.py
- [ ] Implement YAML loading
- [ ] Implement environment variable parsing
- [ ] Implement CLI override merging
- [ ] Validate final config
- [ ] Add comprehensive docstrings
- [ ] Handle missing YAML gracefully

---

## Phase 6: CLI Commands (cli_fallback.py)

**File**: `src/DocsToKG/ContentDownload/cli_fallback.py`
**Lines**: ~150-200
**Complexity**: Medium

### Commands

1. **fallback plan** - Print effective plan

   ```bash
   python -m DocsToKG.ContentDownload.cli fallback plan
   ```

   Output: Human-readable plan (tiers, sources, timeouts)

2. **fallback dryrun** - Simulate resolution without network

   ```bash
   python -m DocsToKG.ContentDownload.cli fallback dryrun --max 10
   ```

   Output: Plan execution trace

3. **fallback tune** - Suggest optimizations from telemetry

   ```bash
   python -m DocsToKG.ContentDownload.cli fallback tune --window-s 3600
   ```

   Output: Tuning recommendations

### Checklist

- [ ] Create cli_fallback.py
- [ ] Implement fallback plan command
- [ ] Implement fallback dryrun command
- [ ] Implement fallback tune command
- [ ] Add to main CLI parser
- [ ] Test all commands

---

## Phase 7: Telemetry (telemetry.py updates)

**File**: `src/DocsToKG/ContentDownload/telemetry.py` (MODIFY)
**Lines**: +50
**Complexity**: Low

### New Table: `fallback_events`

```sql
CREATE TABLE IF NOT EXISTS fallback_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  job_id TEXT,
  ts REAL NOT NULL,
  event_type TEXT NOT NULL,  -- "attempt" | "summary"
  tier TEXT,
  source TEXT,
  outcome TEXT,
  reason TEXT,
  host TEXT,
  status INTEGER,
  elapsed_ms INTEGER,
  winner_source TEXT,
  details TEXT,  -- JSON metadata
  INDEX(run_id, ts),
  INDEX(job_id)
);
```

### New Methods in SqliteSink

- `log_fallback_attempt(event_dict)` - Log single attempt
- `log_fallback_summary(event_dict)` - Log summary after resolution

### Checklist

- [ ] Add fallback_events table to migration
- [ ] Add log_fallback_attempt() method
- [ ] Add log_fallback_summary() method
- [ ] Update SqliteSink to delegate to telemetry
- [ ] Test telemetry logging

---

## Phase 8: Integration (download.py updates)

**File**: `src/DocsToKG/ContentDownload/download.py` (MODIFY)
**Lines**: +30-40
**Complexity**: Low (mostly wiring)

### Feature Gate

```python
ENABLE_FALLBACK = os.getenv("DOCSTOKG_ENABLE_FALLBACK", "false").lower() == "true"
```

### Integration Point: `process_one_work()`

Before streaming attempt:

```python
if ENABLE_FALLBACK:
    fallback_plan = load_fallback_plan(...)
    orchestrator = FallbackOrchestrator(...)
    attempt_result = orchestrator.resolve_pdf(context, adapters)

    if attempt_result.outcome == "success":
        # Use attempt_result.url for streaming
        ...
    else:
        # Fall back to existing download logic
        ...
```

### Checklist

- [ ] Add feature gate to download.py
- [ ] Load fallback plan
- [ ] Create orchestrator
- [ ] Call resolve_pdf()
- [ ] Handle success/failure outcomes
- [ ] Test with feature on/off

---

## Phase 9: Tests (test_fallback.py)

**File**: `tests/content_download/test_fallback.py`
**Lines**: ~600-800
**Complexity**: High (threading, mocking)

### Test Categories

#### 9.1 Types Tests (5 tests)

- [ ] AttemptPolicy creation
- [ ] AttemptResult creation
- [ ] TierPlan creation
- [ ] FallbackPlan validation
- [ ] Outcome enum values

#### 9.2 Orchestrator Tests (15 tests)

- [ ] Winner cancels rest (threading)
- [ ] Budget expiry (timeout)
- [ ] Attempt limit (count)
- [ ] Concurrent limit (threads)
- [ ] Breaker skips (health gate)
- [ ] Offline blocking (health gate)
- [ ] Tier ordering (sequence)
- [ ] Per-source timeout
- [ ] Telemetry emission
- [ ] Error handling
- [ ] Queue safety
- [ ] Cancellation flag
- [ ] Thread cleanup
- [ ] Result collection
- [ ] Exhausted outcome

#### 9.3 Adapter Tests (20+ tests, one per adapter)

- [ ] Unpaywall: successful PDF find
- [ ] Unpaywall: no DOI
- [ ] Unpaywall: no PDF field
- [ ] arXiv: successful PDF
- [ ] arXiv: invalid ID
- [ ] PMC: successful PDF
- [ ] DOI: redirect chain
- [ ] Landing: meta tag extraction
- [ ] Landing: robots respect
- [ ] Europe PMC: successful PDF
- [ ] Wayback: archive find
- [ ] Common: HEAD validation
- [ ] Common: PDF sniffing
- [ ] Common: Content-Type checking
- [ ] Common: Timeout handling

#### 9.4 Config Tests (10 tests)

- [ ] YAML loading
- [ ] Env var override
- [ ] CLI override
- [ ] Precedence (YAML < env < CLI)
- [ ] Validation
- [ ] Missing YAML handling
- [ ] Bad tier names
- [ ] Unknown sources
- [ ] Invalid budgets
- [ ] Merging logic

#### 9.5 Integration Tests (10 tests)

- [ ] Feature gate disabled
- [ ] Feature gate enabled
- [ ] Full resolution flow
- [ ] Crash recovery (idempotency)
- [ ] Telemetry recording
- [ ] Multi-worker scenario
- [ ] Cache integration
- [ ] Breaker integration
- [ ] Rate limiter integration
- [ ] Backward compatibility

### Mocking Strategy

- Mock HTTP clients (head_client, raw_client)
- Mock breaker registry
- Mock rate limiter
- Mock telemetry sink
- Use `unittest.mock.patch` for adapters

### Checklist

- [ ] Create test_fallback.py
- [ ] Create fixtures for plan, orchestrator, mocks
- [ ] Implement all 50+ tests
- [ ] Test threading safety
- [ ] Test budget enforcement
- [ ] Test adapter behavior
- [ ] Test config loading
- [ ] Test integration flow
- [ ] Achieve 90%+ coverage

---

## Phase 10: Documentation (AGENTS.md updates)

**File**: `src/DocsToKG/ContentDownload/AGENTS.md` (MODIFY)
**Lines**: +300-400
**Complexity**: Low (documentation)

### New Section: "Fallback & Resiliency Strategy"

#### 10.1 Overview

- Explain tiered resolution
- Show tier ordering
- Explain budgets
- Explain cancellation

#### 10.2 Configuration

- Example YAML
- Environment variables
- CLI knobs
- Tuning guidelines

#### 10.3 CLI Operations

```bash
# Show plan
python -m DocsToKG.ContentDownload.cli fallback plan

# Dry-run resolution
python -m DocsToKG.ContentDownload.cli fallback dryrun --max 10

# Suggest tuning
python -m DocsToKG.ContentDownload.cli fallback tune
```

#### 10.4 Telemetry & SLOs

- Metrics to watch
- Dashboards
- Alerts

#### 10.5 Operational Playbooks

- **Scenario 1**: Tier stuck (e.g., landing scrape too slow)
- **Scenario 2**: Source unreliable (e.g., Wayback flaky)
- **Scenario 3**: High timeout rate
- **Scenario 4**: Low success rate

#### 10.6 Best Practices

- Start with tier 1 only
- Gradually enable tiers
- Monitor per-source success rate
- Tune timeouts from telemetry

### Checklist

- [ ] Add "Fallback & Resiliency" section to AGENTS.md
- [ ] Document configuration
- [ ] Document CLI commands
- [ ] Document telemetry/SLOs
- [ ] Document 4-5 operational playbooks
- [ ] Document best practices
- [ ] Add example YAML
- [ ] Add troubleshooting guide

---

## Implementation Order

**Week 1:**

1. Phase 1: Types (2 hours)
2. Phase 4: YAML config (1 hour)
3. Phase 5: Loader (3 hours)
4. Phase 2: Orchestrator (4 hours)

**Week 2:**

1. Phase 3: Adapters (6 hours, ~1 hour per adapter)
2. Phase 7: Telemetry (2 hours)
3. Phase 9: Tests (6 hours)

**Week 3:**

1. Phase 6: CLI (3 hours)
2. Phase 8: Integration (2 hours)
3. Phase 10: Documentation (2 hours)
4. Manual testing & fixes (3 hours)

**Total: ~35 hours (~1 month part-time)**

---

## Production Readiness

### Pre-Deployment Checklist

- [ ] All 50+ tests passing
- [ ] Code coverage ≥ 90%
- [ ] Linting clean
- [ ] Type hints complete
- [ ] Docstrings on all public APIs
- [ ] Feature gate defaults to OFF
- [ ] YAML validates on load
- [ ] Thread safety verified
- [ ] Budget enforcement tested
- [ ] Telemetry recording verified
- [ ] CLI commands working
- [ ] AGENTS.md comprehensive
- [ ] 1-week telemetry baseline
- [ ] SLO dashboard ready

### Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Threading contention | Low | High | Load test with 10+ concurrent |
| Adapter failure | Medium | Medium | Fallback to next tier |
| Timeout too aggressive | Medium | High | Start conservative, tune up |
| Telemetry overhead | Low | Low | Keep fields low-cardinality |
| Config complexity | Low | Medium | Validate on load |

---

## Success Criteria

✅ **Functional**

- [ ] 7 adapters working
- [ ] Tiered resolution working
- [ ] Budgets enforced
- [ ] Cancellation working
- [ ] Health gates working

✅ **Operational**

- [ ] CLI commands useful
- [ ] Telemetry comprehensive
- [ ] Configuration tunable
- [ ] Playbooks documented

✅ **Quality**

- [ ] 50+ tests passing
- [ ] Code coverage ≥90%
- [ ] Linting clean
- [ ] Type hints complete

✅ **Production**

- [ ] Feature gate defaults OFF
- [ ] Backward compatible
- [ ] Zero breaking changes
- [ ] Safe to deploy

---

## Next Steps

1. **Start Phase 1**: Create fallback/types.py
2. **Review types**: Ensure all necessary fields present
3. **Proceed Phase 2**: Create orchestrator.py
4. **Iterative testing**: Test as you go
5. **Integration**: Wire into download.py
6. **Documentation**: Update AGENTS.md

---

## Appendix: Tier Strategy

### Why This Tier Order?

1. **direct_oa** (Tier 1): Fastest, highest success, metadata-heavy (Unpaywall/arXiv/PMC)
2. **doi_follow** (Tier 2): Follow redirects, add ~1 hop per attempt
3. **landing_scrape** (Tier 3): Parse HTML, slower but covers publishers
4. **archive** (Tier 4): Slowest, highest latency, last resort

### Why Wayback is Last?

- Slower (CDX lookups, archive.org latency)
- Lower success rate for recent papers
- But excellent for historical papers
- Run only if earlier tiers exhaust budget

### Per-Tier Parallelism

- **Tier 1 (direct_oa)**: 2 parallel (Unpaywall + arXiv)
- **Tier 2 (doi_follow)**: 1 (already slow, no point parallelizing)
- **Tier 3 (landing_scrape)**: 2 (different publishers in parallel)
- **Tier 4 (archive)**: 1 (only one Wayback instance needed)

---

## References

- Original scope: `ContentDownload-optimization-9-fallback&resiliency` (attached)
- Circuit breaker integration: `breakers.py`, `networking.py`
- Rate limiter: `ratelimit.py`
- Existing resolvers: `resolvers/`
- Telemetry: `telemetry.py`
