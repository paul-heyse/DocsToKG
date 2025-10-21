# Phase 8: Fallback & Resiliency Strategy - Complete Implementation Plan

**Date**: October 21, 2025
**Status**: FULL IMPLEMENTATION IN PROGRESS
**Scope**: 3,500+ LOC across orchestrator, adapters, loader, CLI, integration, tests
**Target**: Complete production-ready fallback resolution system

---

## Execution Strategy

### Phase 8.1: Fix & Complete Core Types ✅ DONE
- [x] Fixed AttemptResult to match actual adapter usage
- [x] Verified mypy clean
- [x] Verified ruff clean
- [x] All dataclasses validated

### Phase 8.2: Orchestrator Implementation (2-3 hours)
**File**: `src/DocsToKG/ContentDownload/fallback/orchestrator.py`

**Work**:
1. FallbackOrchestrator class structure
   - `__init__(plan, breaker, rate_limiter, clients, telemetry, logger)`
   - `resolve_pdf(context, adapters)` - main entry point
   - `_health_gate(source_name, context)` - breaker/offline check
   - `_emit_telemetry(tier_name, result, context)` - event logging

2. Threading & concurrency
   - ThreadPoolExecutor for tier parallelization
   - Thread-safe result collection
   - Timeout enforcement
   - Cancellation support

3. Budget enforcement
   - Total timeout tracking
   - Attempt counter
   - Concurrent thread limit

4. Health gates
   - Circuit breaker state check
   - Offline mode detection
   - Rate limiter awareness

### Phase 8.3: Complete Adapters (2-3 hours)
**Directory**: `src/DocsToKG/ContentDownload/fallback/adapters/`

**7 Adapters to verify/complete**:
1. `unpaywall.py` - Unpaywall API
2. `arxiv.py` - arXiv metadata
3. `pmc.py` - PubMed Central
4. `doi_redirect.py` - DOI resolver
5. `landing_scrape.py` - Landing page scraping
6. `europe_pmc.py` - Europe PMC
7. `wayback.py` - Wayback Machine

**Per-adapter tasks**:
- Fix type annotations (status, host, meta fields)
- Implement resolve() method signature
- Add error handling
- Add retry logic
- Add telemetry emission
- Validate against specification

### Phase 8.4: Loader Completion (1-2 hours)
**File**: `src/DocsToKG/ContentDownload/fallback/loader.py`

**Work**:
1. YAML config parsing
2. Default plan generation
3. Validation logic
4. CLI override merging
5. Environment variable support

### Phase 8.5: Integration into download.py (2 hours)
**File**: `src/DocsToKG/ContentDownload/download.py`

**Work**:
1. Add `--enable-fallback-strategy` CLI flag
2. Initialize FallbackOrchestrator in process_one_work()
3. Call orchestrator before resolver pipeline (first attempt)
4. Fall back to resolver pipeline on failure
5. Wire telemetry correlation
6. Add feature gate (ENABLE_FALLBACK_STRATEGY env var)

### Phase 8.6: CLI Commands (1 hour)
**File**: `src/DocsToKG/ContentDownload/fallback/cli_fallback.py`

**Work**:
1. `fallback plan` - Show effective plan
2. `fallback dryrun` - Simulate resolution (no network)
3. `fallback stats` - Statistics from recent runs

### Phase 8.7: Comprehensive Tests (3-4 hours)
**Directory**: `tests/content_download/test_fallback_*.py`

**Test suites**:
1. Unit tests (types, policies, budgets) - 20 tests
2. Adapter tests (mocked HTTP) - 30 tests
3. Orchestrator tests (threading, budgets) - 20 tests
4. Integration tests (full flow) - 15 tests
5. E2E tests (real adapters) - 15 tests

**Target**: 100+ tests, 100% passing

### Phase 8.8: Verification & Deployment (1-2 hours)
**Work**:
1. Full mypy/ruff check
2. All tests passing
3. Performance profiling
4. Documentation update
5. Git commit
6. Completion report

---

## Success Criteria

- [ ] All 3,500+ LOC implemented
- [ ] 0 type errors (mypy clean)
- [ ] 0 linting errors (ruff clean)
- [ ] 100+ tests passing
- [ ] Zero breaking changes
- [ ] 100% backward compatible
- [ ] Feature gate working (off by default)
- [ ] Production-ready code
- [ ] Documentation complete

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Threading bugs | Medium | Thorough thread testing, timeouts |
| Adapter failures | Medium | Mock testing, fallback to pipeline |
| Performance impact | Low | Async where possible, budgets enforce limits |
| Type errors | Low | Mypy strict mode, comprehensive tests |
| Integration issues | Medium | Feature gate allows gradual rollout |

---

## Timeline

| Phase | Effort | Status |
|-------|--------|--------|
| 8.1 Core Types | 0.5h | ✅ DONE |
| 8.2 Orchestrator | 2.5h | ⏳ IN PROGRESS |
| 8.3 Adapters | 2.5h | ⏳ PENDING |
| 8.4 Loader | 1.5h | ⏳ PENDING |
| 8.5 Integration | 2h | ⏳ PENDING |
| 8.6 CLI | 1h | ⏳ PENDING |
| 8.7 Tests | 4h | ⏳ PENDING |
| 8.8 Verification | 1.5h | ⏳ PENDING |
| **Total** | **15.5h** | **IN PROGRESS** |

---

## Next Step

Proceed with Phase 8.2: Orchestrator Implementation

