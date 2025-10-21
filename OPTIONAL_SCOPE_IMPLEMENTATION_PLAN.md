# Optional Scope Implementation Plan - October 21, 2025

**Status**: Planning & Prioritization  
**Objective**: Implement remaining optional items systematically

---

## 1. Additional Gate Wiring (RECOMMENDED PRIORITY - 2-3 days)

### Current Status
- ✅ **config_gate**: Wired in planning.py (line 1864)
- ✅ **url_gate**: Wired in planning.py (line 1201)
- ❌ **filesystem_gate**: Not wired
- ❌ **extraction_gate**: Not wired
- ❌ **storage_gate**: Not wired
- ❌ **db_gate**: Not wired
- ❌ **db_boundary_gate**: Not wired (application boundary)

### Integration Points Remaining

| Gate | Current Use Point | Integration Location | Effort |
|------|------------------|----------------------|--------|
| `filesystem_gate` | Path validation | `planning.py:fetch_one()` | 4 hours |
| `extraction_gate` | Archive extraction | `io/filesystem.py:extract_archive_safe()` | 4 hours |
| `storage_gate` | Cache operations | `io/filesystem.py:save_artifact()` | 3 hours |
| `db_gate` | Database queries | `catalog/queries.py` | 3 hours |
| `db_boundary_gate` | Transaction boundaries | `catalog/boundaries.py` | 3 hours |

**Total Effort**: ~17 hours (2-3 focused days)

### Implementation Strategy

**Phase 1** (4 hours): Filesystem & Extraction Gates
- Wire `filesystem_gate` in artifact writing code
- Wire `extraction_gate` in extraction pipeline
- Test with existing test suite

**Phase 2** (3 hours): Storage Gate
- Wire `storage_gate` in cache operations
- Verify CAS mirror integration
- Add storage-specific telemetry

**Phase 3** (6 hours): Database Gates
- Wire `db_gate` in query operations
- Wire `db_boundary_gate` in transaction contexts
- Add database telemetry

**Outcome**: All 6 gates actively enforcing policy across the entire pipeline

---

## 2. ContentDownload Pydantic v2 Config Refactor (MEDIUM PRIORITY - 3-4 days)

### Current State
- Dataclass-based configuration scattered across modules
- No unified config loader
- Manual parsing of env vars and CLI args
- No resolver registry pattern

### Planned Components
1. **Pydantic v2 Models** (8-10 classes)
   - `RetryPolicy`
   - `HttpClientConfig`
   - `ContentDownloadConfig`
   - `RateLimitConfig`
   - etc.

2. **Config Loader** (loader.py)
   - File → Pydantic model
   - Env var overrides
   - CLI arg precedence

3. **Resolver Registry** (@register pattern)
   - Declarative resolver registration
   - Plugin discovery
   - Type-safe resolver configuration

4. **Unified API Types**
   - `DownloadPlan`
   - `DownloadOutcome`
   - `DownloadMetrics`

5. **CLI Modernization**
   - Typer-based CLI
   - `print-config` command
   - `validate-config` command
   - `explain` command

### Benefit
- Better configuration management
- Environment variable support
- config_hash for provenance
- Extensibility via registration pattern

**Effort**: 3-4 days (750-1000 LOC)  
**Risk**: Medium (touches many modules)  
**Backward Compat**: 100% maintained

---

## 3. Hishel HTTP Caching (LOW PRIORITY - 4-6 weeks)

### Current Planning State
- Comprehensive design document exists (HISHEL_CACHING_COMPREHENSIVE_PLAN.md)
- 5-phase implementation roadmap planned
- RFC 9111 compliant design

### Implementation Phases
1. **Phase 1**: cache_loader + cache_policy (1 week)
2. **Phase 2**: Dual HTTP clients (1 week)
3. **Phase 3**: Networking routing (1 week)
4. **Phase 4**: Telemetry integration (1 week)
5. **Phase 5**: Operations & monitoring (1 week)

### Expected Benefits
- >50% cache hit rate
- >30% bandwidth saved
- <50ms response time

**Effort**: 4-6 weeks (1500-2000 LOC)  
**Risk**: Low (well-planned, isolated)  
**Priority**: Performance optimization (not critical path)

---

## Recommended Implementation Order

### Week 1 (Highest ROI)
**Task**: Additional Gate Wiring (2-3 days)
- Complete all 6 gates integration
- Add gate telemetry
- Run integration tests
- **ROI**: Immediate security benefit, minimal disruption

**Task**: Begin ContentDownload Refactor (2-3 days)
- Design Pydantic v2 models
- Implement config loader
- Create migration guide

### Week 2+
**Task**: Complete ContentDownload Refactor (2-3 more days)
- Finish resolver registry
- Implement CLI modernization
- Comprehensive testing
- Migration to production

**Task**: Hishel HTTP Caching (4-6 weeks as time permits)
- Phase-by-phase implementation
- Performance monitoring
- Gradual rollout

---

## Decision Framework

### Should I implement Additional Gate Wiring?
**YES - Recommended**
- ✅ All gates already fully implemented
- ✅ Just need integration (straightforward)
- ✅ Immediate security/policy enforcement benefit
- ✅ 2-3 focused days
- ✅ Low risk (gates are battle-tested)
- ✅ High ROI (full pipeline protection)

### Should I implement ContentDownload Pydantic v2 Refactor?
**MAYBE - If time permits**
- ✅ Well-scoped (3-4 days)
- ✅ Medium complexity (touches many files)
- ✅ High value (better config management)
- ⚠️ Requires careful testing
- ⚠️ Migration path needed for users

### Should I implement Hishel HTTP Caching?
**NO - Skip for now**
- ✅ Excellent design exists
- ❌ Takes 4-6 weeks
- ❌ Performance optimization (not critical)
- ❌ Can be done later as priorities change
- 📍 Better to complete other items first

---

## Quality Gates

**All implementations must meet**:
- ✅ 100% type-safe (Pydantic v2 or strict types)
- ✅ 95%+ test pass rate
- ✅ 0 linting violations
- ✅ 0 breaking changes (backward compatible)
- ✅ Comprehensive telemetry
- ✅ Clear documentation

---

## Success Criteria

### Gate Wiring Success
- [ ] All 6 gates actively wired into pipeline
- [ ] 100% gate test pass rate
- [ ] Telemetry showing gate metrics
- [ ] Zero security gaps in pipeline

### ContentDownload Refactor Success
- [ ] Pydantic v2 models fully migrated
- [ ] Config loader working with all sources
- [ ] Resolver registry operational
- [ ] CLI commands working
- [ ] 100% backward compatible

### Overall Project Success
- [ ] 95%+ of all tests passing
- [ ] 0 regression in functionality
- [ ] Improved maintainability metrics
- [ ] Production deployable

---

## Recommendation

**Implement in this order**:
1. ✅ **Additional Gate Wiring** (2-3 days) - DO THIS
2. ✅ **ContentDownload Refactor** (3-4 days) - DO THIS IF TIME
3. ⏳ **Hishel HTTP Caching** (4-6 weeks) - DEFER FOR LATER

**Target**: Complete gate wiring this week, start ContentDownload refactor next week

