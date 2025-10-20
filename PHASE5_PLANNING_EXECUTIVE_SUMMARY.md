# Phase 5: Settings System - Executive Summary

**Created**: October 20, 2025
**Status**: Planning Complete ✓ — Ready for Implementation Phase 5.1
**Timeline**: 30 days (4-6 weeks with normal velocity)
**Risk**: MEDIUM (well-scoped but complex)
**Breaking Changes**: ZERO (fully backward compatible)

---

## What We're Building

A **production-grade Pydantic v2 + `pydantic-settings` configuration system** that:

- Centralizes all 50+ OntologyDownload config fields into one validated source-of-truth
- Supports **full source precedence**: CLI → config file → .env → environment → defaults
- Provides **typed validation** (no more string parsing in consumers)
- Enables **deterministic testing** (inject configs via fixtures)
- Delivers **observability** (config provenance, hashing, source tracking)

---

## Why Now?

1. **Database Phase (Phase 4) is complete** — stable foundation
2. **Configuration is scattered** — environment reads throughout codebase
3. **Testing is hard** — no principled way to inject configs
4. **We're past the "maybe" stage** — scope docs are detailed and approved
5. **This enables future work** — Hishel caching, HTTP hardening, etc.

---

## Architecture at a Glance

```
Settings (root, BaseSettings)
├── http (11 fields: timeouts, pool, HTTP/2, UA, proxy trust)
├── cache (3 fields: Hishel settings)
├── retry (3 fields: connect retries, backoff)
├── security (5 fields: allowed hosts/ports, DNS flags)
├── ratelimit (4 fields: global + per-service quotas)
├── extraction (23 fields: safety, throughput, integrity)
├── storage (3 fields: local blob root)
├── db (5 fields: DuckDB catalog settings)
├── logging (2 fields: level, JSON mode)
├── telemetry (2 fields: run ID, event emission)
└── helpers: config_hash(), resolve_sources(), normalize_allowed_hosts(), etc.
```

**Total: 10 domains, 63 fields, 50+ environment variables, all validated**

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic v2 | Speed (Rust core), features, ecosystem, future-proof |
| BaseSettings | Built-in env loading, dotenv, source precedence |
| 10 sub-models | Clarity, reusability, extensibility, testability |
| Singleton `get()` | Cache expensive operations, testable, no global state |
| Immutable models | Prevent accidental mutation mid-run |
| No secrets in hash | Config hashing for provenance; secrets stay safe |
| CSV parsing | Human-friendly env vars; strict validation in validators |
| Frozen models | Configuration is read-only once loaded |

---

## Implementation Roadmap

### Phase 5.1: Foundation (Days 1-5)

- HttpSettings, CacheSettings, RetrySettings, LoggingSettings, TelemetrySettings
- Validators for timeouts, paths, levels
- 40-50 unit tests

### Phase 5.2: Complex Domains (Days 6-12)

- SecuritySettings (host/port/CIDR parsing)
- RateLimitSettings (rate string parsing, per-service)
- ExtractionSettings (safety + throughput + integrity)
- StorageSettings, DuckDBSettings
- 80-100 unit tests

### Phase 5.3: Root & Loading (Days 13-18)

- Root Settings model with all sub-models
- Source precedence logic (CLI → config → .env → env → defaults)
- Singleton getter with caching
- Config hashing and source resolution
- 30-40 integration tests

### Phase 5.4: Integration & Backward Compat (Days 19-24)

- Wire into existing builders (no breaking changes)
- CLI context injection
- Update planning, validation, io modules
- Ensure all tests pass
- 20-30 integration tests

### Phase 5.5: Docs & Polish (Days 25-30)

- `.env.example` with all 50+ fields
- `SETTINGS.md` reference (env matrix, examples, FAQs)
- Migration guide for existing code
- Performance validation
- Code review prep

**Total: 150-200 tests, >90% coverage, zero regressions**

---

## What Changes? What Stays the Same?

### ✅ Stays the Same (Backward Compatible)

- All existing CLI commands work identically
- All existing public builder functions stay (e.g., `build_download_config()`)
- All existing function signatures unchanged
- All existing tests pass without modification
- Zero breaking changes

### ✨ Changes (Internals)

- Settings loaded once via BaseSettings + singleton getter
- Environment reads replaced with `settings.get()`
- All validation happens at load time (not repeatedly at use time)
- Tests can inject custom settings via fixtures
- Consumers read from typed sub-models (e.g., `s.http.timeout_connect`)

---

## Success Criteria (Checklist)

- [ ] All 50+ fields implemented with proper validation
- [ ] 150-200 tests passing, >90% coverage
- [ ] Source precedence verified (CLI → config → .env → env → defaults)
- [ ] Settings load in < 100ms (cached < 1μs)
- [ ] Config hashing stable and reproducible
- [ ] All models frozen/immutable
- [ ] Zero breaking changes to public API
- [ ] Migration guide complete
- [ ] Documentation comprehensive
- [ ] Code review approved

---

## Files Involved

### New Files (Created)

- `PHASE5_SETTINGS_IMPLEMENTATION_PLAN.md` (this detailed plan)
- `PHASE5_PLANNING_EXECUTIVE_SUMMARY.md` (this summary)
- `tests/ontology_download/test_settings_domain_models.py`
- `tests/ontology_download/test_settings_security.py`
- `tests/ontology_download/test_settings_ratelimit.py`
- `tests/ontology_download/test_settings_extraction.py`
- `tests/ontology_download/test_settings_storage_db.py`
- `tests/ontology_download/test_settings_loading.py`
- `tests/ontology_download/test_settings_integration.py`
- `.env.example`
- `docs/06-operations/SETTINGS.md`

### Modified Files (Expanded/Refactored)

- `src/DocsToKG/OntologyDownload/settings.py` — MAJOR EXPANSION (domain models + root Settings + loaders)
- `src/DocsToKG/OntologyDownload/exports.py` — Add Settings to public API
- `src/DocsToKG/OntologyDownload/cli.py` — Wire Settings into typer context
- `src/DocsToKG/OntologyDownload/planning.py` — Read from settings
- `src/DocsToKG/OntologyDownload/validation.py` — Read from settings
- `src/DocsToKG/OntologyDownload/resolvers.py` — Read rate limits from settings
- `src/DocsToKG/OntologyDownload/io/network.py` — Read HTTP config from settings

---

## Environment Variable Naming Convention

All settings use **`ONTOFETCH_*` prefix** with **nested double-underscores**:

```bash
# HTTP settings
ONTOFETCH_HTTP__HTTP2=true
ONTOFETCH_HTTP__TIMEOUT_CONNECT=5
ONTOFETCH_HTTP__TIMEOUT_READ=30
# ... etc

# Security settings
ONTOFETCH_SECURITY__ALLOWED_HOSTS="ebi.ac.uk,*.purl.org,10.0.0.7"
ONTOFETCH_SECURITY__ALLOWED_PORTS="80,443,8443"
# ... etc

# Rate limits
ONTOFETCH_RATELIMIT__DEFAULT="8/second"
ONTOFETCH_RATELIMIT__PER_SERVICE="ols:4/second;bioportal:2/second"
# ... etc
```

Total: **50+ environment variables**, all strictly validated

---

## Testing Strategy

### Unit Tests (150-160)

- Domain model defaults, validation, normalization
- Field validators and edge cases
- Helper functions (rate parsing, host normalization, glob compilation)
- Serialization round-trips

### Integration Tests (30-40)

- Settings loading from each source
- Source precedence verification
- Singleton caching
- Config hashing

### End-to-End Tests (20-30)

- CLI receives settings via context
- Planning layer reads from settings
- Validation applies extraction policy
- Downloader respects rate limits

### Fixture Strategy

- `temp_env`: Override env vars + clear singleton cache
- `settings_from_dict`: Create Settings from test dict
- Per-test isolation to prevent state leakage

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Circular imports | MEDIUM | Lazy imports in validators, TYPE_CHECKING guards |
| Thread safety | MEDIUM | Use `functools.lru_cache`, frozen models |
| Performance regression | LOW | Profile load time, cache expensive operations |
| Breaking changes | HIGH | Extensive backward-compat tests, keep all builders |
| Validation too strict | MEDIUM | Lenient defaults, clear error messages, migration guide |

---

## How This Fits Into the Bigger Picture

### Completed Before Phase 5

- ✅ **Phase 1-3**: URL Canonicalization (Oct 21)
- ✅ **Phase 2-4**: DuckDB Database Deployment (Oct 18-20)
- ✅ **Phase 4**: Plan Caching (Oct 19-20)

### Phase 5 (Starting Soon)

- 🔄 **Phase 5**: Pydantic Settings System (this plan)
- ⏳ Will enable Hishel HTTP caching (planned next)
- ⏳ Will improve HTTP hardening (planned later)

### After Phase 5

- Phase 6+: HTTP caching, hardening, optimization

---

## What I'm Ready To Do

**Starting immediately upon your approval:**

1. **Week 1**: Implement Phase 5.1-5.2 (domain models, complex parsing, 120+ tests)
2. **Week 2-3**: Implement Phase 5.3-5.4 (root settings, loading, integration, backward compat)
3. **Week 4**: Phase 5.5 (docs, examples, migration guide, finalization)

---

## Questions for Clarification (Optional)

Before I start coding, would you like me to:

1. **Expand any particular domain?** (e.g., more detail on extraction policy fields?)
2. **Change the phase breakdown?** (e.g., combine phases or split differently?)
3. **Adjust the scope?** (e.g., skip certain fields for Phase 5, defer to Phase 6?)
4. **Add specific test patterns?** (e.g., mutation testing, performance tests?)
5. **Include specific documentation?** (e.g., migration scripts, troubleshooting guide?)

---

## Next Step: Implementation

Assuming approval, the plan is to:

1. ✅ **This week (Oct 20-24)**: Phase 5.1 complete (domain models foundation)
2. ✅ **Next week (Oct 27-31)**: Phase 5.2 complete (complex domains)
3. ✅ **Week 3 (Nov 3-7)**: Phase 5.3 complete (root settings + loading)
4. ✅ **Week 4 (Nov 10-14)**: Phase 5.4 complete (integration + backward compat)
5. ✅ **Week 5 (Nov 17-21)**: Phase 5.5 complete (docs + finalization)

**Target completion: November 21, 2025**

---

## How to Read the Full Plan

For detailed specifications, see:

- **`PHASE5_SETTINGS_IMPLEMENTATION_PLAN.md`** (637 lines)
  - Full field specifications (50+ fields with types, defaults, validation)
  - Complete environment variable reference
  - Detailed implementation tasks per phase
  - Test strategy and fixture patterns
  - Risk assessment and mitigation
  - File structure and timeline
  - Backward compatibility strategy

This summary covers the essentials; the full plan has the complete technical specification.

---

## Conclusion

Phase 5 is a **well-defined, low-risk, high-value initiative** that:

- ✅ Improves code clarity (typed settings, no string parsing)
- ✅ Enables better testing (inject configs via fixtures)
- ✅ Enhances observability (config provenance, hashing)
- ✅ Maintains full backward compatibility (zero breaking changes)
- ✅ Unblocks future work (Hishel caching, HTTP hardening)

**Ready to proceed upon your approval!**
