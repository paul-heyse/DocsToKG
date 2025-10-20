# Hishel Caching Implementation - Planning Summary

**Date**: October 21, 2025  
**Scope**: ContentDownload-optimization-3-hishel.md  
**Status**: ✅ COMPLETE PLANNING & SPECIFICATION  
**Total Documentation**: 3 comprehensive guides

---

## Executive Summary

This planning phase has created a **complete, production-ready specification** for integrating Hishel RFC-9111-compliant HTTP caching into DocsToKG ContentDownload. The work is ready for phased implementation.

### What Was Delivered

| Document | Purpose | Size | Key Sections |
|----------|---------|------|--------------|
| **HISHEL_CACHING_COMPREHENSIVE_PLAN.md** | Architecture overview, design principles, phased roadmap | 400 lines | 12 parts: overview, components, config, integration, testing, ops, metrics, risks |
| **HISHEL_IMPLEMENTATION_SPECIFICATION.md** | Exact interfaces, algorithms, data structures | 500 lines | 8 parts: module specs, flows, config format, data contracts, error handling, tests, checklists |
| **This Summary** | Quick reference, decision rationale, next steps | 300 lines | Overview + key decisions |

**Total**: 1,200+ lines of detailed specification, ready for implementation.

---

## Part 1: Design Decisions & Rationale

### Decision 1: Dual Client Architecture

**Decision**: Maintain two separate HTTPX clients:
- **Cached Client**: Hishel CacheTransport → RateLimitedTransport → HTTPTransport
- **Raw Client**: RateLimitedTransport → HTTPTransport (no cache)

**Rationale**:
- Artifacts (PDFs) never cached → guaranteed fresh for large files
- Unknown hosts not cached by default → conservative, explicit opt-in
- Cached hits skip rate limiter → efficiency win for metadata/landing
- All requests still protected by breaker/limiter/Tenacity → safety

**Alternative Considered**: Single client with per-request cache enable/disable
- Rejected: Harder to reason about, potential cache coherence issues with artifacts

---

### Decision 2: Role-Based Cache Policy

**Decision**: Cache decisions based on `(host, role)` tuples:
- `metadata`: Cached (API responses, structured data)
- `landing`: Cached (HTML pages, article pages)
- `artifact`: Never cached (PDFs, large documents)

**Rationale**:
- Metadata/landing have high cache potential (stable across runs)
- Artifacts are typically large, rarely repeated → no cache benefit
- Roles encode request intent explicitly → clear semantics

**Alternative Considered**: Mime-type based detection
- Rejected: Requires response headers, too late for routing decision

---

### Decision 3: Conservative Default ("DO_NOT_CACHE")

**Decision**: Unknown hosts are **NOT cached** by default. Explicit opt-in via YAML.

**Rationale**:
- Production safety: unknown hosts won't accidentally fill cache
- Ops controls policy: explicit config in `cache.yaml` for every cached host
- Transparent behavior: no surprises when adding new resolvers

**Alternative Considered**: Heuristic opt-out (cache everything except blacklist)
- Rejected: Too risky for production, violates principle of least surprise

---

### Decision 4: Per-Host TTL + Per-Role Override

**Decision**: Hierarchical TTL policy:
1. Try role-specific TTL (e.g., `api.crossref.org:metadata=259200`)
2. Fall back to host TTL (e.g., `api.crossref.org=172800`)
3. Fall back to global default (e.g., `DO_NOT_CACHE`)

**Rationale**:
- Flexibility for high-value metadata endpoints (longer TTL)
- Simplicity for most hosts (single TTL)
- Ops can tune without code changes

**Example**:
```yaml
api.crossref.org:
  ttl_s: 172800              # 2 days (default for all roles)
  role:
    metadata:
      ttl_s: 259200          # 3 days for metadata (override)
      swrv_s: 180            # 3 min stale-while-revalidate
```

---

### Decision 5: Stale-While-Revalidate (SWrV) for Metadata Only

**Decision**: SWrV enabled only for `role=metadata`, not for landing/artifacts.

**Rationale**:
- Metadata APIs benefit from stale tolerance (rarely changes)
- Landing pages should be fresher (user-visible content)
- Artifacts never cached → SWrV irrelevant

---

### Decision 6: RFC 9111 Compliance Without Validation

**Decision**: Trust server `Cache-Control`, `ETag`, `Last-Modified` headers. Don't invent TTLs.

**Rationale**:
- Server is source of truth for cacheability
- If server provides validators, use revalidation (304 flows)
- If server doesn't set validators, respect server intent

**Safeguard**: `allow_heuristics: false` in controller → Hishel won't guess TTLs

---

### Decision 7: Offline Mode with only-if-cached

**Decision**: `--offline` flag adds `Cache-Control: only-if-cached` to cached requests.

**Behavior**:
- Cache hit → return from cache (deterministic, fast)
- Cache miss → 504 Unsatisfiable Request (Hishel behavior)
- Pipeline sees 504 → skip download, log `blocked_offline`

**Rationale**:
- RFC 9111 compliant behavior
- Fast, deterministic for testing/verification runs
- Transparent to caller (just returns 504)

---

## Part 2: Architecture at a Glance

```
Request Flow
─────────────────────────────────────────────────────────────

  network.request_with_retries(url, role=..., cache_router=...)
                        ↓
  cache_router.resolve_policy(host, role)
                        ↓
        CacheDecision(use_cache, ttl_s, swrv_s)
                        ↓
     ┌──────────────────┴──────────────────┐
     │                                     │
  if use_cache          if not use_cache
     │                         │
  CACHED                     RAW
  CLIENT                    CLIENT
     │                         │
  Hishel────────────────────────────RateLimited
  CacheTransport                    Transport
     │                         │
     └──────────────────┬──────────────────┘
                        ↓
              HTTPTransport (network)
                        ↓
            Hishel extracts telemetry
                        ↓
            telemetry.record_cache_hit()
                        ↓
         Response with cache metadata
```

---

## Part 3: Module Structure

### New Files

1. **cache_loader.py** (NEW)
   - Load YAML configuration
   - Apply env/CLI overlays
   - Normalize host keys (IDNA 2008 + UTS #46)
   - Validate invariants
   - Return `CacheConfig` dataclass

2. **cache_policy.py** (NEW)
   - `CacheDecision` dataclass
   - `CacheRouter` class with `resolve_policy()` method
   - Print effective policy table for ops

### Modified Files

3. **httpx_transport.py** (MODIFIED)
   - Add dual client builders: `_build_cached_client()`, `_build_raw_client()`
   - Add `configure_cache_config(config)` for startup
   - Add `get_cached_http_client()`, `get_raw_http_client()`
   - Keep backward compatibility with existing APIs

4. **networking.py** (MODIFIED)
   - Add `_apply_cache_extensions(request, decision, offline=...)`
   - Add `_extract_cache_telemetry(response)`
   - Modify `request_with_retries()` signature to include `role`, `offline`, `cache_router`
   - Implement cache routing logic

5. **telemetry.py** (MODIFIED)
   - Add 5 cache counters (hit, revalidated, stale, stored, offline_504)
   - Add bandwidth_saved counter
   - Add `record_cache_*()` functions
   - Add `build_cache_summary()` for run summary

6. **args.py** (MODIFIED)
   - Add `--cache-yaml` flag
   - Add `--cache-host` flag (per-host override)
   - Add `--cache-role` flag (per-host, per-role override)
   - Add `--offline` flag
   - Add `--cache-vacuum` flag (optional)

### New Configuration File

7. **config/cache.yaml** (NEW)
   - Template with known hosts (Crossref, OpenAlex, Wayback, etc.)
   - Per-host TTL and role-specific overrides
   - SWrV settings for metadata
   - Storage configuration (FileStorage, check_ttl_every_s)

---

## Part 4: Key Interfaces

### CacheConfig Dataclass

```python
@dataclass(frozen=True)
class CacheConfig:
    storage: CacheStorage                 # Storage backend config
    controller: CacheControllerDefaults   # Global RFC controller settings
    hosts: Dict[str, CacheHostPolicy]     # Per-host policies (normalized keys)
```

### CacheRouter Class

```python
class CacheRouter:
    def __init__(self, config: CacheConfig): ...
    def resolve_policy(self, host: str, role: str = "metadata") -> CacheDecision: ...
    def print_effective_policy(self) -> str: ...
```

### Updated request_with_retries()

```python
def request_with_retries(
    method: str,
    url: str,
    *,
    role: str = "metadata",              # NEW
    offline: bool = False,               # NEW
    cache_router: Optional[CacheRouter] = None,  # NEW
    **kwargs,
) -> httpx.Response: ...
```

---

## Part 5: Configuration Examples

### Conservative (Safe)
```yaml
controller:
  default: "DO_NOT_CACHE"
hosts:
  api.crossref.org: {ttl_s: 172800}      # 2 days
```

### Balanced (Recommended)
```yaml
controller:
  default: "DO_NOT_CACHE"
hosts:
  api.crossref.org:
    ttl_s: 172800
    role:
      metadata: {ttl_s: 259200, swrv_s: 180}
  api.openalex.org: {ttl_s: 172800}
  web.archive.org: {ttl_s: 172800}
```

### Aggressive (High Cache Potential)
```yaml
controller:
  default: "DO_NOT_CACHE"
hosts:
  api.crossref.org: {ttl_s: 259200, role: {metadata: {ttl_s: 518400, swrv_s: 300}}}
  api.openalex.org: {ttl_s: 259200, role: {metadata: {ttl_s: 518400, swrv_s: 300}}}
  # ... 20+ hosts with aggressive TTLs
```

---

## Part 6: Phased Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Implement `cache_loader.py` (YAML → env → CLI)
- [ ] Implement `cache_policy.py` (routing logic)
- [ ] Create `cache.yaml` template
- [ ] Add CLI flags to `args.py`
- **Output**: Configuration system ready, 20 unit tests

### Phase 2: HTTP Transport (Week 2)
- [ ] Modify `httpx_transport.py` (dual clients)
- [ ] Implement Hishel CacheTransport integration
- [ ] Add `configure_cache_config()` call to startup
- **Output**: Dual clients built and ready, 10 tests

### Phase 3: Networking Integration (Week 2-3)
- [ ] Modify `networking.py` (cache routing)
- [ ] Implement `request_with_retries()` changes
- [ ] Add offline mode support
- [ ] Implement telemetry extraction
- **Output**: Full request flow working, 20 integration tests

### Phase 4: Telemetry & Testing (Week 3)
- [ ] Add cache counters to `telemetry.py`
- [ ] Implement cache summary generation
- [ ] End-to-end integration tests
- [ ] RFC 9111 compliance validation
- **Output**: Full instrumentation, 30 tests

### Phase 5: Operations & Tuning (Week 4)
- [ ] Write operational runbooks
- [ ] Canary deployment with monitoring
- [ ] Performance tuning based on telemetry
- [ ] Documentation for ops
- **Output**: Production-ready, monitored

### Phase 6: Stretch Goals (Beyond)
- [ ] Stale-if-error (SIE) support
- [ ] Redis storage backend
- [ ] Dynamic host discovery
- [ ] Cache directory cap management

**Total**: 4-6 weeks for full implementation and deployment.

---

## Part 7: Success Metrics

### Performance Targets

| Metric | Target | Baseline | Expected Improvement |
|--------|--------|----------|----------------------|
| Cache Hit Rate | > 50% | 0% | +50-70% |
| Revalidation Rate | 20-40% | 0% | +20-40% |
| Bandwidth Saved | > 30% | 0% | +40-60% |
| Response Time | < 50ms | ~200-500ms | -75% |

### Operational Targets

- Configuration load time: < 100ms
- Cache policy lookup: O(1)
- Cache vacuum: < 5s for typical runs
- Telemetry overhead: < 1% CPU

### Quality Targets

- 100% test coverage for new code
- Zero cache coherence issues (served stale < 1%)
- Zero data corruption
- RFC 9111 compliance verified

---

## Part 8: Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Stale data served | Conservative TTL defaults, SWrV limits, startup validation |
| Cache coherence | RFC 9111 compliance, proper revalidation, validator headers |
| Performance regression | O(1) lookups, cached hits bypass limiter, early offline detection |
| Configuration errors | Schema validation, startup policy table, CLI preview mode |
| Full disk | Hishel LFU eviction, check_ttl_every_s garbage collection, ops monitoring |

**Rollback Plan**: Set `default: "DO_NOT_CACHE"` and redeploy. All requests route to raw client.

---

## Part 9: Pre-Implementation Checklist

✅ **Architecture**: Complete and documented  
✅ **Data structures**: Fully specified with validation  
✅ **Interfaces**: Every function signature defined  
✅ **Configuration format**: YAML schema complete  
✅ **Integration points**: Exactly where each module connects  
✅ **Error handling**: Edge cases documented  
✅ **Testing strategy**: Unit/integration/e2e tests outlined  
✅ **Operations**: Runbooks, monitoring, dashboards planned  
✅ **RFC 9111 compliance**: Design validated  
✅ **Phased rollout**: Week-by-week implementation plan  

---

## Part 10: Documentation Artifacts

### For Developers
- **HISHEL_IMPLEMENTATION_SPECIFICATION.md** - Exact code signatures, data structures, algorithms
- **Unit test templates** - Sample test cases for each module
- **Integration test templates** - End-to-end scenarios

### For Operations
- **HISHEL_CACHING_COMPREHENSIVE_PLAN.md** - Architecture, config examples, monitoring
- **cache.yaml template** - Pre-configured with known hosts
- **Operational runbooks** - Deployment, tuning, troubleshooting

### For Product/Analytics
- **Success metrics** - Hit rate, revalidation rate, bandwidth saved
- **Run summary format** - Cache performance included in output
- **Telemetry counters** - Per-host, per-role metrics

---

## Part 11: Next Steps

### Immediate (Before Implementation)
1. ✅ Read all three planning documents
2. ✅ Review design decisions in Part 1
3. ✅ Validate architecture with team
4. ✅ Confirm dependency versions (hishel 0.1.5+, httpx 0.28+)

### Phase 1 Kickoff
1. Create `cache_loader.py` skeleton
2. Define all dataclasses from specification
3. Write YAML parsing logic
4. Implement env/CLI overlay
5. Write 20 unit tests

### Ongoing
1. Follow phased roadmap (4-6 weeks)
2. Reference specification for exact interfaces
3. Use test templates as starting point
4. Update this summary after each phase

---

## Conclusion

✅ **Complete planning delivered**:
- 1,200+ lines of specification
- 3 comprehensive documents
- Exact interfaces and algorithms
- Phased implementation roadmap
- Risk mitigation strategies
- Success metrics
- Operational readiness

✅ **Ready for implementation**:
- No architectural ambiguity
- Clear integration points
- Detailed test strategy
- Production deployment plan

**Next action**: Begin Phase 1 (cache_loader.py) implementation using the detailed specification as the source of truth.

---

**Planning completed by**: Comprehensive analysis & design  
**Date**: October 21, 2025  
**Status**: ✅ READY FOR IMPLEMENTATION  
**Estimated Implementation**: 4-6 weeks  
**Risk Level**: LOW (conservative defaults, RFC compliant, extensive testing)

