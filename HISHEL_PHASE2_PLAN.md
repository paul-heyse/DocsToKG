# Hishel Phase 2: HTTP Transport Integration - Implementation Plan

**Date**: October 21, 2025
**Status**: 🔨 IN PROGRESS
**Focus**: Item 2 - Cache-Control Directive Implementation

---

## Phase 2 Roadmap

### Item 1: Integrate Caching into HTTP Transport ✅ (Ready)

**Foundation**: Already completed in Phase 1

- ✅ CacheRouter resolves policies per (host, role)
- ✅ CacheTransport is already in use in httpx_transport.py
- ✅ Hishel is integrated with FileStorage backend
- **Next**: Integrate CacheRouter decision logic

### Item 2: Implement Cache-Control Directives 🔨 (THIS ITEM)

**Objective**: Parse and enforce RFC 9111 cache-control headers

**Components to Build**:

1. **`cache_control.py`** - Parse and enforce cache-control headers
   - Parse RFC 9111 cache-control directives
   - Extract: `no-cache`, `no-store`, `max-age`, `s-maxage`, `public`, `private`
   - Compute cache validity (fresh vs stale)

2. **`cache_transport_wrapper.py`** - Role-aware caching wrapper
   - Use CacheRouter to determine if response should be cached
   - Apply cache-control directives from response headers
   - Enforce role-based caching decisions
   - Feed policies to Hishel Controller

3. **Integration into `httpx_transport.py`**
   - Use CacheRouter when creating CacheTransport
   - Apply role-aware controller configuration
   - Log cache decisions via telemetry

### Item 3: Handle Conditional Requests ⏳ (Next)

- Conditional requests with `ETag`, `Last-Modified`
- `If-None-Match`, `If-Modified-Since` header handling
- 304 (Not Modified) response processing

### Item 4: Manage Cache Expiration & Invalidation ⏳ (Next)

- Expiration logic based on max-age
- Cache invalidation mechanisms
- Stale-while-revalidate implementation

### Item 5: Optimize Cache Storage ⏳ (Next)

- LFU eviction policies
- Storage backend configuration
- Persistent cache across sessions

---

## Implementation Details for Item 2

### 2.1: `cache_control.py` Module

**Responsibility**: Parse and interpret RFC 9111 cache-control headers

**Public API**:

```python
@dataclass(frozen=True)
class CacheControlDirective:
    no_cache: bool                    # Must revalidate
    no_store: bool                    # Must not cache
    public: bool                      # Cache in shared caches
    private: bool                     # Cache in private caches only
    max_age: Optional[int]            # Seconds (absolute freshness)
    s_maxage: Optional[int]           # Seconds (shared cache freshness)
    must_revalidate: bool             # Must not serve stale
    stale_while_revalidate: int = 0   # Serve stale for N seconds
    stale_if_error: int = 0           # Serve stale on error for N seconds

def parse_cache_control(headers: Mapping[str, str]) -> CacheControlDirective: ...

def is_fresh(directive: CacheControlDirective, age_s: float) -> bool: ...
```

### 2.2: `cache_transport_wrapper.py` Module

**Responsibility**: Wrap CacheTransport with role-aware policy enforcement

**Public API**:

```python
class RoleAwareCacheTransport(httpx.BaseTransport):
    """Wraps Hishel CacheTransport with CacheRouter decision logic."""

    def __init__(
        self,
        cache_router: CacheRouter,
        inner_transport: httpx.BaseTransport,
        storage: hishel.BaseStorage,
    ) -> None: ...

    def handle_request(self, request: httpx.Request) -> httpx.Response: ...

def build_role_aware_cache_transport(
    cache_router: CacheRouter,
    base_transport: httpx.BaseTransport,
    storage_path: Path,
) -> RoleAwareCacheTransport: ...
```

### 2.3: Integration into `httpx_transport.py`

**Changes**:

1. Load cache config during client creation
2. Create CacheRouter instance
3. Use RoleAwareCacheTransport wrapper instead of plain CacheTransport
4. Apply cache-control directives to Hishel Controller
5. Log cache decisions in telemetry

---

## Implementation Sequence

### Step 1: Create `cache_control.py`

- [ ] Define CacheControlDirective dataclass
- [ ] Implement parse_cache_control()
- [ ] Implement is_fresh()
- [ ] Write comprehensive tests

### Step 2: Create `cache_transport_wrapper.py`

- [ ] Define RoleAwareCacheTransport
- [ ] Implement handle_request()
- [ ] Integrate CacheRouter decisions
- [ ] Write comprehensive tests

### Step 3: Integrate into `httpx_transport.py`

- [ ] Load cache config from yaml/env/cli
- [ ] Create CacheRouter instance
- [ ] Build RoleAwareCacheTransport
- [ ] Update _create_client_unlocked()
- [ ] Test end-to-end flow

### Step 4: Add Telemetry

- [ ] Log cache decisions per request
- [ ] Track cache hit rate by host/role
- [ ] Add metrics to telemetry.py

### Step 5: Write Integration Tests

- [ ] Test cache hits/misses
- [ ] Test cache-control directives
- [ ] Test role-based routing
- [ ] Test policy enforcement

---

## Architecture Diagram

```
HTTP Request
    ↓
RoleAwareCacheTransport
    ├─ Extract host + role from request
    ├─ CacheRouter.resolve_policy(host, role)
    │   ├─ Check config for host/role
    │   ├─ Return CacheDecision (use_cache, ttl_s, swrv_s)
    ├─ Parse Response Cache-Control header
    ├─ Enforce CacheDecision
    ├─ Create Hishel Controller with directives
    ├─ Delegate to Hishel CacheTransport
    │   ├─ Check cache (hit → return)
    │   ├─ Cache miss → delegate to RateLimitedTransport
    │   ├─ Store response in cache
    │   └─ Respect cache-control headers
    └─ Return Response
         ↓
    Telemetry (cache_hit, ttl, etc.)
```

---

## Success Criteria

### Functionality

- [x] Parse all RFC 9111 cache-control directives
- [ ] Respect `no-cache` and `no-store` directives
- [ ] Enforce `max-age` freshness
- [ ] Apply role-based caching decisions
- [ ] Support `stale-while-revalidate`

### Integration

- [ ] CacheRouter integrated into transport layer
- [ ] All (host, role) combinations properly routed
- [ ] Conservative defaults applied (unknown hosts not cached)
- [ ] Artifacts never cached (by design)

### Quality

- [ ] 20+ tests for cache_control.py
- [ ] 15+ tests for cache_transport_wrapper.py
- [ ] 10+ integration tests for httpx_transport.py
- [ ] 100% test pass rate

### Telemetry

- [ ] Cache hit/miss rates tracked
- [ ] Per-host cache statistics
- [ ] Per-role cache statistics
- [ ] Performance impact measured

---

## Key Design Decisions

### Decision 1: Role Detection

**Choice**: Extract role from request.extensions["docs_request_role"]
**Rationale**: Role already set by caller (networking.py)
**Benefit**: No need to guess role from URL patterns

### Decision 2: Conservative Default for Missing Host

**Choice**: Do not cache if host not in config
**Rationale**: Safer for scrapers; prevent unintended caching
**Benefit**: Explicit allowlist protection

### Decision 3: Artifact Never Cached

**Choice**: CacheRouter always returns use_cache=False for artifact role
**Rationale**: Artifact content too variable
**Benefit**: Predictable, auditable behavior

### Decision 4: Layered Transport Stack

**Choice**: RoleAwareCacheTransport → Hishel CacheTransport → RateLimitedTransport → HTTPTransport
**Rationale**: Separation of concerns, clean integration points
**Benefit**: Easy to test, modify, or disable any layer

---

## File Structure

```
src/DocsToKG/ContentDownload/
├── cache_control.py                    # NEW: RFC 9111 parsing
├── cache_transport_wrapper.py           # NEW: Role-aware transport
├── httpx_transport.py                   # MODIFIED: Integrate CacheRouter
├── cache_loader.py                      # EXISTING: Config loading
├── cache_policy.py                      # EXISTING: Policy routing
└── config/
    └── cache.yaml                       # EXISTING: Configuration

tests/content_download/
├── test_cache_control.py                # NEW: Parser tests
├── test_cache_transport_wrapper.py       # NEW: Transport tests
├── test_httpx_transport_caching.py       # NEW: Integration tests
├── test_cache_loader.py                  # EXISTING: 38 tests ✓
└── test_cache_policy.py                  # EXISTING: 32 tests ✓
```

---

## Estimated Effort

| Component | LOC | Tests | Hours |
|-----------|-----|-------|-------|
| cache_control.py | 200 | 20 | 3 |
| cache_transport_wrapper.py | 250 | 15 | 4 |
| httpx_transport.py updates | 100 | 10 | 3 |
| Integration tests | 300 | 20 | 4 |
| Documentation | 200 | - | 2 |
| **Total** | **1,050** | **65** | **16** |

**Timeline**: 2-3 working days

---

## Next Steps

1. ✅ Phase 1: Foundation COMPLETE (70/70 tests passing)
2. 🔨 Phase 2: HTTP Transport Integration (IN PROGRESS)
   - 🔨 Item 2: Cache-Control Directives (START HERE)
   - ⏳ Item 1: Full Transport Integration
   - ⏳ Item 3: Conditional Requests
   - ⏳ Item 4: Expiration & Invalidation
   - ⏳ Item 5: Storage Optimization
3. ⏳ Phase 3: Integration Testing
4. ⏳ Phase 4: Telemetry & Monitoring
5. ⏳ Phase 5: Production Deployment

---

**Ready to begin Item 2 implementation!** 🚀
