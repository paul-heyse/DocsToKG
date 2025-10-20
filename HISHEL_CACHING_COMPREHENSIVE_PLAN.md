# Hishel Caching System - Comprehensive Implementation Plan

**Scope**: `ContentDownload-optimization-3-hishel.md`  
**Date**: October 21, 2025  
**Status**: Planning & Specification  
**Objective**: Design a robust, long-term HTTP caching solution using Hishel (RFC 9111)

---

## Executive Summary

This document provides a comprehensive, phased implementation plan for integrating Hishel RFC-9111-compliant HTTP caching into the DocsToKG ContentDownload module. The plan prioritizes:

- **RFC 9111 Compliance** - Proper HTTP caching semantics (ETag, Last-Modified, Vary, etc.)
- **Role-Based Caching** - Metadata/landing cached; artifacts (PDFs) never cached
- **Per-Host TTL Configuration** - Flexible, ops-friendly configuration without code changes
- **Offline Mode Support** - Fast deterministic behavior with `only-if-cached`
- **Comprehensive Instrumentation** - Detailed telemetry for optimization
- **Production Safety** - Conservative defaults (unknown hosts not cached) with explicit opt-in

---

## Part 1: Architecture Overview

### 1.1 Caching Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    Request Flow                             │
└─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────┐
        │  networking.request()   │
        │  (with role/host info)  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ resolve_cache_policy()  │
        │ (host + role → decision)│
        └────────────┬────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼──────┐        ┌──────▼─────┐
    │ Cached    │        │ Raw        │
    │ Client    │        │ Client     │
    │           │        │            │
    │ - Storage │        │ - No cache │
    │ - Hishel  │        │ - Direct   │
    │ - ETag    │        │   HTTP     │
    └────┬──────┘        └──────┬─────┘
         │                       │
    ┌────▼──────────────────────▼─────┐
    │ RateLimitedTransport             │
    │ (cached hits bypass limiter)     │
    └────┬──────────────────────┬──────┘
         │                      │
    ┌────▼──────────────────────▼─────┐
    │ BreakerRegistry + Tenacity       │
    │ (all requests still protected)   │
    └────┬──────────────────────┬──────┘
         │                      │
    ┌────▼──────────────────────▼─────┐
    │ HTTPTransport (real network)     │
    └────────────────────────────────┘
```

### 1.2 Key Design Principles

**Principle 1: Conservative by Default**
- Unknown hosts → **NOT cached** (explicit opt-in via YAML)
- Artifacts → **NEVER cached** (PDFs always raw)
- Unknown roles → treated as `metadata` (safest assumption)

**Principle 2: Respect Server Intent**
- Honor `Cache-Control`, `ETag`, `Last-Modified` headers
- Implement proper revalidation (304 Not Modified flows)
- Don't invent TTLs; use server validators when available

**Principle 3: Role-Based Isolation**
- `metadata` - Cached (API responses, structured data)
- `landing` - Cached (HTML pages, article pages)
- `artifact` - Never cached (PDFs, large documents)

**Principle 4: Observability**
- Track hit rate, revalidation rate, stale served %
- Monitor per-host, per-role metrics
- Estimate bandwidth savings

---

## Part 2: Component Architecture

### 2.1 Module Structure

```
ContentDownload/
├── cache_loader.py              # NEW - Config loading & validation
├── cache_policy.py              # NEW - Policy resolution & routing
├── httpx_transport.py           # MODIFIED - Dual client builders
├── networking.py                # MODIFIED - Cache-aware request routing
├── telemetry.py                 # MODIFIED - Cache instrumentation
├── args.py                       # MODIFIED - CLI flags for cache
└── cache.yaml                    # NEW - Configuration template
```

### 2.2 Data Flow

```
User Request
    ↓
[role, host] from request.extensions
    ↓
resolve_cache_policy(host, role)
    ↓ Returns: CacheDecision {use_cache, ttl_s, swrv_s, body_key}
    ↓
SELECT CLIENT:
  - use_cache=True  → CachedClient (Hishel wrapped)
  - use_cache=False → RawClient (direct HTTP)
    ↓
[Offline mode: add Cache-Control: only-if-cached]
    ↓
Send via RateLimitedTransport (cached hits bypass limiter)
    ↓
Response with Hishel extensions
    ↓
Extract cache telemetry
    ↓
Emit metrics (hit, revalidated, stale, stored)
```

---

## Part 3: Configuration System

### 3.1 Configuration Schema

```yaml
# cache.yaml
version: 1

# Storage backend configuration
storage:
  kind: "file"                          # file | memory | (future: redis, sqlite, s3)
  path: "${RUN_DIR}/cache/http"         # Expanded at runtime
  check_ttl_every_s: 600                # Garbage collect every 10 min

# Global cache controller defaults
controller:
  cacheable_methods: ["GET", "HEAD"]
  cacheable_statuses: [200, 301, 308]   # Per RFC 9111
  allow_heuristics: false               # Don't invent TTLs
  default: "DO_NOT_CACHE"               # Unknown hosts not cached

# Per-host policies
hosts:
  # Metadata API endpoints (high cache potential)
  api.crossref.org:
    ttl_s: 172800                       # 2 days default
    role:
      metadata:
        ttl_s: 259200                   # 3 days (override)
        swrv_s: 180                     # 3 min stale-while-revalidate
      landing:
        ttl_s: 86400                    # 1 day

  # OpenAlex API
  api.openalex.org:
    ttl_s: 172800
    role:
      metadata: {ttl_s: 259200, swrv_s: 180}
      landing:  {ttl_s: 86400}

  # Archive services
  web.archive.org:
    ttl_s: 172800
    role:
      metadata: {ttl_s: 172800, swrv_s: 120}
      landing:  {ttl_s: 86400}

  # Metadata-rich sources
  export.arxiv.org:         {ttl_s: 864000}  # 10 days
  europepmc.org:            {ttl_s: 172800}
  eutils.ncbi.nlm.nih.gov:  {ttl_s: 172800}
  api.unpaywall.org:        {ttl_s: 172800}
```

### 3.2 Environment & CLI Overlays

**Precedence**: YAML → env → CLI (later wins)

**Environment Variables**:
```bash
# Overall configuration
export DOCSTOKG_CACHE_YAML=/path/to/cache.yaml

# Per-host overrides
export DOCSTOKG_CACHE_HOST__api_crossref_org="ttl_s:259200"

# Per-host, per-role overrides
export DOCSTOKG_CACHE_ROLE__api_openalex_org__metadata="ttl_s:259200,swrv_s:180"

# Global defaults
export DOCSTOKG_CACHE_DEFAULTS="cacheable_methods:GET HEAD,cacheable_statuses:200,301,308"

# Offline mode
export DOCSTOKG_OFFLINE=1
```

**CLI Arguments**:
```bash
# Primary arguments
--cache-yaml /path/to/cache.yaml

# Per-host override
--cache-host "api.crossref.org=ttl_s:259200"

# Per-host, per-role override
--cache-role "api.openalex.org:metadata=ttl_s:259200,swrv_s:180"

# Global defaults
--cache-defaults "cacheable_methods:GET,cacheable_statuses:200,301,308"

# Offline mode
--offline
```

---

## Part 4: Core Components

### 4.1 Component 1: cache_loader.py (NEW)

**Responsibilities**:
- Load YAML configuration
- Apply env/CLI overlays
- Normalize host keys (lowercase + punycode)
- Validate configuration invariants
- Return `CacheConfig` dataclass

**Key Classes**:
```python
@dataclass
class CacheStorage:
    kind: str                  # "file" | "memory" | ...
    path: str                  # Resolved at runtime
    check_ttl_every_s: int

@dataclass
class CacheRolePolicy:
    ttl_s: Optional[int] = None
    swrv_s: Optional[int] = None         # stale-while-revalidate (metadata only)
    body_key: bool = False               # Include POST body in cache key

@dataclass
class CacheHostPolicy:
    ttl_s: Optional[int] = None
    role: Dict[str, CacheRolePolicy] = field(default_factory=dict)

@dataclass
class CacheControllerDefaults:
    cacheable_methods: List[str]
    cacheable_statuses: List[int]
    allow_heuristics: bool
    default: str                         # "DO_NOT_CACHE" | "CACHE"

@dataclass
class CacheConfig:
    storage: CacheStorage
    controller: CacheControllerDefaults
    hosts: Dict[str, CacheHostPolicy]    # Normalized host keys
```

**Public API**:
```python
def load_cache_config(
    yaml_path: Optional[str | Path],
    *,
    env: Mapping[str, str],
    cli_host_overrides: Sequence[str] | None = None,
    cli_role_overrides: Sequence[str] | None = None,
    cli_defaults_override: Optional[str] = None,
) -> CacheConfig:
    """Load cache configuration with YAML → env → CLI precedence."""
```

**Validation**:
- `ttl_s >= 0`, `swrv_s >= 0`
- If `controller.default="DO_NOT_CACHE"`, ensure `hosts` is non-empty
- Host keys normalized to lowercase + punycode
- Role names validated (metadata, landing, artifact)

### 4.2 Component 2: cache_policy.py (NEW)

**Responsibilities**:
- Provide policy resolution logic
- Route requests to cached vs raw client
- Apply per-request Hishel extensions

**Key Types**:
```python
@dataclass
class CacheDecision:
    use_cache: bool                       # Should use cached client?
    ttl_s: Optional[int] = None
    swrv_s: Optional[int] = None          # stale-while-revalidate
    body_key: bool = False

class CacheRouter:
    """Stateful policy resolver with startup logging."""
    
    def __init__(self, config: CacheConfig):
        """Initialize from CacheConfig; print effective policy at startup."""
    
    def resolve_policy(
        self,
        host: str,
        role: str = "metadata",
    ) -> CacheDecision:
        """Resolve whether to cache; return decision with TTL/SWrV."""
    
    def print_effective_policy(self) -> str:
        """Return human-readable table of effective policies for ops."""
```

**Logic**:
```
if host not in config.hosts:
    return CacheDecision(use_cache=False)  # Unknown host → raw

if role == "artifact":
    return CacheDecision(use_cache=False)  # Artifacts never cached

# Known host + cacheable role
host_policy = config.hosts[host]
role_policy = host_policy.role.get(role)

if role_policy and role_policy.ttl_s is not None:
    ttl = role_policy.ttl_s
    swrv = role_policy.swrv_s if role == "metadata" else None
    return CacheDecision(use_cache=True, ttl_s=ttl, swrv_s=swrv)

# Fall back to host-level TTL
if host_policy.ttl_s is not None:
    return CacheDecision(use_cache=True, ttl_s=host_policy.ttl_s)

# Fall back to controller default
return CacheDecision(
    use_cache=(config.controller.default != "DO_NOT_CACHE")
)
```

### 4.3 Component 3: httpx_transport.py (MODIFIED)

**New Functionality**:
- Build **two separate clients**:
  1. **Cached Client** - Hishel-wrapped for metadata/landing
  2. **Raw Client** - Direct HTTP for artifacts/unknown hosts

**Key Changes**:
```python
def _build_cached_client(
    cache_config: CacheConfig,
    ssl_context: ssl.SSLContext,
) -> httpx.Client:
    """Build client with Hishel CacheTransport + RateLimitedTransport."""
    storage = FileStorage(
        base_path=cache_config.storage.path,
        check_ttl_every=cache_config.storage.check_ttl_every_s,
    )
    controller = CacheController(
        cacheable_methods=cache_config.controller.cacheable_methods,
        cacheable_statuses=cache_config.controller.cacheable_statuses,
        allow_heuristics=cache_config.controller.allow_heuristics,
    )
    base_transport = httpx.HTTPTransport(
        ssl_context=ssl_context,
        retries=2,
    )
    rate_limited = RateLimitedTransport(base_transport, ...)
    cached = CacheTransport(
        transport=rate_limited,
        storage=storage,
        controller=controller,
    )
    return httpx.Client(
        transport=cached,
        limits=_DEFAULT_LIMITS,
        timeout=_DEFAULT_TIMEOUT,
        ...
    )

def _build_raw_client(
    ssl_context: ssl.SSLContext,
) -> httpx.Client:
    """Build client without cache, direct to rate limiter + HTTP."""
    base_transport = httpx.HTTPTransport(
        ssl_context=ssl_context,
        retries=2,
    )
    rate_limited = RateLimitedTransport(base_transport, ...)
    return httpx.Client(
        transport=rate_limited,
        limits=_DEFAULT_LIMITS,
        timeout=_DEFAULT_TIMEOUT,
        ...
    )

# Module-level clients
_CACHED_CLIENT: Optional[httpx.Client] = None
_RAW_CLIENT: Optional[httpx.Client] = None

def get_cached_http_client() -> httpx.Client:
    """Get or create cached client."""

def get_raw_http_client() -> httpx.Client:
    """Get or create raw (non-cached) client."""
```

### 4.4 Component 4: networking.py (MODIFIED)

**New Functions**:
```python
def _apply_cache_extensions(
    request: httpx.Request,
    decision: CacheDecision,
    offline: bool = False,
) -> None:
    """Apply Hishel extensions to request based on cache decision."""
    if not decision.use_cache:
        return
    
    extensions = request.extensions
    if decision.ttl_s is not None:
        extensions["hishel_ttl"] = decision.ttl_s
    
    if decision.swrv_s is not None and is_metadata_role(request):
        extensions["hishel_stale_while_revalidate"] = decision.swrv_s
    
    if decision.body_key:
        extensions["hishel_body_key"] = True
    
    if offline:
        # only-if-cached → 504 on miss
        request.headers["Cache-Control"] = "only-if-cached"
        extensions["docs_offline"] = True

def _extract_cache_telemetry(
    response: httpx.Response,
) -> Dict[str, Any]:
    """Extract Hishel cache telemetry from response extensions."""
    return {
        "from_cache": response.extensions.get("hishel_from_cache", False),
        "revalidated": response.extensions.get("hishel_revalidated", False),
        "stored": response.extensions.get("hishel_stored", False),
        "stale": response.extensions.get("hishel_stale", False),
        "age_s": response.extensions.get("hishel_age_s"),
    }
```

**Modified request_with_retries()**:
```python
def request_with_retries(
    method: str,
    url: str,
    *,
    role: str = "metadata",
    offline: bool = False,
    cache_router: Optional[CacheRouter] = None,
    **kwargs,
) -> httpx.Response:
    """
    Execute HTTP request with retry, rate limiting, and optional caching.
    
    New parameters:
    - role: "metadata" | "landing" | "artifact" (defaults to "metadata")
    - offline: Enable only-if-cached mode
    - cache_router: CacheRouter instance for policy resolution
    """
    
    # 1. Resolve cache policy
    decision = cache_router.resolve_policy(host, role) if cache_router else CacheDecision(False)
    
    # 2. Choose client
    client = (
        get_cached_http_client() if decision.use_cache else get_raw_http_client()
    )
    
    # 3. Build request with cache extensions
    request = client.build_request(method, url, **kwargs)
    _apply_cache_extensions(request, decision, offline=offline)
    
    # 4. Send (still wrapped by breaker/limiter/Tenacity)
    response = send_with_tenacity_and_breaker(request, ...)
    
    # 5. Extract cache telemetry
    cache_telemetry = _extract_cache_telemetry(response)
    meta = request.extensions.setdefault("docs_network_meta", {})
    meta["cache"] = cache_telemetry
    
    return response
```

### 4.5 Component 5: telemetry.py (MODIFIED)

**New Counters**:
```python
# Per-host, per-role counters
cache_hit_total = Counter(
    "cache_hit_total",
    "Number of cache hits",
    ["host", "role"],
)

cache_revalidated_total = Counter(
    "cache_revalidated_total",
    "Number of cache revalidations (304)",
    ["host", "role"],
)

cache_stale_total = Counter(
    "cache_stale_total",
    "Number of stale responses served",
    ["host", "role"],
)

cache_stored_total = Counter(
    "cache_stored_total",
    "Number of responses stored in cache",
    ["host", "role"],
)

cache_offline_504_total = Counter(
    "cache_offline_504_total",
    "Number of 504 responses from only-if-cached in offline mode",
    ["host", "role"],
)

# Bandwidth saved estimate
cache_bandwidth_saved_bytes = Counter(
    "cache_bandwidth_saved_bytes",
    "Estimated bytes saved by cache hits and revalidations",
    ["host"],
)
```

**Run Summary Section**:
```python
def build_cache_summary(run_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build cache performance summary from telemetry."""
    return {
        "hit_rate": hits / (hits + network_fetches),
        "revalidation_rate": revalidated / network_fetches,
        "stale_served_percent": (stale / hits) * 100,
        "bandwidth_saved_mb": bandwidth_saved / 1_000_000,
        "top_hosts_by_hits": [
            {"host": h, "hits": c} for h, c in sorted_by_hit_count[:5]
        ],
    }
```

### 4.6 Component 6: args.py (MODIFIED)

**New CLI Flags**:
```python
parser.add_argument(
    "--cache-yaml",
    type=Path,
    help="Path to cache configuration YAML",
    default=None,
)

parser.add_argument(
    "--cache-host",
    action="append",
    dest="cache_host_overrides",
    help="Per-host cache override: api.example.com=ttl_s:259200",
    default=[],
)

parser.add_argument(
    "--cache-role",
    action="append",
    dest="cache_role_overrides",
    help="Per-host, per-role override: api.example.com:metadata=ttl_s:259200,swrv_s:180",
    default=[],
)

parser.add_argument(
    "--offline",
    action="store_true",
    help="Run in offline mode (only-if-cached for all cached requests)",
)

parser.add_argument(
    "--cache-vacuum",
    action="store_true",
    help="Vacuum expired cache entries and exit",
)
```

### 4.7 Component 7: cache.yaml (NEW TEMPLATE)

See Section 3.1 for complete configuration template.

---

## Part 5: Integration Checklist

### 5.1 Startup Sequence

```
1. Load arguments (args.py)
2. Load cache config (cache_loader.py)
3. Create CacheRouter (cache_policy.py)
4. Print effective policy table
5. Build cached + raw HTTP clients (httpx_transport.py)
6. Initialize telemetry counters (telemetry.py)
7. Begin request loop
```

### 5.2 Request Handling Flow

```
1. network.request_with_retries(method, url, role=..., offline=...)
2. CacheRouter.resolve_policy(host, role)
3. Choose client (cached vs raw)
4. Apply cache extensions
5. Send request (still protected by breaker/limiter/Tenacity)
6. Extract cache telemetry
7. Update counters
8. Return response
```

### 5.3 Offline Mode

```
1. --offline flag enables
2. CacheRouter.resolve_policy() returns same decision
3. network._apply_cache_extensions() adds "Cache-Control: only-if-cached"
4. Hishel returns 504 on cache miss
5. Telemetry records cache_offline_504_total
6. Pipeline can log blocked_offline reason
```

---

## Part 6: Testing Strategy

### 6.1 Unit Tests (cache_loader_test.py)

```python
def test_load_yaml_basic(): ...
def test_env_override_precedence(): ...
def test_cli_override_precedence(): ...
def test_host_key_normalization(): ...
def test_validation_fail_negative_ttl(): ...
def test_validation_fail_empty_hosts_when_default_no_cache(): ...
```

### 6.2 Unit Tests (cache_policy_test.py)

```python
def test_unknown_host_not_cached(): ...
def test_known_host_metadata_cached(): ...
def test_known_host_artifact_never_cached(): ...
def test_role_specific_ttl_override(): ...
def test_fallback_to_host_ttl(): ...
def test_effective_policy_table(): ...
```

### 6.3 Integration Tests (test_cache_integration.py)

```python
def test_first_request_stores_in_cache(): ...
def test_second_request_hits_cache(): ...
def test_server_etag_revalidation_flow(): ...
def test_stale_while_revalidate_metadata_only(): ...
def test_offline_mode_504_on_miss(): ...
def test_offline_mode_hit_from_cache(): ...
def test_role_isolation_metadata_cached_artifact_raw(): ...
def test_post_body_keying(): ...
def test_vary_header_handling(): ...
def test_cache_telemetry_extraction(): ...
def test_cached_hits_bypass_rate_limiter(): ...
```

### 6.4 End-to-End Tests (test_cache_e2e.py)

```python
def test_full_request_flow_with_cache(): ...
def test_bandwidth_saved_calculation(): ...
def test_cache_summary_generation(): ...
```

---

## Part 7: Deployment & Operations

### 7.1 Pre-Deployment

1. Generate `cache.yaml` from known hosts (Crossref, OpenAlex, Wayback, etc.)
2. Review TTL/SWrV values with ops
3. Run integration tests with mock responses
4. Validate cache directory permissions

### 7.2 Canary Deployment

1. Enable caching for small host list first (Crossref, OpenAlex)
2. Monitor hit rate, revalidation rate, stale %
3. Verify bandwidth savings estimate
4. Watch for unexpected 304/504 responses

### 7.3 Monitoring

**Dashboards**:
- Cache hit rate by host
- Revalidation rate by host
- Stale served % by host
- Bandwidth saved per run
- Offline 504 count

**Alerts**:
- Hit rate < 10% on high-traffic hosts
- Revalidation rate > 50% (hosts sending too many 304s)
- Stale served > 5% (TTL misconfiguration)

### 7.4 Operations Commands

```bash
# Vacuum expired cache entries
docstokg download ... --cache-vacuum

# Run in offline mode (testing)
docstokg download --offline --cache-yaml cache.yaml ...

# Override specific host TTL
docstokg download --cache-host "api.crossref.org=ttl_s:86400" ...

# Override host+role combination
docstokg download --cache-role "api.openalex.org:metadata=ttl_s:518400" ...
```

---

## Part 8: Success Metrics

### 8.1 Performance Metrics

| Metric | Target | Baseline | Expected Improvement |
|--------|--------|----------|----------------------|
| Cache Hit Rate | > 50% | 0% | +50-70% |
| Revalidation Rate | 20-40% | 0% | +20-40% |
| Bandwidth Saved | > 30% | 0% | +40-60% |
| Avg Response Time | < 50ms | ~200-500ms | -75% |

### 8.2 Operational Metrics

- Configuration load time: < 100ms
- Cache policy lookup: O(1)
- Cache file vacuum: < 5s for typical runs
- Telemetry overhead: < 1% CPU

### 8.3 Quality Metrics

- 100% test coverage for cache components
- Zero cache coherence issues (served stale % < 1%)
- Zero data corruption (cache integrity verified)
- RFC 9111 compliance verified

---

## Part 9: Phased Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Implement `cache_loader.py` with YAML/env/CLI support
- [ ] Implement `cache_policy.py` with routing logic
- [ ] Create `cache.yaml` template with known hosts
- [ ] Add CLI flags to `args.py`

### Phase 2: Integration (Week 2)
- [ ] Modify `httpx_transport.py` for dual clients
- [ ] Modify `networking.py` for cache-aware routing
- [ ] Implement cache extension application
- [ ] Add telemetry extraction

### Phase 3: Telemetry & Testing (Week 3)
- [ ] Add cache counters to `telemetry.py`
- [ ] Implement cache summary generation
- [ ] Write comprehensive test suite
- [ ] Integration tests with mock server

### Phase 4: Operations & Optimization (Week 4)
- [ ] Implement offline mode
- [ ] Create operational runbooks
- [ ] Canary deployment with monitoring
- [ ] Performance tuning based on telemetry

### Phase 5: Stretch Goals (Beyond)
- [ ] Stale-if-error (SIE) support
- [ ] Redis storage backend
- [ ] Dynamic host discovery from telemetry
- [ ] Cache directory cap management

---

## Part 10: Risk Mitigation

### 10.1 Common Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Stale data served | High | Medium | TTL validation, SWrV conservative defaults |
| Cache coherence issues | Medium | High | RFC 9111 compliance, validator revalidation |
| Performance regression | Low | High | Benchmark, O(1) lookups, early offline detection |
| Configuration errors | High | Low | Schema validation, startup policy table, dry-run mode |

### 10.2 Rollback Plan

1. Set `--offline` for fast, safe fallback (only-if-cached)
2. Delete `cache.yaml` or set `default: "DO_NOT_CACHE"`
3. Redeploy without cache, all requests routed to raw client
4. Verify metrics return to baseline

---

## Part 11: Dependencies & Requirements

### 11.1 Required Libraries

- `hishel == 0.1.5+` (RFC 9111 HTTP caching)
- `httpx == 0.28+` (already installed)
- `pyyaml == 6.0.3` (already installed)
- `idna == 3.11` (already installed)

### 11.2 Storage Requirements

- **Filesystem**: ~500MB to 2GB typical (metadata/landing only)
- **Memory**: ~50MB to 200MB (in-memory storage option)
- **Cleanup**: Hishel handles TTL expiry automatically

---

## Part 12: Configuration Examples

### 12.1 Conservative (Safe)
```yaml
controller:
  default: "DO_NOT_CACHE"          # Unknown hosts never cached
hosts:
  api.crossref.org: {ttl_s: 172800}  # 2 days
```

### 12.2 Balanced (Recommended)
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

### 12.3 Aggressive (High Cache Potential)
```yaml
controller:
  default: "DO_NOT_CACHE"
hosts:
  api.crossref.org: {ttl_s: 259200, role: {metadata: {ttl_s: 518400, swrv_s: 300}}}
  api.openalex.org: {ttl_s: 259200, role: {metadata: {ttl_s: 518400, swrv_s: 300}}}
  web.archive.org: {ttl_s: 172800}
  # ... 20+ hosts with aggressive TTLs
```

---

## Conclusion

This comprehensive plan provides:

✅ **RFC 9111 Compliance** - Proper HTTP caching semantics  
✅ **Role-Based Architecture** - Metadata/landing cached; artifacts raw  
✅ **Configuration Flexibility** - YAML + env + CLI without code changes  
✅ **Offline Support** - Fast deterministic behavior  
✅ **Comprehensive Instrumentation** - Hit rate, revalidation, bandwidth savings  
✅ **Production Safety** - Conservative defaults with explicit opt-in  
✅ **Clear Deployment Path** - Phased, testable rollout  
✅ **Operational Clarity** - Monitoring, dashboards, runbooks  

The system is designed for long-term robustness and ops-friendly operations while maintaining RFC 9111 compliance and data integrity.

