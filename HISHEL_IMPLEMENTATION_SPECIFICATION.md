# Hishel Caching Implementation - Detailed Specification

**Document**: Implementation Specification & Architecture Decision Record  
**Date**: October 21, 2025  
**Scope**: `ContentDownload-optimization-3-hishel.md`  
**Status**: Ready for Phase-by-Phase Implementation  

---

## Document Purpose

This specification details the **exact interfaces, data structures, algorithms, and integration points** needed to implement the Hishel caching system described in the Comprehensive Plan. It serves as both:

1. **Architecture Decision Record (ADR)** - Rationale for design choices
2. **Implementation Specification** - Exact code signatures and behaviors
3. **Integration Checklist** - Verification points for each component

---

## Part 1: Module Interface Specifications

### 1.1 cache_loader.py - Configuration Loading

**File**: `src/DocsToKG/ContentDownload/cache_loader.py`

#### Data Structures

```python
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence
from pathlib import Path
from enum import Enum

class StorageKind(str, Enum):
    """Storage backend types."""
    FILE = "file"
    MEMORY = "memory"
    REDIS = "redis"          # Future
    SQLITE = "sqlite"        # Future
    S3 = "s3"                # Future

class CacheDefault(str, Enum):
    """Global caching policy for unknown hosts."""
    DO_NOT_CACHE = "DO_NOT_CACHE"
    CACHE = "CACHE"

@dataclass(frozen=True)
class CacheStorage:
    """Storage backend configuration."""
    kind: StorageKind
    path: str                                    # Resolved at runtime
    check_ttl_every_s: int = 600
    
    def __post_init__(self):
        if self.check_ttl_every_s < 60:
            raise ValueError("check_ttl_every_s must be >= 60")

@dataclass(frozen=True)
class CacheRolePolicy:
    """Per-role cache policy (e.g., metadata vs landing)."""
    ttl_s: Optional[int] = None
    swrv_s: Optional[int] = None                # stale-while-revalidate
    body_key: bool = False
    
    def __post_init__(self):
        if self.ttl_s is not None and self.ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        if self.swrv_s is not None and self.swrv_s < 0:
            raise ValueError("swrv_s must be >= 0")

@dataclass(frozen=True)
class CacheHostPolicy:
    """Per-host cache policy with role-specific overrides."""
    ttl_s: Optional[int] = None
    role: Dict[str, CacheRolePolicy] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.ttl_s is not None and self.ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        for role_name in self.role.keys():
            if role_name not in ("metadata", "landing", "artifact"):
                raise ValueError(f"Invalid role: {role_name}")

@dataclass(frozen=True)
class CacheControllerDefaults:
    """Global cache controller defaults."""
    cacheable_methods: List[str] = field(default_factory=lambda: ["GET", "HEAD"])
    cacheable_statuses: List[int] = field(default_factory=lambda: [200, 301, 308])
    allow_heuristics: bool = False
    default: CacheDefault = CacheDefault.DO_NOT_CACHE

@dataclass(frozen=True)
class CacheConfig:
    """Complete cache configuration."""
    storage: CacheStorage
    controller: CacheControllerDefaults
    hosts: Dict[str, CacheHostPolicy]          # Keys are normalized
    
    def __post_init__(self):
        if self.controller.default == CacheDefault.DO_NOT_CACHE and not self.hosts:
            raise ValueError(
                "If default=DO_NOT_CACHE, must specify at least one host"
            )
```

#### Public API

```python
def load_cache_config(
    yaml_path: Optional[str | Path] = None,
    *,
    env: Mapping[str, str],
    cli_host_overrides: Sequence[str] | None = None,
    cli_role_overrides: Sequence[str] | None = None,
    cli_defaults_override: Optional[str] = None,
) -> CacheConfig:
    """
    Load cache configuration with YAML → env → CLI precedence.
    
    Args:
        yaml_path: Path to cache.yaml configuration file
        env: Environment variables (typically os.environ)
        cli_host_overrides: List of "host=ttl_s:259200" strings
        cli_role_overrides: List of "host:role=ttl_s:259200,swrv_s:180" strings
        cli_defaults_override: "cacheable_methods:GET,cacheable_statuses:200,301,308"
    
    Returns:
        CacheConfig with all values resolved and validated
    
    Raises:
        FileNotFoundError: If yaml_path provided but not found
        ValueError: If any config value fails validation
        yaml.YAMLError: If YAML parsing fails
    """

def _normalize_host_key(host: str) -> str:
    """
    Normalize host to lowercase + punycode (IDNA 2008 + UTS #46).
    
    Examples:
        "API.Crossref.Org" → "api.crossref.org"
        "münchen.example" → "xn--mnich-kva.example"
    
    Returns:
        Normalized hostname for use as dictionary key
    """

def _validate_cache_config(cfg: CacheConfig) -> None:
    """
    Validate all invariants in CacheConfig.
    
    Checks:
        - ttl_s >= 0 for all policies
        - swrv_s >= 0 for all policies
        - If default=DO_NOT_CACHE, hosts is non-empty
        - Role names are valid (metadata, landing, artifact)
    
    Raises:
        ValueError: If any invariant violated
    """
```

#### Integration Points

- **Environment Variables** (read from `env` dict):
  - `DOCSTOKG_CACHE_YAML=/path/to/cache.yaml`
  - `DOCSTOKG_CACHE_HOST__api_crossref_org=ttl_s:259200`
  - `DOCSTOKG_CACHE_ROLE__api_openalex_org__metadata=ttl_s:259200,swrv_s:180`
  - `DOCSTOKG_CACHE_DEFAULTS=cacheable_methods:GET,cacheable_statuses:200,301,308`

- **CLI Arguments** (from `args.py`):
  - `--cache-yaml`
  - `--cache-host`
  - `--cache-role`
  - `--cache-defaults`

---

### 1.2 cache_policy.py - Policy Resolution

**File**: `src/DocsToKG/ContentDownload/cache_policy.py`

#### Data Structures

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CacheDecision:
    """Decision whether to cache a specific request."""
    use_cache: bool
    ttl_s: Optional[int] = None
    swrv_s: Optional[int] = None                # Only for metadata role
    body_key: bool = False
```

#### Public API

```python
class CacheRouter:
    """
    Stateful policy resolver that routes requests to cached vs raw clients.
    
    Responsibilities:
        - Hold CacheConfig loaded from cache_loader
        - Resolve (host, role) → CacheDecision
        - Print effective policy table at startup for ops
        - Handle host key normalization
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize router with cache configuration.
        
        Args:
            config: CacheConfig from cache_loader.load_cache_config()
        
        Side Effects:
            Logs effective policy table at INFO level
        """
    
    def resolve_policy(
        self,
        host: str,
        role: str = "metadata",
    ) -> CacheDecision:
        """
        Resolve caching policy for a request.
        
        Logic:
            1. Normalize host key (lowercase + punycode)
            2. If host not in config.hosts → CacheDecision(use_cache=False)
            3. If role == "artifact" → CacheDecision(use_cache=False)
            4. If role in host_policy.role → use role-specific TTL/SWrV
            5. Else if host_policy.ttl_s is set → use host TTL
            6. Else → use controller.default
        
        Args:
            host: Hostname (will be normalized)
            role: "metadata" | "landing" | "artifact" (default: "metadata")
        
        Returns:
            CacheDecision with use_cache flag and optional TTL/SWrV
        
        Examples:
            >>> router.resolve_policy("api.crossref.org", "metadata")
            CacheDecision(use_cache=True, ttl_s=259200, swrv_s=180)
            
            >>> router.resolve_policy("example.com", "metadata")
            CacheDecision(use_cache=False)
            
            >>> router.resolve_policy("api.crossref.org", "artifact")
            CacheDecision(use_cache=False)
        """
    
    def print_effective_policy(self) -> str:
        """
        Generate human-readable policy table for operations.
        
        Returns:
            Multi-line string with table of hosts and their TTL/SWrV values
        
        Example output:
            Host                    Role        TTL (days)  SWrV (min)
            ──────────────────────  ──────────  ──────────  ──────────
            api.crossref.org        metadata    3           3
            api.crossref.org        landing     1           -
            api.openalex.org        metadata    3           3
            ...
        """
```

#### Algorithm Detail: resolve_policy()

```python
def resolve_policy(self, host: str, role: str = "metadata") -> CacheDecision:
    # 1. Normalize host key
    normalized_host = _normalize_host_key(host)
    
    # 2. Unknown host → not cached
    if normalized_host not in self.config.hosts:
        return CacheDecision(use_cache=False)
    
    # 3. Artifacts never cached
    if role == "artifact":
        return CacheDecision(use_cache=False)
    
    # 4. Get host policy
    host_policy = self.config.hosts[normalized_host]
    
    # 5. Try role-specific policy
    if role in host_policy.role:
        role_policy = host_policy.role[role]
        if role_policy.ttl_s is not None:
            swrv = (
                role_policy.swrv_s 
                if role == "metadata" and role_policy.swrv_s is not None 
                else None
            )
            return CacheDecision(
                use_cache=True,
                ttl_s=role_policy.ttl_s,
                swrv_s=swrv,
                body_key=role_policy.body_key,
            )
    
    # 6. Fall back to host-level TTL
    if host_policy.ttl_s is not None:
        return CacheDecision(use_cache=True, ttl_s=host_policy.ttl_s)
    
    # 7. Fall back to controller default
    use_cache = (self.config.controller.default == CacheDefault.CACHE)
    return CacheDecision(use_cache=use_cache)
```

---

### 1.3 httpx_transport.py - Client Factory (MODIFIED)

**File**: `src/DocsToKG/ContentDownload/httpx_transport.py`

#### New Module-Level State

```python
_CACHED_CLIENT: Optional[httpx.Client] = None
_RAW_CLIENT: Optional[httpx.Client] = None
_CACHE_CONFIG: Optional[CacheConfig] = None
_CACHE_CONFIG_LOCK = threading.RLock()
```

#### New Functions

```python
def configure_cache_config(config: CacheConfig) -> None:
    """
    Set the cache configuration for dual client builders.
    
    Must be called during startup before get_cached_http_client().
    Thread-safe via internal locking.
    
    Args:
        config: CacheConfig from cache_loader.load_cache_config()
    """

def _build_cached_client(
    cache_config: CacheConfig,
    ssl_context: ssl.SSLContext,
    rate_limiter: RateLimitedTransport,
) -> httpx.Client:
    """
    Build HTTPX client with Hishel caching layer.
    
    Transport stack (outermost first):
        1. Hishel CacheTransport
        2. RateLimitedTransport (cached hits bypass this)
        3. HTTPTransport (real network)
    
    Args:
        cache_config: CacheConfig with storage/controller settings
        ssl_context: SSL context from _build_ssl_context()
        rate_limiter: RateLimitedTransport wrapping HTTPTransport
    
    Returns:
        httpx.Client configured with Hishel caching
    
    Side Effects:
        Creates FileStorage at cache_config.storage.path
        Logs Hishel controller configuration
    """

def _build_raw_client(
    ssl_context: ssl.SSLContext,
    rate_limiter: RateLimitedTransport,
) -> httpx.Client:
    """
    Build HTTPX client WITHOUT caching (direct to rate limiter).
    
    Transport stack (outermost first):
        1. RateLimitedTransport
        2. HTTPTransport (real network)
    
    Used for:
        - Artifact downloads (PDFs)
        - Unknown hosts
        - Offline-failed requests (504s)
    
    Args:
        ssl_context: SSL context from _build_ssl_context()
        rate_limiter: RateLimitedTransport wrapping HTTPTransport
    
    Returns:
        httpx.Client without caching layer
    """

def get_cached_http_client() -> httpx.Client:
    """
    Get or create the cached HTTP client.
    
    Must call configure_cache_config() at startup first.
    
    Returns:
        Process-wide cached httpx.Client (thread-safe)
    
    Raises:
        RuntimeError: If cache config not yet configured
    """

def get_raw_http_client() -> httpx.Client:
    """
    Get or create the raw (non-cached) HTTP client.
    
    Returns:
        Process-wide raw httpx.Client (thread-safe)
    """
```

#### Key Changes to Existing Functions

```python
def get_http_client(use_cache: bool = True) -> httpx.Client:
    """
    [DEPRECATED] Use get_cached_http_client() or get_raw_http_client() instead.
    
    For backward compatibility, routes to appropriate client based on use_cache.
    """

def reset_http_client_for_tests() -> None:
    """
    Reset both cached and raw clients (for testing).
    
    Also needs to reset _CACHE_CONFIG to None.
    """
```

---

### 1.4 networking.py - Cache-Aware Routing (MODIFIED)

**File**: `src/DocsToKG/ContentDownload/networking.py`

#### New Functions

```python
def _apply_cache_extensions(
    request: httpx.Request,
    decision: CacheDecision,
    offline: bool = False,
) -> None:
    """
    Apply Hishel extensions to request based on cache decision.
    
    Sets extensions that Hishel CacheTransport will use:
        - hishel_ttl: Override TTL for this request
        - hishel_stale_while_revalidate: SWrV for metadata
        - hishel_body_key: Include POST body in cache key
    
    For offline mode:
        - Cache-Control: only-if-cached header
        - docs_offline: True extension
    
    Args:
        request: httpx.Request object to annotate
        decision: CacheDecision from cache_policy.resolve_policy()
        offline: Enable only-if-cached mode
    
    Side Effects:
        Modifies request.headers and request.extensions
    """

def _extract_cache_telemetry(response: httpx.Response) -> Dict[str, Any]:
    """
    Extract Hishel cache telemetry from response.
    
    Reads Hishel extensions from response.extensions:
        - hishel_from_cache: bool
        - hishel_revalidated: bool
        - hishel_stored: bool
        - hishel_stale: bool
        - hishel_age_s: Optional[int]
    
    Args:
        response: httpx.Response with Hishel extensions
    
    Returns:
        Dict with cache telemetry:
            {
                "from_cache": bool,
                "revalidated": bool,
                "stored": bool,
                "stale": bool,
                "age_s": Optional[int],
            }
    """

def _get_role_from_request(request: httpx.Request) -> str:
    """
    Extract role from request.extensions or default to "metadata".
    
    Args:
        request: httpx.Request with possible docs_role extension
    
    Returns:
        "metadata" | "landing" | "artifact"
    """
```

#### Modified request_with_retries() Signature

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
    
    NEW PARAMETERS:
    - role: "metadata" | "landing" | "artifact" (default: "metadata")
    - offline: Enable only-if-cached mode (returns 504 on cache miss)
    - cache_router: CacheRouter instance for policy resolution
    
    EXISTING PARAMETERS:
    - method, url: Standard HTTP method and URL
    - **kwargs: Passed to httpx.Client.request()
    
    FLOW:
        1. Validate inputs
        2. Resolve cache policy (if cache_router provided)
        3. Choose client (cached vs raw based on decision)
        4. Build request
        5. Apply cache extensions
        6. Send request (with breaker/limiter/Tenacity retries)
        7. Extract cache telemetry
        8. Record telemetry in response.extensions["docs_network_meta"]["cache"]
        9. Return response
    
    CACHE BYPASS:
        - Cached hits skip rate limiter (efficiency win)
        - All requests still pass through breaker/Tenacity (safety)
        - Offline mode returns 504 on cache miss (deterministic behavior)
    
    Returns:
        httpx.Response with cache telemetry in extensions
    """
```

#### Implementation Pseudocode

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
    
    # 1. Parse URL to extract host
    parsed = urlsplit(url)
    host = parsed.hostname or "unknown"
    
    # 2. Resolve cache policy
    if cache_router:
        decision = cache_router.resolve_policy(host, role)
    else:
        decision = CacheDecision(use_cache=False)
    
    # 3. Choose client
    client = (
        httpx_transport.get_cached_http_client() 
        if decision.use_cache 
        else httpx_transport.get_raw_http_client()
    )
    
    # 4. Build request
    request = client.build_request(method, url, **kwargs)
    request.extensions["docs_role"] = role
    
    # 5. Apply cache extensions
    _apply_cache_extensions(request, decision, offline=offline)
    
    # 6. Send with retries/breaker/limiter
    response = client.send(
        request,
        follow_redirects=kwargs.get("follow_redirects", False),
    )
    
    # 7. Extract cache telemetry
    cache_telemetry = _extract_cache_telemetry(response)
    
    # 8. Record in network metadata
    meta = response.extensions.setdefault("docs_network_meta", {})
    meta["cache"] = cache_telemetry
    
    # 9. Return response
    return response
```

---

### 1.5 telemetry.py - Instrumentation (MODIFIED)

**File**: `src/DocsToKG/ContentDownload/telemetry.py`

#### New Counters/Gauges

```python
from prometheus_client import Counter

# Per-host, per-role cache metrics
cache_hit_total = Counter(
    "cache_hit_total",
    "Number of cache hits by host and role",
    ["host", "role"],
)

cache_revalidated_total = Counter(
    "cache_revalidated_total",
    "Number of cache revalidations (304 responses) by host and role",
    ["host", "role"],
)

cache_stale_total = Counter(
    "cache_stale_total",
    "Number of stale responses served from cache by host and role",
    ["host", "role"],
)

cache_stored_total = Counter(
    "cache_stored_total",
    "Number of responses stored in cache by host and role",
    ["host", "role"],
)

cache_offline_504_total = Counter(
    "cache_offline_504_total",
    "Number of 504 responses from only-if-cached in offline mode by host and role",
    ["host", "role"],
)

# Bandwidth savings
cache_bandwidth_saved_bytes = Counter(
    "cache_bandwidth_saved_bytes",
    "Estimated bytes saved by cache hits and revalidations by host",
    ["host"],
)
```

#### New Functions

```python
def record_cache_hit(host: str, role: str) -> None:
    """Record a cache hit for metrics."""
    cache_hit_total.labels(host=host, role=role).inc()

def record_cache_revalidated(host: str, role: str) -> None:
    """Record a cache revalidation (304 response)."""
    cache_revalidated_total.labels(host=host, role=role).inc()

def record_cache_stale(host: str, role: str) -> None:
    """Record a stale response served from cache."""
    cache_stale_total.labels(host=host, role=role).inc()

def record_cache_stored(host: str, role: str) -> None:
    """Record a response stored in cache."""
    cache_stored_total.labels(host=host, role=role).inc()

def record_cache_offline_504(host: str, role: str) -> None:
    """Record a 504 from only-if-cached in offline mode."""
    cache_offline_504_total.labels(host=host, role=role).inc()

def record_cache_bandwidth_saved(host: str, bytes_saved: int) -> None:
    """Record estimated bandwidth saved by cache."""
    cache_bandwidth_saved_bytes.labels(host=host).inc(bytes_saved)

def build_cache_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build cache performance summary from run statistics.
    
    Args:
        stats: Statistics dict with cache metrics
    
    Returns:
        {
            "cache_hit_rate": float,              # 0.0-1.0
            "cache_revalidation_rate": float,     # 0.0-1.0
            "cache_stale_served_percent": float,  # 0.0-100.0
            "cache_bandwidth_saved_mb": float,
            "cache_top_hosts_by_hits": [
                {"host": str, "hits": int, "revalidations": int},
                ...
            ],
        }
    """
```

---

## Part 2: Integration & Flow Diagrams

### 2.1 Startup Sequence

```
CLI Invocation
    ↓
args.py: Parse --cache-yaml, --cache-host, --cache-role, --offline, etc.
    ↓
cache_loader.load_cache_config(yaml_path, env, cli_overrides)
    ↓ Returns: CacheConfig
cache_policy.CacheRouter(cache_config)
    ↓ Prints: Effective policy table
httpx_transport.configure_cache_config(cache_config)
httpx_transport.get_cached_http_client()  (initializes storage + Hishel)
httpx_transport.get_raw_http_client()     (initializes raw transport)
    ↓
pipeline.ResolverPipeline(cache_router=...)
    ↓
Download loop begins
```

### 2.2 Request Processing

```
pipeline.iter_urls(resolver, url, role="metadata")
    ↓
networking.request_with_retries(
    method="GET",
    url=url,
    role=role,
    offline=offline,
    cache_router=cache_router,
)
    ↓
cache_policy.resolve_policy(host="api.crossref.org", role="metadata")
    ↓ Returns: CacheDecision(use_cache=True, ttl_s=259200, swrv_s=180)
    ↓
Choose client: httpx_transport.get_cached_http_client()
    ↓
networking._apply_cache_extensions(request, decision, offline=False)
    ↓ Sets: hishel_ttl=259200, hishel_stale_while_revalidate=180
    ↓
Send request (through RateLimitedTransport → HTTPTransport)
    ↓
Hishel intercepts:
  - Cache hit? Return cached response immediately (bypass limiter)
  - Cache miss? Fetch from origin
  - 304 Not Modified? Update metadata, return cached body
    ↓
networking._extract_cache_telemetry(response)
    ↓ Reads: hishel_from_cache, hishel_revalidated, hishel_stale, etc.
    ↓
Record metrics: telemetry.record_cache_hit(host, role)
    ↓
Return response to caller
```

### 2.3 Offline Mode Flow

```
--offline flag set
    ↓
cache_router.resolve_policy() → CacheDecision(use_cache=True, ...)
    ↓
networking._apply_cache_extensions(..., offline=True)
    ↓ Sets: Cache-Control: only-if-cached header
    ↓
Hishel receives request with only-if-cached:
  - In cache & fresh? Return from cache
  - Not in cache or stale? Return 504 Unsatisfiable Request
    ↓
Pipeline sees 504:
  - Log reason="offline_miss" or "blocked_offline"
  - Skip download, continue to next URL
    ↓
Telemetry: record_cache_offline_504(host, role)
```

---

## Part 3: Configuration File Format

**File**: `config/cache.yaml` (or user-provided path)

```yaml
version: 1

# Storage configuration
storage:
  kind: "file"                              # "file" | "memory"
  path: "${DOCSTOKG_DATA_ROOT}/cache/http"  # Env var interpolation
  check_ttl_every_s: 600                    # Garbage collect every 10 min

# Global cache controller
controller:
  cacheable_methods: ["GET", "HEAD"]
  cacheable_statuses: [200, 301, 308]
  allow_heuristics: false                   # Don't invent TTLs
  default: "DO_NOT_CACHE"                   # Unknown hosts not cached

# Per-host policies (normalized keys: lowercase + punycode)
hosts:
  api.crossref.org:
    ttl_s: 172800                           # 2 days
    role:
      metadata:
        ttl_s: 259200                       # 3 days (override)
        swrv_s: 180                         # 3 min stale-while-revalidate
      landing:
        ttl_s: 86400                        # 1 day
      # artifact: (not specified → never cached)

  api.openalex.org:
    ttl_s: 172800
    role:
      metadata:
        ttl_s: 259200
        swrv_s: 180
      landing:
        ttl_s: 86400

  web.archive.org:
    ttl_s: 172800
    role:
      metadata:
        ttl_s: 172800
        swrv_s: 120
      landing:
        ttl_s: 86400

  # Metadata-rich sources (2-day cache)
  export.arxiv.org: {ttl_s: 864000}         # 10 days
  europepmc.org: {ttl_s: 172800}
  eutils.ncbi.nlm.nih.gov: {ttl_s: 172800}
  api.unpaywall.org: {ttl_s: 172800}
```

---

## Part 4: Data Contracts & Serialization

### 4.1 Response Telemetry Extension

**Location**: `response.extensions["docs_network_meta"]["cache"]`

```python
{
    "from_cache": bool,                       # Served from cache
    "revalidated": bool,                      # Server returned 304
    "stored": bool,                           # Freshly stored in cache
    "stale": bool,                            # Served stale-while-revalidate
    "age_s": Optional[int],                   # Age in seconds if available
}
```

### 4.2 Manifest Record Additions

**Location**: Manifest JSONL records in `record_type="attempt"`

```python
{
    "cache": {
        "from_cache": bool,
        "revalidated": bool,
        "stored": bool,
        "stale": bool,
        "age_s": Optional[int],
    },
}
```

### 4.3 Run Summary Cache Section

**Location**: `manifest.summary.json["cache"]`

```python
{
    "hit_rate": 0.65,                         # 65% of requests from cache
    "revalidation_rate": 0.15,                # 15% revalidations (304)
    "stale_served_percent": 1.2,              # 1.2% served stale
    "bandwidth_saved_mb": 145.3,              # ~145 MB saved
    "top_hosts_by_hits": [
        {"host": "api.crossref.org", "hits": 1250, "revalidations": 180},
        {"host": "api.openalex.org", "hits": 890, "revalidations": 120},
        {"host": "web.archive.org", "hits": 450, "revalidations": 60},
        ...
    ],
}
```

---

## Part 5: Error Handling & Edge Cases

### 5.1 Configuration Errors

| Error | Behavior | Recovery |
|-------|----------|----------|
| YAML file not found | FileNotFoundError raised at startup | Fix path or use env/CLI override |
| Invalid YAML syntax | yaml.YAMLError raised at startup | Fix YAML syntax |
| Negative TTL in config | ValueError raised during validation | Fix config value |
| Unknown role name | ValueError raised during load | Only use metadata/landing/artifact |
| Empty hosts when default=DO_NOT_CACHE | ValueError raised during validation | Add at least one host |

### 5.2 Runtime Errors

| Error | Behavior | Recovery |
|-------|----------|----------|
| Cache storage full | Hishel LFU eviction (FileStorage) | Increase disk space or reduce TTL |
| Cache directory permission denied | FileNotFoundError during client init | Fix directory permissions |
| IDNA encoding fails on host | Fallback to lowercase in _normalize_host_key() | Server hostname issue, not app |
| Hishel extensions missing | Graceful defaults in _extract_cache_telemetry() | None values in telemetry dict |
| 504 from only-if-cached | Normal offline behavior | Log blocked_offline reason |

### 5.3 Cache Coherence Safeguards

```python
# Principle 1: Respect server headers
# - Honor Cache-Control, ETag, Last-Modified
# - Don't override unless explicitly configured (hishel_ttl)

# Principle 2: Conservative revalidation
# - Always include Vary headers in cache key
# - Don't cache responses with Set-Cookie (server domain scoped)

# Principle 3: Fallback on errors
# - If cache lookup fails, fetch fresh
# - If Hishel errors, log and continue

# Principle 4: Artifacts never cached
# - All PDF/binary responses bypas Hishel CacheTransport
```

---

## Part 6: Testing Requirements

### 6.1 Unit Tests

**File**: `tests/content_download/test_cache_loader.py`

```python
class TestCacheLoaderYaml:
    def test_load_basic_yaml(): ...
    def test_load_with_env_override(): ...
    def test_env_override_precedence_over_yaml(): ...
    def test_cli_override_precedence_over_env(): ...
    def test_host_key_normalization(): ...
    def test_validation_negative_ttl_fails(): ...
    def test_validation_empty_hosts_fails(): ...

class TestCacheRouter:
    def test_unknown_host_not_cached(): ...
    def test_artifact_never_cached(): ...
    def test_metadata_role_uses_role_specific_ttl(): ...
    def test_fallback_to_host_ttl(): ...
    def test_effective_policy_table_generation(): ...
```

### 6.2 Integration Tests

**File**: `tests/content_download/test_cache_integration.py`

```python
class TestCacheFlow:
    def test_first_request_stores_in_cache(): ...
    def test_second_request_hits_cache(): ...
    def test_etag_revalidation_flow(): ...
    def test_stale_while_revalidate_metadata(): ...
    def test_offline_mode_504_on_miss(): ...
    def test_offline_mode_hit_from_cache(): ...
    def test_role_isolation(): ...
    def test_cached_hits_bypass_limiter(): ...
    def test_cache_telemetry_extraction(): ...
```

---

## Part 7: Checklist for Each Phase

### Phase 1: Foundation
- [ ] Create `cache_loader.py` with all dataclasses
- [ ] Implement YAML loading with validation
- [ ] Implement env/CLI overlay logic
- [ ] Write unit tests for cache_loader
- [ ] Create `cache.yaml` template
- [ ] Add CLI args to `args.py`

### Phase 2: Routing & Policy
- [ ] Create `cache_policy.py` with CacheRouter
- [ ] Implement resolve_policy() logic
- [ ] Implement policy table generation
- [ ] Write unit tests for cache_policy
- [ ] Add effective_policy logging to startup

### Phase 3: HTTP Transport
- [ ] Modify `httpx_transport.py` for dual clients
- [ ] Implement _build_cached_client()
- [ ] Implement _build_raw_client()
- [ ] Add configure_cache_config()
- [ ] Add get_cached_http_client() / get_raw_http_client()

### Phase 4: Networking Integration
- [ ] Add _apply_cache_extensions() to networking.py
- [ ] Add _extract_cache_telemetry() to networking.py
- [ ] Modify request_with_retries() signature
- [ ] Implement cache routing logic
- [ ] Add offline mode support

### Phase 5: Telemetry
- [ ] Add cache counters to telemetry.py
- [ ] Implement record_cache_*() functions
- [ ] Implement build_cache_summary()
- [ ] Add cache section to manifest.summary.json

### Phase 6: Testing & Validation
- [ ] Write cache_loader unit tests
- [ ] Write cache_policy unit tests
- [ ] Write cache integration tests
- [ ] End-to-end test with mock server
- [ ] Validate RFC 9111 compliance

---

## Part 8: Success Criteria

✅ All components deployed  
✅ Configuration loading works with YAML/env/CLI  
✅ Policy routing selects cached vs raw client correctly  
✅ Cache hits return from storage without network I/O  
✅ Revalidation (304) flows work correctly  
✅ Offline mode returns 504 on cache miss  
✅ Telemetry captures hit rate, revalidation, stale, bandwidth  
✅ All tests passing (100% coverage for new code)  
✅ RFC 9111 compliance verified  
✅ Performance: cache lookup O(1), storage vacuum < 5s  

---

## Conclusion

This specification provides the **exact implementation guidance** needed to build the Hishel caching system with:

✅ **Clear interface definitions** - Every class, function, data structure specified  
✅ **Integration points** - Exactly where each module connects  
✅ **Algorithm details** - Logic for cache policy resolution  
✅ **Data contracts** - Telemetry format and serialization  
✅ **Testing strategy** - What to verify at each phase  
✅ **Error handling** - How to handle edge cases gracefully  

Ready for phased implementation across 4-6 weeks.

