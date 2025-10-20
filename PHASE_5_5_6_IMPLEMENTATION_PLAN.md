# Phase 5.5-5.6 Implementation Plan: HTTP & Rate-Limiting

## Overview

This document outlines the implementation of two critical subsystems:
- **Phase 5.5**: HTTP Client Factory (HTTPX + Hishel + Tenacity)
- **Phase 5.6**: Rate-Limit Façade (pyrate-limiter with multi-window support)

These phases build on Phase 5.4's settings/CLI foundation and integrate key libraries for robust networking.

---

## Phase 5.5: HTTP Client Factory

### Architecture

**Key Components:**
1. `network/policy.py` - HTTP policy constants (timeouts, headers, caching)
2. `network/client.py` - HTTPX client factory with lifecycle
3. `network/instrumentation.py` - Request/response hooks for telemetry
4. `network/redirect.py` - Safe redirect validation
5. `network/retry.py` - Tenacity backoff policies

**Design Principles (from httpx.md, hishel.md, tenacity.md):**

- **One shared HTTPX client** (sync) for all network I/O
- **Redirects disabled globally** - validated per-hop via URL security gate
- **Hishel caching** (RFC 9111 compliant) with file storage
- **Tenacity backoff** for 429/5xx with full-jitter exponential
- **Transport-level retries** for connect errors only
- **Streaming by default** - never buffer full bodies
- **Event hooks** for instrumentation (request/response timing)
- **Thread-safe** for concurrent access

### Module Breakdown

#### `network/policy.py`
```python
# HTTP timeouts (per-phase): connect < read
HTTP_CONNECT_TIMEOUT = 5.0  # seconds
HTTP_READ_TIMEOUT = 30.0
HTTP_WRITE_TIMEOUT = 15.0
HTTP_POOL_TIMEOUT = 5.0

# Connection pooling
MAX_CONNECTIONS = 100
MAX_KEEPALIVE_CONNECTIONS = 20
KEEPALIVE_EXPIRY = 5.0

# Hishel cache controller (RFC 9111)
# - Respect Cache-Control
# - Validate with ETag/Last-Modified
# - Allow heuristic caching
# - Private cache (not shared)
# - No stale-on-error (be strict)

# User-Agent format: "ontofetch/VERSION (+URL) run_id"
USER_AGENT_FORMAT = "ontofetch/{version} (+{project_url}) {run_id}"

# Allowed schemes: https (with http→https upgrade)
# Disabled: ftp, gopher, etc.
```

#### `network/client.py`
```python
class HttpClientFactory:
    """Manages HTTPX client lifecycle with pooling and caching."""
    
    def __init__(self, settings: HttpSettings, run_id: str):
        """Initialize with settings and run_id for user-agent."""
        self.settings = settings
        self.run_id = run_id
        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()
    
    def get_client(self) -> httpx.Client:
        """Get or create the shared HTTPX client."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client
    
    def _create_client(self) -> httpx.Client:
        """Create HTTPX client with Hishel + hooks."""
        # Build SSL context
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        
        # Build cache transport (Hishel)
        base_transport = httpx.HTTPTransport(
            retries=2,  # connect errors only
            verify=ssl_ctx,
        )
        cache_transport = hishel.CacheTransport(
            transport=base_transport,
            storage=hishel.FileStorage(
                base_path=self.settings.cache_dir,
                ttl=self.settings.cache_ttl,
            ),
            controller=hishel.Controller(
                cacheable_methods=["GET", "HEAD"],
                cacheable_status_codes=[200, 301, 308],
                allow_heuristics=False,  # strict RFC
                cache_private=True,
            ),
        )
        
        # Create client
        client = httpx.Client(
            transport=cache_transport,
            verify=ssl_ctx,
            timeout=httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT,
                read=HTTP_READ_TIMEOUT,
                write=HTTP_WRITE_TIMEOUT,
                pool=HTTP_POOL_TIMEOUT,
            ),
            limits=httpx.Limits(
                max_connections=MAX_CONNECTIONS,
                max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                keepalive_expiry=KEEPALIVE_EXPIRY,
            ),
            follow_redirects=False,  # explicit, audited redirects only
            http2=True,
            event_hooks={
                "request": [self._on_request],
                "response": [self._on_response],
            },
        )
        return client
    
    def _on_request(self, request: httpx.Request) -> None:
        """Hook: stamp correlation ID, timing, user-agent."""
        # Add run_id to user-agent
        # Record monotonic start time
        
    def _on_response(self, response: httpx.Response) -> None:
        """Hook: compute timings, extract cache status, emit event."""
        # Compute: dns_ms (if available), connect_ms, ttfb_ms, read_ms
        # Extract cache status from extensions
        # Emit net.request event with all metrics
    
    def close(self) -> None:
        """Close the client and release resources."""
        if self._client:
            self._client.close()
            self._client = None
```

#### `network/instrumentation.py`
```python
def on_request_hook(request: httpx.Request) -> None:
    """Record request start time and metadata."""
    # Use contextvars to store per-request state
    # Store: request_id (UUID), method, host, path, ts_start (monotonic)

def on_response_hook(response: httpx.Response) -> None:
    """Compute timings and emit net.request event."""
    # Retrieve stored state
    # Compute: elapsed_ms, ttfb_ms (if available)
    # Extract cache status: response.extensions["from_cache"]
    # Extract revalidation: response.extensions["revalidated"]
    # Emit structured event:
    #   {service, host, method, status, elapsed_ms, ttfb_ms, 
    #    request_id, cache_status, http2, reused_conn}

def on_error_hook(exc: httpx.RequestError) -> None:
    """Map HTTPX exceptions to domain error taxonomy."""
    # ConnectTimeout → E_NET_CONNECT
    # ReadTimeout → E_NET_READ
    # SSLError → E_TLS
    # RemoteProtocolError → E_NET_PROTOCOL
    # Emit net.error event
```

#### `network/redirect.py`
```python
def safe_get_with_redirect(
    client: httpx.Client,
    url: str,
    max_hops: int = 5,
    security_gate: SecurityPolicy = None,
) -> httpx.Response:
    """
    Make GET request with explicit redirect auditing.
    
    Every redirect hop is validated by security_gate before following.
    Builds audit trail of all hops.
    """
    response = client.get(url, follow_redirects=False)
    hops = [url]
    
    for _ in range(max_hops):
        if response.status_code not in {301, 302, 303, 307, 308}:
            break
        
        location = response.headers.get("location")
        if not location:
            break
        
        # Validate location with security gate
        if security_gate and not security_gate.validate_url(location):
            raise URLNotAllowedError(f"Redirect target blocked: {location}")
        
        hops.append(location)
        response = client.get(location, follow_redirects=False)
    
    # Record audit trail
    return response
```

#### `network/retry.py`
```python
def create_http_retry_policy(
    max_attempts: int = 6,
    max_delay_seconds: int = 30,
) -> Retrying:
    """
    Create Tenacity retry policy for HTTP calls.
    
    - Retries on: 429, 5xx (idempotent methods only)
    - Strategy: full-jitter exponential backoff
    - Honors Retry-After header
    - Max delay overall
    """
    def wait_with_retry_after(retry_state):
        exc = retry_state.outcome.exception()
        if exc and hasattr(exc, "response"):
            ra = exc.response.headers.get("Retry-After")
            if ra:
                try:
                    return int(ra)
                except ValueError:
                    # Parse HTTP-date
                    dt = email.utils.parsedate_to_datetime(ra)
                    return max(0, (dt - datetime.now(dt.tzinfo)).total_seconds())
        return 0  # Let exponential backoff handle it
    
    return Retrying(
        stop=stop_after_delay(max_delay_seconds),
        wait=wait_random_exponential(multiplier=0.5, max=min(60, max_delay_seconds)),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        )) | retry_if_result(lambda r: r.status_code in {429, 500, 502, 503, 504}),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
        reraise=True,
    )
```

---

## Phase 5.6: Rate-Limit Façade

### Architecture

**Key Components:**
1. `ratelimit/config.py` - RateSpec parsing & normalization
2. `ratelimit/manager.py` - pyrate-limiter façade
3. `ratelimit/instrumentation.py` - Rate limit event telemetry

**Design Principles (from pyrate-limiter.md):**

- **Single façade** `acquire(service, host, weight, mode)` for all rate limiting
- **Multi-window support** (e.g., 5/sec AND 300/min)
- **Per-key registry** with `{service}:{host}` keys
- **Two modes**: block (sleep) or fail-fast
- **Single-process**: InMemoryBucket (default)
- **Multi-process**: SQLiteBucket with file locks (when configured)
- **Weighted requests** for differential quotas

### Module Breakdown

#### `ratelimit/config.py`
```python
class RateSpec:
    """Normalized rate specification."""
    
    limit: int  # events allowed
    interval_ms: int  # per this many milliseconds
    
    @property
    def rps(self) -> float:
        """Requests per second."""
        return (self.limit * 1000) / self.interval_ms

def parse_rate_string(spec: str) -> RateSpec:
    """Parse '5/second' or '300/minute' into RateSpec."""
    # Extract number and unit
    # Normalize to interval_ms

def parse_rate_dict(rates: Dict[str, str]) -> List[RateSpec]:
    """Parse {'ols': '4/second', 'bioportal': '2/second'}."""
    # Parse each value, sort by interval ascending
    # Validate rate ordering via pyrate_limiter.validate_rate_list()

def normalize_per_service_rates(
    config: RateLimitSettings,
) -> Dict[str, List[RateSpec]]:
    """Build registry of service→rates from settings."""
    # {
    #   "ols": [Rate(4, Duration.SECOND)],
    #   "bioportal": [Rate(2, Duration.SECOND)],
    #   "_": [Rate(8, Duration.SECOND)]  # default
    # }
```

#### `ratelimit/manager.py`
```python
class RateLimitManager:
    """
    Façade for pyrate-limiter.
    
    Coordinates acquiring slots with service/host-based keying.
    Supports block (sleep) and fail-fast modes.
    """
    
    def __init__(
        self,
        config: RateLimitSettings,
        shared_dir: Optional[Path] = None,  # for SQLiteBucket
    ):
        """Initialize with settings."""
        self.config = config
        self.shared_dir = shared_dir
        self._registry: Dict[str, Limiter] = {}
        self._lock = threading.Lock()
    
    def acquire(
        self,
        service: Optional[str] = None,
        host: Optional[str] = None,
        *,
        weight: int = 1,
        mode: Literal["block", "fail"] = "block",
    ) -> bool:
        """
        Acquire rate-limit slot.
        
        Args:
            service: service name (e.g., "ols", "bioportal")
            host: target host (e.g., "ebi.ac.uk")
            weight: slots to consume (default 1)
            mode: "block" = sleep until available, "fail" = raise if full
        
        Returns:
            True if acquired successfully
        
        Raises:
            BucketFullException: if mode="fail" and limit exceeded
        
        Emits:
            ratelimit.acquire event with {service, host, weight, mode, 
                                         blocked_ms, outcome}
        """
        key = self._make_key(service, host)
        limiter = self._get_or_create_limiter(key)
        
        start_ms = current_time_ms()
        
        try:
            if mode == "block":
                limiter.try_acquire(key, weight=weight, blocking=True)
            else:  # fail
                limiter.try_acquire(key, weight=weight, blocking=False)
            
            blocked_ms = current_time_ms() - start_ms
            emit_acquire_event(
                key=key,
                weight=weight,
                mode=mode,
                blocked_ms=blocked_ms,
                outcome="ok",
            )
            return True
        
        except BucketFullException as e:
            blocked_ms = current_time_ms() - start_ms
            emit_acquire_event(
                key=key,
                weight=weight,
                mode=mode,
                blocked_ms=blocked_ms,
                outcome="exceeded",
                rate_info=str(e.meta_info),
            )
            raise
    
    def _make_key(self, service: Optional[str], host: Optional[str]) -> str:
        """Build rate-limit key."""
        return f"{service or '_'}:{host or 'default'}"
    
    def _get_or_create_limiter(self, key: str) -> Limiter:
        """Get limiter for key, creating if needed."""
        if key not in self._registry:
            with self._lock:
                if key not in self._registry:
                    rates = self.config.per_service.get(
                        key.split(":")[0],  # extract service
                        [self.config.default_rate],
                    )
                    bucket = self._create_bucket(key, rates)
                    self._registry[key] = Limiter(bucket)
        return self._registry[key]
    
    def _create_bucket(self, key: str, rates: List[RateSpec]) -> AbstractBucket:
        """Create appropriate bucket (in-mem or SQLite)."""
        if self.shared_dir:
            # Multi-process: SQLite with file lock
            return SQLiteBucket.init(
                rates,
                db_path=self.shared_dir / "ratelimit.sqlite",
            )
        else:
            # Single-process: in-memory
            return InMemoryBucket(rates)

def emit_acquire_event(
    key: str,
    weight: int,
    mode: str,
    blocked_ms: int,
    outcome: str,
    rate_info: Optional[str] = None,
) -> None:
    """Emit structured rate-limit acquisition event."""
    # Structured log with: {key, weight, mode, blocked_ms, outcome, rate_info}
```

---

## Integration Points

### With Settings (Phase 5.4)

```python
@dataclass
class HttpSettings:
    # ... existing fields ...
    cache_dir: Path  # Where Hishel stores cache
    cache_ttl: int  # Cache TTL in seconds
    http2_enabled: bool  # Enable HTTP/2

@dataclass
class RateLimitSettings:
    default_rate: Optional[RateSpec]  # e.g., "8/second"
    per_service: Dict[str, RateSpec]  # service-specific rates
    shared_dir: Optional[Path]  # For multi-process SQLite
```

### With CLI (Phase 5.5 integration)

```bash
# Show rate limits
$ ontofetch-cli settings show --format json | jq .ratelimit

# Run with specific rate limits
$ ontofetch-cli --config settings.yaml pull hp

# Dry-run to see what would happen
$ ontofetch-cli --dry-run pull hp
```

### With Observability (Events)

- `net.request` - emitted by HTTP hooks
- `net.error` - emitted on HTTP failures
- `ratelimit.acquire` - emitted per acquisition
- `download.fetch` - higher-level event referencing request_id

---

## Testing Strategy

### Phase 5.5 (HTTP) - 20+ Tests

1. **Client Factory**
   - ✅ Creates HTTPX client with correct timeouts
   - ✅ Reuses client on subsequent calls
   - ✅ Thread-safe access
   - ✅ Properly closes client

2. **Caching (Hishel)**
   - ✅ RFC 9111 compliance (respects Cache-Control)
   - ✅ ETag/Last-Modified validation
   - ✅ Cache hit/miss telemetry
   - ✅ File storage persistence

3. **Redirects**
   - ✅ Disabled globally
   - ✅ Safe redirect validation per-hop
   - ✅ Blocks unsafe hops
   - ✅ Audit trail recorded

4. **Retries (Tenacity)**
   - ✅ Retries on connect errors
   - ✅ Respects Retry-After header
   - ✅ Full-jitter exponential backoff
   - ✅ Eventually gives up

5. **Instrumentation**
   - ✅ Request/response timing accuracy
   - ✅ Cache status in events
   - ✅ Error mapping to taxonomy
   - ✅ User-agent contains run_id

### Phase 5.6 (Rate-Limiting) - 25+ Tests

1. **Single-Window (Simple)**
   - ✅ Enforces limit per window
   - ✅ Block mode sleeps
   - ✅ Fail mode raises

2. **Multi-Window (Complex)**
   - ✅ Enforces ALL rates simultaneously
   - ✅ Rates must be ordered correctly
   - ✅ Blocking respects multiple windows

3. **Per-Service Config**
   - ✅ Parses rate specs correctly
   - ✅ Applies service-specific rates
   - ✅ Falls back to default

4. **Weighted Requests**
   - ✅ Weight > 1 consumes multiple slots
   - ✅ Atomic: either all slots fit or fail

5. **Multi-Process (SQLite)**
   - ✅ Two processes share bucket
   - ✅ File locking ensures atomicity
   - ✅ Rates are global across processes

6. **Telemetry**
   - ✅ Events record blocked_ms
   - ✅ Success/failure outcomes tracked
   - ✅ Rate info included on exceed

---

## Success Criteria

✅ All libraries used per their documentation best practices
✅ 45+ comprehensive tests (20 HTTP + 25 rate-limiting)
✅ Thread-safe and process-safe operations
✅ Zero network calls in tests (MockTransport)
✅ Full integration with Phase 5.4 settings
✅ Instrumentation ready for observability
✅ Production-quality error handling
✅ Type hints throughout

---

## Next Steps

1. Implement Phase 5.5: HTTP Client Factory
   - network/policy.py
   - network/client.py
   - network/instrumentation.py
   - network/redirect.py
   - network/retry.py
   - tests/network/ (20 tests)

2. Implement Phase 5.6: Rate-Limit Façade
   - ratelimit/config.py
   - ratelimit/manager.py
   - ratelimit/instrumentation.py
   - tests/ratelimit/ (25 tests)

3. Integration testing
   - End-to-end download with rate-limiting
   - Cache hit scenarios
   - Redirect audit trails

