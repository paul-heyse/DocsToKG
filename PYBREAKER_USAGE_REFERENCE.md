# Pybreaker Usage Reference for DocsToKG.ContentDownload

Based on **pybreaker 1.4.1** and the comprehensive library documentation provided.

## Quick API Reference

### States
```python
import pybreaker

# States (strings)
pybreaker.STATE_CLOSED      # "closed" — accepting calls normally
pybreaker.STATE_OPEN        # "open" — rejecting calls immediately
pybreaker.STATE_HALF_OPEN   # "half_open" — one trial call allowed
```

### Creating a Breaker
```python
import pybreaker

breaker = pybreaker.CircuitBreaker(
    fail_max=5,                        # Trip after 5 consecutive failures
    reset_timeout=60,                  # Try recovery after 60s
    success_threshold=None,            # (optional) require N successes in half-open to close
    exclude=None,                      # (optional) exceptions to ignore
    listeners=None,                    # (optional) telemetry hooks
    state_storage=None,                # (optional) for cross-process sharing
    name="my-breaker",                 # (optional) friendly name for logs
)
```

### Using the Breaker

#### 1. As a Decorator
```python
@breaker
def call_api():
    response = httpx.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# Calling it
try:
    data = call_api()
except pybreaker.CircuitBreakerError:
    # Breaker is open; short-circuited
    logger.error("API circuit breaker is open")
```

#### 2. As a Direct Call
```python
def call_api():
    response = httpx.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# Wrapping it
try:
    data = breaker.call(call_api)
except pybreaker.CircuitBreakerError:
    logger.error("API circuit breaker is open")
```

#### 3. As a Context Manager
```python
try:
    with breaker.calling():
        response = httpx.get("https://api.example.com/data")
        response.raise_for_status()
except pybreaker.CircuitBreakerError:
    logger.error("API circuit breaker is open")
```

### Failure Classification

**What counts as failure (trip the breaker):**
- Any exception raised by the wrapped function/call
- (Or custom predicate on response if not using decorator)

**How to exclude business errors:**
```python
# Define which exceptions to IGNORE (don't count as failures)
breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    exclude=[
        lambda e: isinstance(e, httpx.HTTPStatusError) 
                  and e.response.status_code < 500
        # ^ Excludes 4xx client errors; only 5xx trips the breaker
    ]
)

# Now 404s won't trip the breaker
@breaker
def call_api():
    response = httpx.get("https://api.example.com/item/12345")
    response.raise_for_status()  # Raises HTTPStatusError on 404
    return response.json()
```

### Monitoring & Inspection

```python
breaker.current_state       # "closed" | "open" | "half_open"
breaker.fail_counter        # Read-only: consecutive failures
breaker.success_counter     # Read-only: consecutive successes

# Manually tune (if needed)
breaker.fail_max = 3                # Change threshold
breaker.reset_timeout = 120         # Change recovery window
breaker.success_threshold = 2       # (if set) require 2 successes to close
```

### Operational Controls

```python
# Force states (for maintenance or testing)
breaker.open()              # Force open (block calls)
breaker.half_open()         # Force half-open (allow 1 trial)
breaker.close()             # Force closed (resume normal)
```

### Telemetry via Listeners

```python
import pybreaker

class MyListener(pybreaker.CircuitBreakerListener):
    def before_call(self, cb, func, *args, **kwargs):
        print(f"About to call {func.__name__} (state: {cb.current_state})")
    
    def success(self, cb):
        print(f"Call succeeded (fail_counter reset to 0)")
    
    def failure(self, cb, exc):
        print(f"Call failed: {exc} (fail_counter: {cb.fail_counter})")
    
    def state_change(self, cb, old_state, new_state):
        print(f"State changed: {old_state} → {new_state}")

breaker = pybreaker.CircuitBreaker(
    listeners=[MyListener()],
    name="api-breaker",
    fail_max=5,
    reset_timeout=60
)
```

### Cross-Process Sharing (Redis)

```python
import redis
import pybreaker

# DO NOT use decode_responses=True
redis_client = redis.StrictRedis(
    host="localhost", port=6379, db=0,
    socket_timeout=2.0
)

storage = pybreaker.CircuitRedisStorage(
    pybreaker.STATE_CLOSED,  # Initial state
    redis_client,
    namespace="my-service-api"  # Unique per breaker
)

breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    state_storage=storage
)
```

---

## How DocsToKG.ContentDownload Uses Pybreaker

### BreakerRegistry Pattern

Instead of creating individual breakers, we use a centralized `BreakerRegistry`:

```python
from DocsToKG.ContentDownload.breakers import (
    BreakerRegistry, BreakerConfig, RequestRole, BreakerOpenError
)

# 1. Load config (from YAML + env + CLI overlays)
cfg = load_breaker_config(yaml_path="config/breakers.yaml", env=os.environ)

# 2. Create registry
registry = BreakerRegistry(cfg, cooldown_store=cooldown_store)

# 3. Pre-flight check (before sending request)
try:
    registry.allow(
        host="api.crossref.org",
        role=RequestRole.METADATA,
        resolver="crossref"  # optional
    )
except BreakerOpenError as e:
    logger.error(f"Breaker open: {e}")
    return error_response

# 4. Send HTTP request (via Tenacity, caching, etc.)
response = send_request(...)

# 5. Post-request update
if response.status_code in {429, 500, 502, 503, 504}:
    registry.on_failure(
        host="api.crossref.org",
        role=RequestRole.METADATA,
        status=response.status_code,
        retry_after_s=parsed_retry_after
    )
elif response.status_code in {401, 403, 404, 410, 451}:
    # Neutral — don't update breaker
    pass
else:
    registry.on_success(
        host="api.crossref.org",
        role=RequestRole.METADATA
    )
```

### Key Differences from Raw Pybreaker

| Feature | Raw Pybreaker | BreakerRegistry |
|---------|---------------|-----------------|
| **State machine** | ✅ (Closed, Open, Half-Open) | ✅ (uses pybreaker internally) |
| **Failure classification** | Manual (via exclude=[...]) | Configured (BreakerClassification) |
| **Retry-After** | ❌ Not aware | ✅ Cooldown overrides |
| **Rolling window** | ❌ Not available | ✅ N failures in W seconds |
| **Per-role tuning** | ❌ Per-breaker only | ✅ Per-(host, role) |
| **Half-open probes/role** | Fixed (1) | Configurable per role |
| **Cross-process** | ✅ (Redis) | ✅ (SQLite, Redis, in-memory) |

---

## Pybreaker Best Practices (from Official Docs)

### 1. Keep Breakers Global (Singletons)
```python
# ❌ BAD
def make_request():
    breaker = CircuitBreaker()  # New breaker every call
    return breaker.call(api_call)

# ✅ GOOD
api_breaker = CircuitBreaker(name="api")  # Global/module-level

def make_request():
    return api_breaker.call(api_call)
```

### 2. Name Your Breakers
```python
# ✅ GOOD - names appear in logs
breaker = CircuitBreaker(name="api.crossref.org", fail_max=5)
```

### 3. Use success_threshold in Half-Open
```python
# ✅ If recovery is flaky, require 2 successes before closing
breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    success_threshold=2  # Require 2 consecutive successes in half-open
)
```

### 4. Set Timeouts on HTTP Client (Not In Breaker)
```python
# ✅ GOOD - timeouts work with breaker
client = httpx.Client(timeout=5.0)

@breaker
def call_api():
    response = client.get("https://api.example.com")
    response.raise_for_status()
    return response
```

### 5. Preserve Original Exceptions (Optional)
```python
# By default, CircuitBreakerError masks the original exception
breaker = CircuitBreaker(
    throw_new_error_on_trip=False  # Re-raise original exception when breaker trips
)
```

---

## Testing with Pybreaker

```python
import pytest
import pybreaker

def test_breaker_trips_on_consecutive_failures():
    breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=60)
    
    def failing_call():
        raise ValueError("API is down")
    
    # Fail 3 times
    for _ in range(3):
        with pytest.raises(ValueError):
            breaker.call(failing_call)
    
    # 4th call short-circuits
    with pytest.raises(pybreaker.CircuitBreakerError):
        breaker.call(failing_call)
    
    assert breaker.current_state == pybreaker.STATE_OPEN

def test_half_open_recovery():
    breaker = pybreaker.CircuitBreaker(
        fail_max=2,
        reset_timeout=0.1,  # 100ms for testing
        success_threshold=2
    )
    
    def failing_call():
        raise ValueError("API is down")
    
    # Trip the breaker
    for _ in range(2):
        with pytest.raises(ValueError):
            breaker.call(failing_call)
    
    # Wait for reset_timeout
    time.sleep(0.2)
    
    # Now in half-open state
    assert breaker.current_state == pybreaker.STATE_HALF_OPEN
    
    # Define a working call
    def working_call():
        return "success"
    
    # Need 2 successes to close
    result1 = breaker.call(working_call)
    result2 = breaker.call(working_call)
    
    assert breaker.current_state == pybreaker.STATE_CLOSED
    assert result2 == "success"
```

---

## Common Pitfalls & How DocsToKG Avoids Them

| Pitfall | Problem | DocsToKG Solution |
|---------|---------|-------------------|
| **No timeouts** | Request hangs forever | Set httpx client timeout |
| **4xx counts as failure** | Business errors trip breaker | BreakerClassification.neutral_statuses |
| **No Retry-After support** | Ignore server signals | Cooldown store + Retry-After header parsing |
| **Clock drift in multiprocess** | Deadlines become invalid | Wall-clock storage + monotonic runtime |
| **Too aggressive tuning** | Breaker flaps | Auto-tuner clamps suggestions to safe ranges |
| **No observability** | Can't debug why breaker opened | NetworkBreakerListener → telemetry |
| **Redis decode_responses=True** | Serialization errors | Never enable; use bytes ✅ |

---

## Example: Complete Integration

```python
# networking.py
from DocsToKG.ContentDownload.breakers import (
    BreakerRegistry, BreakerOpenError, RequestRole
)

def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    breaker_registry: Optional[BreakerRegistry] = None,
    role: RequestRole = RequestRole.METADATA,
    resolver: Optional[str] = None,
    **kwargs
) -> httpx.Response:
    """Execute HTTP request with retry + circuit breaker."""
    
    host = extract_host(url)
    
    # PRE-FLIGHT: Check if breaker is open
    if breaker_registry:
        try:
            breaker_registry.allow(host, role=role, resolver=resolver)
        except BreakerOpenError as e:
            logger.warning(f"Breaker open: {e}")
            raise  # Fail fast, don't retry
    
    # SEND: (with Tenacity retry logic)
    try:
        response = client.request(method, url, **kwargs)
    except Exception as e:
        # Network error → on_failure
        if breaker_registry:
            breaker_registry.on_failure(
                host, role=role, resolver=resolver, exception=e
            )
        raise
    
    # POST-RESPONSE: Update breaker state
    if breaker_registry:
        retry_after_s = parse_retry_after_header(response)
        
        if response.status_code in {429, 500, 502, 503, 504}:
            breaker_registry.on_failure(
                host,
                role=role,
                resolver=resolver,
                status=response.status_code,
                retry_after_s=retry_after_s
            )
        elif response.status_code in {401, 403, 404, 410, 451}:
            # Neutral — don't update
            pass
        else:
            breaker_registry.on_success(host, role=role, resolver=resolver)
    
    return response
```

That's it! Pybreaker handles the state machine; we handle the classification, telemetry, and operational controls.

