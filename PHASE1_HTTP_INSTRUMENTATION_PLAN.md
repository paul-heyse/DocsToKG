# Phase 1: HTTP Layer Instrumentation — Implementation Plan

**Date:** 2025-10-21
**Duration:** 1–1.5 days
**Risk:** LOW
**Files to Modify:** networking.py (primary), urls.py (for hashing), httpx_transport.py (for cache metadata)

---

## Objective

Instrument `request_with_retries()` to emit telemetry events capturing:

- HTTP request metadata (host, role, method, status, URL hash)
- Cache metadata (from_cache, revalidated, stale)
- Retry/backoff info (retry_count, retry_after_s)
- Rate limiter metadata (rate_delay_ms)
- Circuit breaker state (breaker_state, breaker_recorded)
- Performance (elapsed_ms, error)

**Output:** All HTTP calls logged to `http_events` table via `emit_http_event()` helper.

---

## Architecture

### Data Flow

```
request_with_retries()
    ↓
    ├─ Pre-request: Capture start time + URL
    ├─ During request: Record retries, cache state, breaker state
    ├─ Post-request: Calculate elapsed time
    │  └─ Call: emit_http_event(telemetry, run_id, host, role, ...)
    │     └─ Populates: http_events table in telemetry.sqlite3
    └─ Return: Response (unchanged)
```

### Key Integration Points

1. **Telemetry Injection**: Pass optional `telemetry` object to `request_with_retries()`
   - Already have: `telemetry` parameter (check signature)
   - If None: Silent no-op (graceful degradation)

2. **Cache Metadata Extraction**: From Hishel transport
   - Check response extensions for `cache_status`
   - Map to: `from_cache`, `revalidated`, `stale` flags

3. **URL Hashing**: Use canonical URL + SHA256
   - Don't store raw URL (privacy)
   - Use: `urls.canonical_for_index()` + `hashlib.sha256()`

4. **Breaker State**: Already captured in `breaker_meta` dict
   - Extract: `host_state`, `resolver_state`, `error`
   - Map to: `breaker_state`, `breaker_recorded`

5. **Rate Limiter Wait**: From `docs_network_meta`
   - Already populated by rate limiter transport
   - Extract: `rate_limiter_wait_ms`

---

## Implementation Steps

### Step 1: Telemetry Parameter

In `request_with_retries()` signature (around line 607), the `telemetry` parameter likely doesn't exist yet. We need to:

```python
def request_with_retries(
    client: Optional[httpx.Client],
    method: str,
    url: str,
    *,
    role: str = DEFAULT_ROLE,
    # ... other params ...
    telemetry: Optional[Any] = None,  # NEW: Optional telemetry sink
    run_id: Optional[str] = None,     # NEW: Run ID for telemetry
    **kwargs: Any,
) -> httpx.Response:
```

### Step 2: Event Emission Location

After the response is received (around line 900+), before the response is returned:

```python
# After final response is obtained, before returning:
try:
    # Extract telemetry data
    elapsed_ms = int((time.time() - start_time_stamp) * 1000)

    # Call emit_http_event to log to telemetry
    from DocsToKG.ContentDownload.telemetry_helpers import emit_http_event

    emit_http_event(
        telemetry=telemetry,
        run_id=run_id or "unknown",
        host=request_host,
        role=policy_role,
        method=method,
        status=response.status_code,
        url_hash=_compute_url_hash(canonical_index),
        from_cache=_extract_from_cache(response),
        revalidated=_extract_revalidated(response),
        stale=_extract_stale(response),
        retry_count=retry_attempt - 1,
        retry_after_s=_extract_retry_after(response),
        rate_delay_ms=_extract_rate_delay(network_meta),
        breaker_state=_extract_breaker_state(breaker_meta),
        breaker_recorded=_extract_breaker_recorded(network_meta),
        elapsed_ms=elapsed_ms,
        error=None,
    )
except Exception as exc:
    # Silently log exceptions to avoid breaking requests
    LOGGER.debug(f"Telemetry emission failed: {exc}")
```

### Step 3: Helper Functions

Add these to `networking.py`:

```python
def _compute_url_hash(url: str) -> str:
    """Hash URL for privacy."""
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def _extract_from_cache(response: httpx.Response) -> Optional[int]:
    """Extract cache hit status from response."""
    # Check extensions set by Hishel transport
    extensions = getattr(response, "extensions", {}) or {}
    cache_status = extensions.get("cache_status")
    if cache_status == "HIT":
        return 1
    elif cache_status in ("MISS", "EXPIRED"):
        return 0
    return None

def _extract_revalidated(response: httpx.Response) -> Optional[int]:
    """Was response a 304 revalidation?"""
    return 1 if response.status_code == 304 else 0

def _extract_stale(response: httpx.Response) -> Optional[int]:
    """Was response stale (SWrV)?"""
    extensions = getattr(response, "extensions", {}) or {}
    return 1 if extensions.get("stale") else 0

def _extract_retry_after(response: httpx.Response) -> Optional[int]:
    """Extract Retry-After header value."""
    try:
        retry_after_str = response.headers.get("Retry-After")
        if retry_after_str:
            return int(float(retry_after_str))
    except (ValueError, TypeError):
        pass
    return None

def _extract_rate_delay(network_meta: Dict[str, Any]) -> Optional[int]:
    """Extract rate limiter wait time."""
    if isinstance(network_meta, dict):
        rate_info = network_meta.get("rate_limiter") or {}
        if isinstance(rate_info, dict):
            delay = rate_info.get("wait_ms")
            if isinstance(delay, (int, float)):
                return int(delay)
    return None

def _extract_breaker_state(breaker_meta: Dict[str, Any]) -> Optional[str]:
    """Extract breaker state: closed/half_open/open."""
    if isinstance(breaker_meta, dict):
        host_state = breaker_meta.get("host_state")
        if host_state:
            # Map pybreaker states to standardized names
            if "open" in str(host_state).lower():
                return "open"
            elif "half" in str(host_state).lower():
                return "half_open"
            else:
                return "closed"
    return None

def _extract_breaker_recorded(network_meta: Dict[str, Any]) -> Optional[str]:
    """Extract breaker recorded outcome: success/failure/none."""
    if isinstance(network_meta, dict):
        breaker_info = network_meta.get("breaker") or {}
        if isinstance(breaker_info, dict):
            recorded = breaker_info.get("recorded")
            if recorded in ("success", "failure"):
                return recorded
    return None
```

---

## Testing Strategy

### Unit Tests (test_networking_telemetry.py)

```python
def test_emit_http_event_basic():
    """Test basic HTTP event emission."""
    telemetry = MockTelemetry()
    response = request_with_retries(
        client=mock_client,
        method="GET",
        url="https://example.org/test",
        telemetry=telemetry,
        run_id="test-run-123",
    )

    # Verify event was emitted
    assert telemetry.emit_http_event.called
    event = telemetry.emit_http_event.call_args[1]
    assert event["host"] == "example.org"
    assert event["role"] == "metadata"
    assert event["method"] == "GET"
    assert event["status"] == 200
    assert event["run_id"] == "test-run-123"

def test_emit_http_event_with_retries():
    """Test telemetry includes retry count."""
    # ... simulate 2 retries, then success
    event = telemetry.emit_http_event.call_args[1]
    assert event["retry_count"] == 2

def test_emit_http_event_cache_hit():
    """Test cache metadata extraction."""
    # ... response with cache hit
    event = telemetry.emit_http_event.call_args[1]
    assert event["from_cache"] == 1

def test_emit_http_event_none_telemetry():
    """Test graceful handling when telemetry=None."""
    response = request_with_retries(
        client=mock_client,
        method="GET",
        url="https://example.org/test",
        telemetry=None,  # None should not crash
    )
    assert response.status_code == 200  # Should still work

def test_url_hash_privacy():
    """Test URLs are hashed, not stored raw."""
    event = telemetry.emit_http_event.call_args[1]
    # url_hash should be short hash, not full URL
    assert len(event["url_hash"]) == 16  # SHA256 first 16 chars
    assert "example.org" not in event["url_hash"]
```

### Integration Tests

```python
def test_request_with_retries_with_sqlite_telemetry(tmp_path):
    """Test full integration: HTTP request → SQLite table."""
    db_path = tmp_path / "telemetry.sqlite3"

    # Initialize schema
    initialize_telemetry_schema(db_path)
    telemetry = SqliteSink(db_path)

    # Make request
    response = request_with_retries(
        client=mock_client,
        method="GET",
        url="https://example.org/test",
        telemetry=telemetry,
        run_id="test-run",
    )

    # Verify table populated
    cx = sqlite3.connect(db_path)
    rows = cx.execute("SELECT COUNT(*) FROM http_events").fetchone()
    assert rows[0] == 1

    # Verify record content
    row = cx.execute("SELECT * FROM http_events LIMIT 1").fetchone()
    assert row["host"] == "example.org"
    assert row["status"] == 200
```

---

## Success Criteria

- ✅ All HTTP calls logged to `http_events` table
- ✅ No raw URLs stored (only hashes)
- ✅ Cache metadata captured (from_cache, revalidated, stale)
- ✅ Retry counts accurate
- ✅ Breaker state captured
- ✅ Rate delay captured
- ✅ Graceful degradation (telemetry=None doesn't break requests)
- ✅ >90% test coverage
- ✅ Zero linting/type errors
- ✅ Response unchanged (transparent instrumentation)

---

## Files to Create/Modify

| File | Changes | LOC |
|------|---------|-----|
| `networking.py` | Add telemetry param, event emission, helpers | +80 |
| `tests/content_download/test_networking_telemetry.py` | Unit + integration tests | +200 |

**Total Phase 1: ~280 LOC (80 prod + 200 test)**

---

## Timeline

1. **Hour 1:** Implement helpers + event emission in networking.py
2. **Hour 0.5:** Wire telemetry parameter through call sites
3. **Hour 0:** Testing + verification

**Ready to start!**
