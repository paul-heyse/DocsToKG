# Phase 2: Networking Hub Instrumentation & Request Shaping

## Objective
Integrate URL canonicalization with the networking hub to ensure:
1. Every HTTP request uses a canonical URL
2. Request shaping headers are applied per role (metadata/landing/artifact)
3. Instrumentation tracks URL normalization effectiveness
4. Strict mode can be enabled for development/canary validation

## Implementation Strategy

### A. URL Normalization Instrumentation (networking.py)

**What to add:**

1. **Metrics tracking** (module-level thread-safe counters):
   - `_url_normalization_stats` – dict with keys:
     - `normalized_total` (count of URLs processed)
     - `changed_total` (count where url != canonical)
     - `hosts_seen` (set of unique canonical hosts)
     - `roles_used` (counter by role)

2. **Strict mode checking** (env var `DOCSTOKG_URL_STRICT`):
   - If `DOCSTOKG_URL_STRICT=1`: raise ValueError when input URL != canonical
   - If `DOCSTOKG_URL_STRICT=0` (default): log WARN once per host + URL pattern

3. **Request header shaping** (role-based):
   - `metadata` role: `Accept: application/json, text/javascript;q=0.9, */*;q=0.1`
   - `landing` role: `Accept: text/html, application/xhtml+xml;q=0.9, */*;q=0.8`
   - `artifact` role: `Accept: application/pdf, */*;q=0.1`
   - All roles: preserve existing Accept header if provided

4. **Logging enhancements**:
   - Log once per host when URL is modified (params dropped)
   - Include in request extensions: `docs_url_normalized=true/false`, `docs_url_changed=true/false`

### B. Request Shaping Helpers (new: urls_networking.py)

**New module with**:
- `apply_role_headers(headers: dict, role: str) -> dict` – adds role-specific Accept headers
- `get_url_normalization_stats() -> dict` – returns metrics
- `reset_url_normalization_stats()` – for tests
- `set_strict_mode(enabled: bool)` – toggle strict validation

### C. Integration Points in request_with_retries()

After canonicalization (lines 691-699):

```python
# 1. Track metrics
_record_url_normalization(source_url, request_url, url_role)

# 2. Check strict mode
if _is_strict_mode_enabled() and source_url != request_url:
    raise ValueError(f"Non-canonical URL in strict mode: {source_url} → {request_url}")

# 3. Apply role-based headers
kwargs = _apply_role_headers(kwargs, url_role)

# 4. Log param changes (once per host)
if request_url != source_url:
    _log_url_change_once(source_url, request_url, host_hint)
```

### D. Test Coverage (test_networking_url_instrumentation.py)

**Tests to add:**
1. URL normalization metrics collection
2. Strict mode validation (pass/fail cases)
3. Role-based header injection (all three roles)
4. Parameter change logging (once per host)
5. Integration with existing request_with_retries tests

## Files to Modify

- `src/DocsToKG/ContentDownload/networking.py` – add instrumentation
- (NEW) `src/DocsToKG/ContentDownload/urls_networking.py` – helper module
- `tests/content_download/test_networking_url_instrumentation.py` – new tests
- `src/DocsToKG/ContentDownload/README.md` – document strict mode + metrics

## Success Criteria

✅ All HTTP requests use canonical URLs
✅ Request headers shaped by role
✅ Metrics tracked (normalized_total, changed_total, roles_used)
✅ Strict mode enforces non-canonical rejection
✅ Tests verify all scenarios
✅ Zero breaking changes to existing API
✅ Backward compatible with existing resolvers

## Rollout Sequence

1. Add instrumentation + helpers to networking
2. Create comprehensive tests
3. Validate metrics collection
4. Enable strict mode in CI (canary)
5. Monitor for unexpected URL changes (warnings logged)
6. Document in README
7. Ready for Phase 3 (resolver integration)

