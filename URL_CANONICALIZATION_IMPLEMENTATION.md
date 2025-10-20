# URL Canonicalization Implementation Summary

**Date**: 2025-10-20  
**Status**: Core infrastructure complete; integration and instrumentation pending  
**Owner**: DocsToKG.ContentDownload  

## Overview

This implementation establishes **one authoritative source of truth** for URL normalization across the ContentDownload module, ensuring stable cache keys, consistent rate-limiter/breaker routing, deterministic dedupe indices, and predictable request shaping by role.

## Completed Components

### 1. `src/DocsToKG/ContentDownload/urls.py` (✅ Enhanced)

**What was added/improved:**

- **`canonical_for_index(url: str) -> str`**
  - Returns RFC 3986/3987 normalized URL for dedupe/manifest indexing
  - No param filtering; preserves all query strings
  - Handles: case normalization, percent-encoding, dot-segments, default ports, IDN→punycode, fragment removal
  - Used by: resolvers (when emitting candidates), pipeline (dedupe/resume), telemetry

- **`canonical_for_request(url: str, role: Role, origin_host: str | None) -> str`**
  - Normalizes URL + applies role-based param filtering for HTTP requests
  - Roles:
    - `metadata`: preserve all params (signed APIs, auth)
    - `landing`: drop known trackers unless in allowlist
    - `artifact`: preserve all params (CDN signatures, etc.)
  - Supports relative URLs via `origin_host` parameter
  - Used by: networking hub just before issuing requests

- **`canonical_host(url: str) -> str`**
  - Extracts normalized hostname (lowercase, punycode, no port)
  - Used by: limiter/breaker for consistent key derivation, HTTPX mounts
  - Safe for thread-safe key hashing

- **`DROP_PARAMS_DEFAULT` constant**
  - Frozen set of known tracker parameters to drop on `landing` role
  - Includes: `utm_*`, `gclid`, `fbclid`, `yclid`, `mc_cid`, `ref`, `spm`, `igshid`, `mkt_tok`, `msclkid`, `_hsenc`, etc.
  - Conservative and excludes auth/signature params intentionally

- **`UrlPolicy` dataclass + configuration functions**
  - `configure_url_policy()` – runtime override (tests, CLI bootstrap)
  - `get_url_policy()` – return frozen copy
  - `reset_url_policy_for_tests()` – test fixture hook
  - `parse_param_allowlist_spec()` – parse CLI/env allowlist strings
  - Respects environment variables: `DOCSTOKG_URL_DEFAULT_SCHEME`, `DOCSTOKG_URL_FILTER_LANDING`, `DOCSTOKG_URL_PARAM_ALLOWLIST`

- **Comprehensive module docstring**
  - Full RFC 3986/3987 normalization rules with examples
  - Role-based behavior documentation
  - Gotchas (e.g., port 80 ≠ default for HTTPS when scheme=https)
  - Integration guidance for resolvers, pipeline, networking
  - Policy configuration surfaces

### 2. Test Suite (`tests/content_download/test_urls.py`) (✅ Created)

**22 comprehensive tests covering:**

- **RFC normalization**: case, escapes, dot-segments, ports, IDN/punycode, fragments
- **Role-specific behavior**: param filtering, allowlists, relative URL handling
- **Host extraction**: lowercase, port stripping, punycode
- **Policy configuration**: defaults, environment overrides, parsing
- **Edge cases**: None rejection, IPv4 addresses, special characters
- **All tests passing** ✅

Example test cases:
```python
# Case normalization
canonical_for_index("HTTP://EXAMPLE.COM/path") == "http://example.com/path"

# Landing role filters trackers
result = canonical_for_request("...?utm_source=x&id=1", role="landing")
assert "utm_source" not in result and "id=1" in result  # w/ allowlist

# Port 80 kept when scheme=https (gotcha!)
canonical_for_index("www.example.com:80/foo") contains ":80"
```

### 3. README Documentation (`src/DocsToKG/ContentDownload/README.md`) (✅ Updated)

**New section: "URL Canonicalization & Request Shaping"**

Covers:
- Canonicalization surfaces (`canonical_for_index`, `canonical_for_request`, `canonical_host`)
- Three roles and their param-handling semantics
- Request header shaping table (Accept, Cache-Control per role)
- Resolver integration examples
- Policy configuration (CLI, environment, runtime)
- Cache/limiter alignment details

---

## Pending Components (Next Phase)

### 2. Networking Hub Instrumentation (`src/DocsToKG/ContentDownload/networking.py`)

**To implement:**
- Just-in-time canonicalization in `request_with_retries()` (already partially there; needs completion)
- Metrics: `urls_normalized_total`, `urls_changed_total`
- Strict mode: `DOCSTOKG_URL_STRICT=1` to reject non-canonical inputs
- Per-host logging (once/host) for param changes
- Role-aware header shaping (Accept, Cache-Control by role)

### 3. Resolver Integration

**To implement across all resolvers:**
- Emit `canonical_url` alongside `original_url` in candidate records
- Forward canonical URL + role to networking hub
- Pass `origin_host` for relative link resolution (landing pages)

### 4. Pipeline & Telemetry Updates

**To implement:**
- Use `canonical_url` as primary key in `ManifestUrlIndex` and dedupe caches
- Persist both `original_url` and `canonical_url` in manifest records for audit
- Update resume hydration to use `canonical_url`
- Verify cache hit-rate improvements post-rollout

### 5. Advanced Features (Optional)

- Per-domain URL policy file (`configs/url_policy.yaml`) with custom allowlists and HTTP/2 denylist
- `is_canonical(url: str) -> (bool, str)` helper for static validation
- Pre-commit hook to forbid raw HTTP calls outside the hub
- DOI normalization helpers (`is_doi()`, `doi_to_url()`)

---

## Key Design Decisions

1. **No reordering of query params** – v2.0+ behavior of url-normalize; order may be semantically meaningful
2. **Explicit role-based filtering** – landing pages drop trackers by default, but artifacts preserve all params
3. **Conservative default tracker list** – excludes auth/CDN signature params to avoid breaking authenticated requests
4. **Per-domain allowlists** – allow fine-grained control without affecting other domains
5. **Frozen policies at initialization** – tests reset between cases; avoids cross-contamination
6. **Triple URL storage in manifest** – `original_url` (audit), `canonical_url` (dedupe), `request_url` (actual sent) for traceability

---

## Testing & Validation

```bash
# Run URL canonicalization tests
pytest tests/content_download/test_urls.py -v

# Check linting
ruff check src/DocsToKG/ContentDownload/urls.py
mypy src/DocsToKG/ContentDownload/urls.py

# Verify imports & basic usage
python -c "from DocsToKG.ContentDownload.urls import canonical_for_index, canonical_host; print(canonical_for_index('HTTPS://EXAMPLE.COM/'))"
```

---

## Migration & Rollout Plan

1. **Phase 1 (current)**: Core URL module + tests ✅
2. **Phase 2 (pending)**: Networking instrumentation + resolver integration
3. **Phase 3 (pending)**: Pipeline/manifest updates + cache hit rate validation
4. **Phase 4 (optional)**: Advanced features (policy files, pre-commit hooks)

**Canary approach:**
- Run with `DOCSTOKG_URL_STRICT=0` (warn-only) to surface param changes
- Monitor cache hit-rate and dedup accuracy before enabling strict mode
- Adjust tracker allowlists based on resolver data

---

## Impact & Benefits

| Metric | Before | Expected After |
|--------|--------|-----------------|
| Cache hit-rate (identical logical requests) | Varies (case-sensitive) | +10-15% (stable keys) |
| Limiter key consistency | Port/case variants split quota | 100% consolidated |
| Resume accuracy | Case mismatches cause re-downloads | 100% accurate dedup |
| Manifest searchability | Multiple representations per URL | Single canonical form |

---

## References

- **RFC 3986**: URI Generic Syntax (normalization rules)
- **RFC 3987**: IRI (internationalized domains)
- **url-normalize** v2.2.1: RFC-compliant Python library
- **HTTPX/Hishel**: Caching and request transport
- **Module docstring** (`urls.py`): Comprehensive policy documentation with gotchas

---

## Status

- ✅ Core URL module (`urls.py`) – complete with comprehensive docstring
- ✅ Test suite (`test_urls.py`) – 22 tests passing
- ✅ README documentation – detailed section added
- ⏳ Networking integration – in progress
- ⏳ Resolver/pipeline updates – pending
- ⏳ Instrumentation & metrics – pending

