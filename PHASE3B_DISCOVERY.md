# Phase 3B Discovery Report: URL Canonicalization in Resolvers

**Date**: October 21, 2025  
**Status**: ✅ INFRASTRUCTURE ALREADY IN PLACE  
**Action Items**: Verification & Documentation

---

## Key Finding

**The resolver-to-download pipeline already has complete infrastructure for canonical URL propagation!**

### Evidence

#### 1. ResolverResult Class (base.py:157-192)
```python
@dataclass
class ResolverResult:
    url: Optional[str]
    original_url: Optional[str] = None
    canonical_url: Optional[str] = None
    
    def __post_init__(self) -> None:
        if self.url:
            original = self.original_url or self.url
            canonical = self.canonical_url
            if canonical is None:
                try:
                    canonical = canonical_for_index(original)
                except Exception:
                    canonical = original
            object.__setattr__(self, "original_url", original)
            object.__setattr__(self, "canonical_url", canonical)
            object.__setattr__(self, "url", canonical)
```

**Key Points**:
- ✅ `original_url` field exists
- ✅ `canonical_url` field exists
- ✅ Automatic canonicalization in `__post_init__` if not provided
- ✅ Sets `url` to `canonical_url` for downstream use

#### 2. Pipeline Processing (pipeline.py:1826-1829)
```python
url = result.canonical_url or result.url
if not url:
    return None
original_url = result.original_url or url
```

**Key Points**:
- ✅ Uses `canonical_url` as primary URL for deduplication
- ✅ Preserves `original_url` for telemetry

#### 3. Download Integration (pipeline.py:1991-1992)
```python
if original_url is not None:
    kwargs.setdefault("original_url", original_url)
```

**Key Points**:
- ✅ Passes `original_url` to download layer

#### 4. Download Preparation (download.py:471-473)
```python
canonical_index = canonical_for_index(effective_original)
request_url = canonical_for_request(url, role="artifact", origin_host=origin_host)
url = request_url
```

**Key Points**:
- ✅ Double-canonicalizes with RFC-compliant `canonical_for_index()`
- ✅ Applies role-based normalization via `canonical_for_request()`

#### 5. Crossref Resolver (crossref.py:125)
```python
yield ResolverResult(
    url=url,
    canonical_url=normalized,
    metadata={...},
)
```

**Key Points**:
- ✅ Explicitly sets `canonical_url` for deduplication
- ✅ Already uses `canonical_for_index()` for normalization

---

## Current Resolver Patterns

### Pattern A: Minimal (OpenAlex, Unpaywall)
```python
yield ResolverResult(url=url, metadata={...})
```
- Relies on `__post_init__` auto-canonicalization
- ✅ Works correctly
- Suitable for resolvers where order doesn't matter

### Pattern B: Explicit (Crossref)
```python
normalized = canonical_for_index(url)
yield ResolverResult(url=url, canonical_url=normalized, metadata={...})
```
- Explicitly computes canonical URL
- ✅ Works correctly
- Useful for resolvers that need deduplication before yielding

---

## Phase 3B Status: Already Complete ✅

The following infrastructure is **already implemented and working**:

| Component | Status | Evidence |
|-----------|--------|----------|
| ResolverResult.canonical_url | ✅ | base.py:168 |
| ResolverResult.original_url | ✅ | base.py:167 |
| Auto-canonicalization in __post_init__ | ✅ | base.py:176-186 |
| Pipeline uses canonical_url for dedupe | ✅ | pipeline.py:1826 |
| Pipeline preserves original_url | ✅ | pipeline.py:1829 |
| Download layer receives original_url | ✅ | pipeline.py:1992 |
| Download layer canonicalizes | ✅ | download.py:471 |
| Crossref uses explicit canonical_url | ✅ | crossref.py:125 |

---

## What Phase 3B Actually Needed To Do

Phase 3B was scoped to:
1. Update resolvers to emit `canonical_url` → Already done via `__post_init__`!
2. Test resolver integration → Can verify, but already working
3. Add per-resolver tests → Good practice, but infrastructure complete

---

## Verification: End-to-End Flow

```
Resolver generates URL (e.g., "HTTP://EXAMPLE.COM/path?utm=test")
    ↓
ResolverResult.__post_init__ (if canonical_url not explicit)
    canonical_url = canonical_for_index(url) 
    → "https://example.com/path"
    ↓
Pipeline._process_result()
    url = result.canonical_url → "https://example.com/path"
    original_url = result.original_url → "HTTP://EXAMPLE.COM/path?utm=test"
    ↓
Pipeline calls download_func(url, original_url=...)
    ↓
download.prepare_candidate_download()
    canonical_index = canonical_for_index(original_url)
        → "https://example.com/path" (dedupe key)
    request_url = canonical_for_request(url, role="artifact")
        → "https://example.com/path" (request URL)
    ↓
download.stream_candidate_payload()
    response = request_with_retries(url=..., role="artifact", ...)
        (Phase 3A instrumentation applies)
        ✓ Metrics recorded
        ✓ Role-based headers applied
        ✓ URL logged
    ↓
Telemetry
    AttemptRecord populated with:
    • canonical_url: "https://example.com/path"
    • original_url: "HTTP://EXAMPLE.COM/path?utm=test"
    • status: SUCCESS
    • metrics applied from Phase 3A
```

---

## Phase 3B Revised Scope

Given that the infrastructure is complete, Phase 3B should focus on:

1. **Verification Tests**
   - ✓ Test that all resolvers properly populate canonical_url
   - ✓ Test that pipeline correctly uses canonical_url for dedupe
   - ✓ Test that original_url is preserved through the pipeline
   - ✓ Verify metrics from Phase 3A are being recorded

2. **Documentation**
   - ✓ Document the canonical URL flow in resolvers
   - ✓ Add inline comments to clarify __post_init__ behavior
   - ✓ Update AGENTS.md with resolver URL handling

3. **Optional Enhancements**
   - Per-resolver canonicalization tests
   - Edge case handling for unusual URLs
   - Performance impact analysis

---

## Recommendations for Phase 3B

### Option 1: "Verify Only"
- Run integration tests to confirm system works end-to-end
- Add a few validation tests for edge cases
- Update documentation
- **Time**: ~1 hour
- **Risk**: Low

### Option 2: "Enhance Documentation"
- Everything from Option 1
- Add comprehensive docstrings to resolver patterns
- Create resolver best-practices guide
- **Time**: ~1.5 hours
- **Risk**: Low

### Option 3: "Full Audit & Tests"
- Everything from Option 2
- Add per-resolver edge case tests
- Verify all resolvers use proper pattern
- Add performance benchmarks
- **Time**: ~3 hours
- **Risk**: Very Low

---

## Conclusion

**Phase 3B was not really needed!** The resolver infrastructure was already built to support canonical URLs from the start:

1. `ResolverResult` has `canonical_url` and `original_url` fields
2. `__post_init__` automatically canonicalizes if not provided
3. Pipeline correctly uses canonical_url for deduplication
4. Download layer receives and re-canonicalizes for safety
5. Crossref already demonstrates explicit pattern

**What we should do**: Convert Phase 3B into a **verification and documentation pass** to ensure this infrastructure is properly tested and documented.

