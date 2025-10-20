# Phase 3 Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 1: Core Module                        │
│  urls.py (RFC 3986 canonicalization + role-based policies)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Phase 2: Instrumentation                        │
│  urls_networking.py (metrics, strict-mode, header-shaping)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌─────▼──────────┐
│ Networking   │  │ Resolvers   │  │ Pipeline &     │
│ Hub (Phase   │  │ (Phase 3B)  │  │ Manifest       │
│ 3A)          │  │             │  │ (Phase 3C)     │
│              │  │             │  │                │
│ Wire instr.  │  │ Emit        │  │ Dedupe by      │
│ calls into   │  │ canonical   │  │ canonical_url  │
│ request_w_r  │  │ URLs        │  │                │
└──────────────┘  └─────────────┘  └────────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                    ✅ All requests
                    use canonical URLs
                    ✅ Metrics tracked
                    ✅ Cache hits +10-15%
```

## Integration Order (Sequential Dependencies)

### Step 1: Networking Hub (Phase 3A) ✅ FOUNDATION
- Wire `urls_networking` calls into `request_with_retries()`
- Verify metrics collection
- Validate strict mode enforcement
- **Blockers**: None (Phase 2 complete)
- **Duration**: ~2 hours
- **Tests**: 5-10 new integration tests

### Step 2: Pipeline Updates (Phase 3C) ✅ NEXT
- Update `ManifestUrlIndex` to use canonical URLs
- Modify dedupe logic in `download.process_one_work()`
- Update telemetry to track both original and canonical
- **Blockers**: Step 1 complete (needs networking integration)
- **Duration**: ~2 hours
- **Tests**: 5-8 dedupe/pipeline tests

### Step 3: Resolver Integration (Phase 3B) ✅ FOLLOW-UP
- Update resolvers to emit canonical_url (start with 3-5)
- Verify backward compatibility
- Update all remaining resolvers
- **Blockers**: Steps 1-2 complete (foundation in place)
- **Duration**: ~3 hours (distributed across resolvers)
- **Tests**: Per-resolver validation

### Step 4: Validation & Monitoring (Phase 3D) ✅ FINAL
- Run integration test suite
- Deploy to canary
- Monitor metrics improvements
- Validate cache hit-rate gains
- **Blockers**: Steps 1-3 complete
- **Duration**: ~1-2 hours
- **Tests**: End-to-end integration suite

## Key Integration Points

### Networking Hub (networking.py)

**Location**: Lines 800-820 (after canonicalization, before request execution)

```python
# After: canonical_index = canonical_for_index(source_url)
# Before: http_client.request(method, request_url, ...)

# ADD:
from DocsToKG.ContentDownload.urls_networking import (
    record_url_normalization,
    log_url_change_once,
    apply_role_headers,
)

record_url_normalization(source_url, request_url, url_role)
kwargs["headers"] = apply_role_headers(kwargs.get("headers"), url_role)
if request_url != source_url:
    log_url_change_once(source_url, request_url, host_hint)
```

### Pipeline (download.py, pipeline.py)

**Location**: `process_one_work()` function (around line 150-200)

```python
# ADD at start of function:
from DocsToKG.ContentDownload.urls import canonical_for_index

canonical_url = canonical_for_index(work.url)

# In dedupe check:
if self.url_index.is_downloaded(canonical_url):
    return skip_result()
```

### Resolvers (resolvers/*.py)

**Location**: Each resolver's candidate emission (varies per resolver)

```python
# ADD import
from DocsToKG.ContentDownload.urls import canonical_for_index

# ADD in candidate creation
canonical_url = canonical_for_index(original_url)
candidate = Candidate(..., url=canonical_url, ...)
```

## Rollout Strategy

### Phase 3A: Networking (First - Critical Path)
1. Add imports to networking.py
2. Wire instrumentation calls
3. Add 5-10 unit tests
4. Run existing test suite (no regressions)
5. **Verify**: Metrics collection working

### Phase 3C: Pipeline (Second - Depends on 3A)
1. Update ManifestUrlIndex
2. Update process_one_work() dedupe logic
3. Add telemetry tracking
4. Add 5-8 integration tests
5. **Verify**: Dedupe working with canonical URLs

### Phase 3B: Resolvers (Third - Can be done in parallel)
1. Update 3 core resolvers first (openalex, unpaywall, crossref)
2. Test with sample runs
3. Update remaining resolvers
4. Add per-resolver tests
5. **Verify**: Canonical URLs propagating through system

### Phase 3D: Validation (Final - All prerequisites complete)
1. Run full test suite
2. Deploy to canary
3. Monitor metrics
4. Validate cache/dedupe improvements
5. **Verify**: 10-15% cache hit-rate improvement

## Backward Compatibility

✅ All changes are **backward compatible**:
- Networking still accepts non-canonical URLs (logs warning)
- Pipeline falls back to original_url if canonical_url unavailable
- Resolvers can emit either format (migration period)
- Metrics are **additive** (no breaking changes)

## Measurement Points

Track these metrics throughout Phase 3:

| Metric | Phase 1 | Phase 3A | Phase 3B | Phase 3C | Phase 3D |
|--------|---------|---------|---------|----------|----------|
| Cache hit-rate | Baseline | +1-2% | +3-5% | +8-12% | +10-15% |
| Dedupe accuracy | 95% | 95% | 97% | 99% | 99.5% |
| Metrics tracked | ❌ | ✅ | ✅ | ✅ | ✅ |
| Strict mode | ❌ | ✅ | ✅ | ✅ | ✅ |
| Canonical URLs | 0% | 5% | 25% | 80% | 100% |

## Risk Mitigation

1. **Integration complexity**: Use feature flags + canary deployment
2. **Resolver changes**: Update 3 at a time, test each batch
3. **Cache invalidation**: Keep original_url for fallback during migration
4. **Metrics accuracy**: Validate against manual sampling

## Success Definition

Phase 3 is complete when:
- ✅ All HTTP requests use canonical URLs
- ✅ Networking instrumentation wired & tested
- ✅ Pipeline dedupe uses canonical URLs
- ✅ Resolvers emit canonical URLs (all)
- ✅ Integration tests passing (100%)
- ✅ Cache hit-rate improved by 10-15%
- ✅ Metrics collection validated
- ✅ Ready for canary deployment

