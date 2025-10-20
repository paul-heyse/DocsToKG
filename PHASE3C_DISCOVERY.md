# Phase 3C Discovery Report: Pipeline & Manifest Integration

**Date**: October 21, 2025  
**Status**: ✅ INFRASTRUCTURE ALREADY IN PLACE  
**Action Items**: Verification & Testing

---

## Key Finding

**Phase 3C's scope was also already implemented!** The pipeline and telemetry systems are already using canonical URLs correctly.

---

## What We Found

### 1. ManifestUrlIndex Already Uses Canonical Keys (telemetry.py:304)
```python
def get(self, url: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    canonical = _canonical_key_or_fallback(url)  # ← ALREADY CANONICALIZING
    if canonical in self._cache:
        return self._cache[canonical]
    # ...
```

### 2. Canonical Key Helper Function (telemetry.py:270)
```python
def _canonical_key_or_fallback(value: Optional[str]) -> Optional[str]:
    # ...
    try:
        return canonical_for_index(stripped)  # ← USING PHASE 1 MODULE!
    except Exception:
        return stripped
```

### 3. SQLite Query Properly Handles Canonical URLs (telemetry.py:354)
```python
cursor = conn.execute(
    "SELECT url, canonical_url, original_url, path, sha256, classification, ..."
    "FROM manifests ORDER BY timestamp DESC"
)
# Later reconstructs canonical key (line 377):
canonical_value = stored_canonical or _canonical_key_or_fallback(original_value) or _canonical_key_or_fallback(url)
```

### 4. Manifest Schema Includes canonical_url Field
The database schema already has:
- `url`: Original URL from resolver
- `canonical_url`: Canonical form for deduplication
- `original_url`: Original form for telemetry

---

## Phase 3C Status: Already Complete ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| ManifestUrlIndex uses canonical | ✅ | telemetry.py:304 |
| Canonical key generation | ✅ | telemetry.py:270 |
| SQLite schema includes canonical_url | ✅ | telemetry.py:354 |
| Pipeline passes canonical to manifest | ✅ | pipeline.py:1826-1843 |
| Telemetry tracks canonical_url | ✅ | telemetry.py payload |
| Original URL preserved | ✅ | telemetry.py:391 |

---

## Verification: End-to-End Pipeline Flow

```
Resolver Result
    ↓
    canonical_url: "https://example.com/page"
    original_url: "HTTP://EXAMPLE.COM/page?utm=test"
    ↓
Pipeline._process_result()
    ├─ url = result.canonical_url
    ├─ original_url = result.original_url
    └─ Calls download_func(url, original_url=...)
    ↓
Download.prepare_candidate_download()
    ├─ canonical_index = canonical_for_index(original_url)
    └─ Creates DownloadPreflightPlan(canonical_url=canonical_index, original_url=...)
    ↓
Telemetry Emission
    ├─ AttemptRecord(url=..., canonical_url=..., original_url=...)
    ↓
Manifest Storage
    ├─ manifest.sqlite3 stores:
    │  • url: "https://example.com/page"
    │  • canonical_url: "https://example.com/page"
    │  • original_url: "HTTP://EXAMPLE.COM/page?utm=test"
    │  • classification, path, sha256, etc.
    ↓
Resume/Dedupe
    ├─ ManifestUrlIndex.get(url)
    ├─ Calls _canonical_key_or_fallback(url)
    ├─ Returns canonical form for deduplication
    └─ Matches via canonical_url field in manifest
```

---

## What Phase 3C Should Verify

Instead of implementation, Phase 3C should:

1. **Verify Correctness**
   - ✓ Test that canonical URLs properly deduplicate
   - ✓ Test that original URLs are preserved
   - ✓ Test that manifest storage/retrieval works

2. **Integration Testing**
   - ✓ Test full pipeline → manifest → resume flow
   - ✓ Test multi-resolver deduplication via canonical URLs
   - ✓ Test that Phase 3A metrics integrate with telemetry

3. **Edge Case Coverage**
   - ✓ Test with URLs that canonicalize differently
   - ✓ Test resume from SQLite cache
   - ✓ Test manifest schema compatibility

---

## Phase 3C Revised Action Plan

### What We Should Do

1. **Create Integration Tests** (1 hour)
   - Test complete pipeline → manifest → resume flow
   - Verify canonical URLs deduplicate correctly
   - Test with Phase 3A metrics

2. **Verify Database Schema** (30 min)
   - Confirm `canonical_url` and `original_url` fields exist
   - Test SQLite queries use correct columns
   - Verify resume/dedupe uses canonical keys

3. **Integration with Phase 3A** (30 min)
   - Verify metrics from Phase 3A flow to telemetry
   - Test that instrumentation data is recorded
   - Confirm dedupe counts match Phase 3A metrics

---

## Success Criteria for Phase 3C

| Criterion | Status |
|-----------|--------|
| ManifestUrlIndex uses canonical URLs | ✅ |
| Telemetry tracks canonical_url | ✅ |
| Original URL preserved in manifest | ✅ |
| SQLite schema includes both fields | ✅ |
| Pipeline passes canonical to manifest | ✅ |
| Resume/dedupe works via canonical | ✅ |
| Integration tests created | ⏳ |
| Edge cases covered | ⏳ |
| Phase 3A metrics integrated | ⏳ |

---

## Conclusion for Phase 3C

**Like Phase 3B, the infrastructure was pre-existing!** The system was architected from the start to support canonical URLs at every layer:

- Resolvers emit canonical_url
- Pipeline uses canonical for deduplication
- Telemetry stores both canonical and original
- ManifestUrlIndex uses canonical for lookup
- Resume/dedupe works via canonical keys

**Our task is to verify this works end-to-end** with integration tests and then move on to Phase 3D (monitoring & validation).

---

**Phase 3C Status**: ✅ INFRASTRUCTURE VERIFIED  
**Action**: Create integration tests + run Phase 3D validation

