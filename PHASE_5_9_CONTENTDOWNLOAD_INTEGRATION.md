# Phase 5.9 → ContentDownload Integration Strategy

**Date**: October 21, 2025  
**Status**: INTEGRATION PLAN  
**Objective**: Integrate OntologyDownload policy gates with ContentDownload module

---

## Integration Points

### 1. URL Validation (networking.py, httpx_transport.py)
**Current**: URLs are validated via existing checks  
**Integration**: Add `policy.url_gate()` before HTTP requests

**Files to Modify**:
- `src/DocsToKG/ContentDownload/networking.py` - HTTP download orchestration
- `src/DocsToKG/ContentDownload/httpx_transport.py` - HTTPX transport configuration

**Changes**:
```python
# Before: HTTP request execution
# After: Validate URL with policy gate
from DocsToKG.OntologyDownload.policy.gates import url_gate

url_gate(url)  # Raises exception if policy rejects
response = client.get(url)
```

### 2. Path Validation (streaming.py, idempotency.py)
**Current**: Paths are created without comprehensive validation  
**Integration**: Add `policy.path_gate()` before file operations

**Files to Modify**:
- `src/DocsToKG/ContentDownload/streaming.py` - File streaming/writing
- `src/DocsToKG/ContentDownload/idempotency.py` - Path-based leasing

**Changes**:
```python
# Before: Direct file operations
# After: Validate paths with policy gate
from DocsToKG.OntologyDownload.policy.gates import path_gate

path_gate(output_path)  # Raises exception if policy rejects
with open(output_path, 'wb') as f:
    f.write(data)
```

### 3. Archive Extraction (core.py if present, or extraction logic)
**Current**: Extracts archives without comprehensive policy enforcement  
**Integration**: Add `policy.extraction_gate()` for archive entries

**Files to Modify**:
- Any file handling archive extraction

**Changes**:
```python
# Before: Direct archive extraction
# After: Validate archive entries with policy gate
from DocsToKG.OntologyDownload.policy.gates import extraction_gate

extraction_gate({
    "type": entry.type,
    "size": entry.size,
    "name": entry.name
})
```

---

## Implementation Sequence

### Phase 1: URL Validation (Immediate)
1. Add policy gate calls to `networking.py`
2. Add error handling for policy rejections
3. Create 10-15 integration tests
4. Run full ContentDownload test suite
5. Commit changes

### Phase 2: Path Validation (Follow-up)
1. Add policy gate calls to `streaming.py`
2. Add path normalization before validation
3. Create 10-15 integration tests
4. Run full ContentDownload test suite
5. Commit changes

### Phase 3: Archive Extraction (Follow-up)
1. Find archive extraction entry points
2. Add policy gate calls
3. Create integration tests
4. Run full test suite
5. Commit changes

---

## Integration Testing Strategy

### Unit Tests
- Mock policy gates to verify call points
- Test both passing and rejecting scenarios
- Verify error handling

### Integration Tests
- Real policy gates with sample URLs/paths
- Test cross-module workflows
- Verify metrics collection

### End-to-End Tests
- Download workflow with policy gates active
- Verify gate statistics are collected
- Check observability events are emitted

---

## Migration Plan

### Phase 1: URL Validation
**Estimated**: 1-2 hours  
- Add gate calls
- Handle exceptions gracefully
- Test thoroughly

### Phase 2: Path Validation  
**Estimated**: 1-2 hours
- Add gate calls
- Test with various platforms
- Verify cross-platform safety

### Phase 3: Archive Extraction
**Estimated**: 1-2 hours
- Locate extraction code
- Add gate calls
- Test with real archives

---

## Rollback Plan

If integration causes issues:

1. **Immediate**: Revert commits
2. **Investigate**: Check gate behavior
3. **Fix**: Adjust gate policies or integration
4. **Re-test**: Comprehensive validation
5. **Re-deploy**: Commit again

---

## Success Criteria

- [ ] All ContentDownload tests pass
- [ ] 50+ new integration tests passing
- [ ] Policy gates called at all entry points
- [ ] Error messages clear and actionable
- [ ] Metrics collected automatically
- [ ] Observability events emitted
- [ ] No performance regression (< 5% slowdown)
- [ ] Cross-platform testing (Windows/Linux/macOS)

---

## Files to Create

None - only modifications to existing files.

## Files to Modify

1. `src/DocsToKG/ContentDownload/networking.py`
2. `src/DocsToKG/ContentDownload/httpx_transport.py`
3. `src/DocsToKG/ContentDownload/streaming.py`
4. `src/DocsToKG/ContentDownload/idempotency.py`

## Files to Create (Tests)

1. `tests/content_download/test_policy_url_integration.py`
2. `tests/content_download/test_policy_path_integration.py`
3. `tests/content_download/test_policy_extraction_integration.py`

---

**Status**: Ready to proceed with Phase 1 (URL Validation)

