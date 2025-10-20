# Phase 3: Resolver Integration & Pipeline Updates (IN PROGRESS)

## Objective

Wire URL canonicalization throughout the download pipeline:
1. **Networking Hub Integration** – Add instrumentation calls to `request_with_retries()`
2. **Resolver Integration** – Emit both `original_url` and `canonical_url` in candidates
3. **Pipeline Updates** – Use `canonical_url` as primary key for dedupe/manifest
4. **Validation & Monitoring** – Track improvements in cache hit-rate and dedupe accuracy

## Phase 3A: Networking Hub Integration

**File**: `src/DocsToKG/ContentDownload/networking.py`

**Integration Points** (after line 700 in request_with_retries):

```python
# After canonicalization happens (lines 691-699)
from DocsToKG.ContentDownload.urls_networking import (
    record_url_normalization,
    log_url_change_once,
    apply_role_headers,
    get_strict_mode,
)

# 1. Record metrics & validate strict mode
try:
    record_url_normalization(source_url, request_url, url_role)
except ValueError as e:
    logger.error(f"URL normalization failed in strict mode: {e}")
    raise

# 2. Apply role-based Accept headers
if "headers" not in kwargs:
    kwargs["headers"] = {}
kwargs["headers"] = apply_role_headers(kwargs.get("headers"), url_role)

# 3. Log URL changes once per host (if different from original)
if request_url != source_url:
    log_url_change_once(source_url, request_url, host_hint)

# Store in extensions for downstream access
extensions.setdefault("docs_url_changed", request_url != source_url)
```

## Phase 3B: Resolver Integration

**Files to Update**: `src/DocsToKG/ContentDownload/resolvers/*.py`

**Changes per resolver**:

1. Import canonicalization helpers:
   ```python
   from DocsToKG.ContentDownload.urls import canonical_for_index
   ```

2. When emitting a candidate URL:
   ```python
   original_url = response.json()["landing_page_url"]  # or similar
   canonical_url = canonical_for_index(original_url)
   
   candidate = Candidate(
       work_id=work_id,
       original_url=original_url,  # For audit/debug
       url=canonical_url,          # For request/dedupe
       resolver_name="resolver_name",
       ...
   )
   ```

3. For landing page HTML parsing with relative URLs:
   ```python
   origin_host = "example.com"
   relative_link = "/path/to/pdf"
   canonical = canonical_for_request(
       relative_link, 
       role="artifact",
       origin_host=origin_host
   )
   ```

## Phase 3C: Pipeline & Manifest Updates

**File**: `src/DocsToKG/ContentDownload/pipeline.py`

**Changes**:

1. Update `ManifestUrlIndex` to index by `canonical_url`:
   ```python
   # In resume hydration
   for record in manifest_records:
       canonical = canonical_for_index(record["url"])
       self._successful_urls.add(canonical)
   ```

2. Update `download.process_one_work()` to use canonical:
   ```python
   canonical_url = canonical_for_index(work.url)
   if self.url_index.is_url_downloaded(canonical_url):
       return skip_result()
   ```

3. Update telemetry to track both:
   ```python
   {
       "original_url": original_url,
       "canonical_url": canonical_url,
       "url_changed": original_url != canonical_url,
       ...
   }
   ```

## Phase 3D: Testing & Validation

**New Tests**:
- `tests/content_download/test_networking_integration.py` – Verify instrumentation in request_with_retries
- `tests/content_download/test_resolver_canonicalization.py` – Verify resolver emissions
- `tests/content_download/test_pipeline_dedupe.py` – Verify canonical URL dedupe

**Validation Metrics**:
1. Cache hit-rate improvement (target: +10-15%)
2. Dedupe accuracy (target: 100% on canonical URLs)
3. Networking instrumentation (verify metrics collection)
4. Strict mode validation (canary validation)

## Success Criteria

✅ All HTTP requests use canonical URLs  
✅ Request headers shaped by role  
✅ Resolvers emit canonical_url  
✅ Pipeline uses canonical for dedupe  
✅ Metrics tracked and exposed  
✅ Tests verify all scenarios  
✅ Cache hit-rate improved by 10-15%  
✅ Zero breaking changes  

## Rollout Sequence

1. ✅ Phase 1: Core URLs module
2. ✅ Phase 2: Networking instrumentation
3. 🔄 **Phase 3: Integration**
   - [ ] Wire networking instrumentation
   - [ ] Update resolvers (start with 2-3, then all)
   - [ ] Update pipeline dedupe logic
   - [ ] Create integration tests
   - [ ] Validate metrics collection
   - [ ] Run canary with DOCSTOKG_URL_STRICT=0
4. 📋 Phase 4: Monitoring & Optimization
   - [ ] Deploy to production
   - [ ] Monitor metrics
   - [ ] Adjust policies if needed
   - [ ] Enable strict mode in CI

