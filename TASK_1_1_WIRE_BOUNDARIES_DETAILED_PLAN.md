# 📋 TASK 1.1 - WIRE BOUNDARIES INTO PLANNING.PY - DETAILED IMPLEMENTATION PLAN

**Date**: October 21, 2025  
**Duration**: ~2.5 hours  
**Priority**: CRITICAL - Highest impact on Phase 1 completion  
**Approach**: Deliberate, robust implementation with comprehensive testing

---

## 0. EXECUTIVE SUMMARY

The DuckDB boundaries (functions that record catalog transactions) are implemented but **not called** from the core download flow in `planning.py`. This is a critical gap because:

- ❌ Downloads are not recorded in the DuckDB catalog
- ❌ Extractions are not tracked
- ❌ Validations are not stored
- ❌ Doctor/prune operations have no data to work with

**Task 1.1 closes this gap** by wiring the 4 boundary functions into the `fetch_one()` function at strategic points:

1. **download_boundary()** - After HTTP download succeeds (line ~1891)
2. **extraction_boundary()** - After archive extraction completes (line ~1951)
3. **validation_boundary()** - After validators complete (line ~1963)
4. **set_latest_boundary()** - When marking as latest successful version (line ~2094)

---

## 1. PREREQUISITES & SETUP

### 1.1 Current State
- ✅ Settings (Task 1.4) complete: `config.defaults.db` and `config.defaults.storage` available
- ✅ Boundary functions exist in `catalog/boundaries.py`
- ✅ `config_hash()` method available for event correlation
- ✅ Planning.py has full infrastructure for catalog operations

### 1.2 Dependencies Already Available
```python
# All these are already in settings.py and accessible:
config.defaults.db.path          # DuckDB file location
config.defaults.db.threads       # Thread count
config.defaults.db.writer_lock   # Writer lock setting
config.defaults.storage.root     # Storage root
config_hash = config.config_hash()  # Configuration fingerprint
```

### 1.3 Boundary Functions Already Exist
```python
# In catalog/boundaries.py:
- download_boundary(cfg: DuckDBConfig, inp: DownloadBoundaryInput) -> None
- extraction_boundary(cfg: DuckDBConfig, version_id: str, files: Iterable[ExtractedFileMeta]) -> int
- validation_boundary(cfg: DuckDBConfig, rows: Iterable[ValidationMeta]) -> int
- set_latest_boundary(cfg: DuckDBConfig, version_id: str, by: str | None = None) -> None
```

---

## 2. DETAILED INTEGRATION POINTS

### 2.1 INTEGRATION POINT #1: download_boundary() - Line ~1891

**Current Code** (simplified):
```python
result = download_stream(
    url=pending_secure_url,
    destination=pending_destination,
    headers=candidate.plan.headers,
    ...
)
```

**After Integration**:
```python
result = download_stream(...)

# ← NEW: Wire download to DuckDB catalog
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import download_boundary, DownloadBoundaryInput
    from DocsToKG.OntologyDownload.catalog.connection import DuckDBConfig
    
    db_cfg = DuckDBConfig(
        path=active_config.defaults.db.path,
        threads=active_config.defaults.db.threads,
        readonly=False,
        writer_lock=active_config.defaults.db.writer_lock
    )
    
    # Map result to boundary input
    boundary_input = DownloadBoundaryInput(
        version_id=pending_version,
        service=effective_spec.resolver,
        artifact=ArtifactMeta(
            artifact_id=result.sha256,
            version_id=pending_version,
            service=effective_spec.resolver,
            source_url=pending_secure_url,
            etag=result.etag,
            last_modified=result.last_modified,
            content_type=result.content_type,
            size_bytes=content_length or 0,
            fs_relpath=str(destination.relative_to(STORAGE.base_path()) if STORAGE.base_path() else destination),
            status="fresh"
        )
    )
    
    download_boundary(db_cfg, boundary_input)
    adapter.info("download recorded in catalog", extra={
        "stage": "catalog",
        "version_id": pending_version,
        "artifact_id": result.sha256
    })
except Exception as e:
    adapter.warning(
        "failed to record download in catalog",
        extra={"error": str(e), "severity": "non-critical"}
    )
    # Don't fail download if catalog recording fails
```

**Rationale**:
- Called AFTER successful download (result object available)
- Has all metadata needed: SHA256, ETags, last_modified, size
- Safe to fail (wrapped in try/except) - download succeeds regardless

---

### 2.2 INTEGRATION POINT #2: extraction_boundary() - Line ~1951

**Current Code** (simplified):
```python
extracted_paths = extract_archive_safe(
    destination,
    extraction_dir,
    logger=adapter,
    ...
)
artifacts.extend(str(path) for path in extracted_paths)
```

**After Integration**:
```python
extracted_paths = extract_archive_safe(...)
artifacts.extend(str(path) for path in extracted_paths)

# ← NEW: Record extracted files in catalog
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import extraction_boundary, ExtractedFileMeta
    
    # Build file records for all extracted items
    file_records = []
    for extracted_path in extracted_paths:
        rel_path = extracted_path.relative_to(extraction_dir) if extracted_path.parent == extraction_dir else extracted_path.name
        file_records.append(ExtractedFileMeta(
            file_id=hashlib.sha256(str(extracted_path).encode()).hexdigest(),
            artifact_id=result.sha256,
            version_id=pending_version,
            relpath_in_version=str(rel_path),
            format=extracted_path.suffix.lower() or "unknown",
            size_bytes=extracted_path.stat().st_size if extracted_path.exists() else 0,
            mtime=datetime.fromtimestamp(extracted_path.stat().st_mtime).isoformat() if extracted_path.exists() else None,
            cas_relpath=None
        ))
    
    db_cfg = DuckDBConfig(
        path=active_config.defaults.db.path,
        threads=active_config.defaults.db.threads,
        readonly=False,
        writer_lock=active_config.defaults.db.writer_lock
    )
    
    count_inserted = extraction_boundary(db_cfg, pending_version, file_records)
    adapter.info("extraction recorded in catalog", extra={
        "stage": "catalog",
        "version_id": pending_version,
        "files_recorded": count_inserted
    })
except Exception as e:
    adapter.warning(
        "failed to record extraction in catalog",
        extra={"error": str(e), "severity": "non-critical"}
    )
    # Don't fail extraction if catalog recording fails
```

**Rationale**:
- Called AFTER successful extraction (all file paths available)
- Records individual files for later reconciliation
- Safe to fail (try/except)
- Bulk operation via extraction_boundary for efficiency

---

### 2.3 INTEGRATION POINT #3: validation_boundary() - Line ~1963

**Current Code** (simplified):
```python
validation_results = run_validators(validation_requests, adapter)
failed_validators = [...]
```

**After Integration**:
```python
validation_results = run_validators(validation_requests, adapter)

# ← NEW: Record validation results in catalog
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import validation_boundary, ValidationMeta
    
    # Build validation records
    validation_records = []
    for file_record in file_records:  # or extracted_paths if not extracted
        for validator_name, validation_result in validation_results.items():
            validation_records.append(ValidationMeta(
                validation_id=hashlib.sha256(
                    f"{file_record.file_id}:{validator_name}".encode()
                ).hexdigest(),
                file_id=file_record.file_id,
                validator=validator_name,
                passed=getattr(validation_result, "ok", False),
                details_json=json.dumps(getattr(validation_result, "details", {})) if getattr(validation_result, "details") else None,
                run_at=datetime.now(timezone.utc).isoformat()
            ))
    
    db_cfg = DuckDBConfig(
        path=active_config.defaults.db.path,
        threads=active_config.defaults.db.threads,
        readonly=False,
        writer_lock=active_config.defaults.db.writer_lock
    )
    
    count_inserted = validation_boundary(db_cfg, validation_records)
    adapter.info("validations recorded in catalog", extra={
        "stage": "catalog",
        "version_id": pending_version,
        "records_recorded": count_inserted
    })
except Exception as e:
    adapter.warning(
        "failed to record validations in catalog",
        extra={"error": str(e), "severity": "non-critical"}
    )
    # Don't fail validations if catalog recording fails
```

**Rationale**:
- Called AFTER validators complete (results available)
- Cross-tabulates files with validators (N:M relationship)
- Safe to fail

---

### 2.4 INTEGRATION POINT #4: set_latest_boundary() - Line ~2094

**Current Code** (simplified):
```python
STORAGE.finalize_version(effective_spec.id, version, base_dir)

adapter.info(
    "fetch complete",
    extra={...}
)

return FetchResult(...)
```

**After Integration**:
```python
STORAGE.finalize_version(effective_spec.id, version, base_dir)

# ← NEW: Mark this version as latest in catalog
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import set_latest_boundary
    
    db_cfg = DuckDBConfig(
        path=active_config.defaults.db.path,
        threads=active_config.defaults.db.threads,
        readonly=False,
        writer_lock=active_config.defaults.db.writer_lock
    )
    
    set_latest_boundary(
        db_cfg,
        version_id=pending_version,
        by=f"{effective_spec.resolver}@{correlation}"
    )
    adapter.info("set as latest in catalog", extra={
        "stage": "catalog",
        "version_id": pending_version,
        "by": f"{effective_spec.resolver}@{correlation}"
    })
except Exception as e:
    adapter.warning(
        "failed to set latest in catalog",
        extra={"error": str(e), "severity": "non-critical"}
    )
    # Don't fail fetch if latest-setting fails

adapter.info("fetch complete", extra={...})
return FetchResult(...)
```

**Rationale**:
- Called AFTER successful completion (right before return)
- Marks this version as the authoritative "latest"
- Used by doctor/prune operations for reconciliation
- Safe to fail

---

## 3. IMPORTS NEEDED

Add to imports section of planning.py:

```python
# Catalog integration (Phase 1.1)
from DocsToKG.OntologyDownload.catalog.boundaries import (
    DownloadBoundaryInput,
    ExtractedFileMeta,
    ValidationMeta,
    download_boundary,
    extraction_boundary,
    validation_boundary,
    set_latest_boundary,
)
from DocsToKG.OntologyDownload.catalog.connection import DuckDBConfig
```

---

## 4. HELPER UTILITIES

Create small inline helpers in planning.py to avoid code duplication:

```python
def _get_duckdb_config(active_config: ResolvedConfig) -> DuckDBConfig:
    """Create DuckDBConfig from resolved settings."""
    return DuckDBConfig(
        path=active_config.defaults.db.path,
        threads=active_config.defaults.db.threads,
        readonly=False,
        writer_lock=active_config.defaults.db.writer_lock
    )

def _safe_call_boundary(
    func,
    adapter: logging.LoggerAdapter,
    stage_name: str,
    **kwargs
) -> bool:
    """Safely call a boundary function with consistent error handling."""
    try:
        result = func(**kwargs)
        adapter.info(f"{stage_name} recorded in catalog", extra={
            "stage": "catalog",
            "boundary": stage_name,
            "result": result
        })
        return True
    except Exception as e:
        adapter.warning(
            f"failed to record {stage_name} in catalog",
            extra={
                "error": str(e),
                "severity": "non-critical",
                "boundary": stage_name
            }
        )
        return False
```

---

## 5. TESTING STRATEGY

### 5.1 Unit Tests (test_planning_boundaries.py)
- Verify each boundary call with mock DuckDB
- Verify correct data transformation (result → BoundaryInput)
- Verify error handling (boundaries fail gracefully)

### 5.2 Integration Tests (test_planning_end_to_end.py)
- Full fetch_one() flow with real DuckDB
- Verify data recorded matches original inputs
- Verify latest pointer is set correctly

### 5.3 Smoke Tests (against test fixtures)
- Small download with extraction and validation
- Verify catalog records everything
- Doctor finds no orphans or inconsistencies

---

## 6. IMPLEMENTATION SEQUENCE

1. **Phase 1 (10 min)**: Add imports and helper utilities
2. **Phase 2 (20 min)**: Wire download_boundary() at line ~1891
3. **Phase 3 (20 min)**: Wire extraction_boundary() at line ~1951
4. **Phase 4 (20 min)**: Wire validation_boundary() at line ~1963
5. **Phase 5 (15 min)**: Wire set_latest_boundary() at line ~2094
6. **Phase 6 (30 min)**: Create integration tests
7. **Phase 7 (15 min)**: Verify syntax, run linting, test smoke tests

**Total: ~2 hours** (with buffer for iteration)

---

## 7. QUALITY GATES

- ✅ All imports work
- ✅ No new linting errors
- ✅ No mypy errors
- ✅ Backward compatible (boundaries are optional)
- ✅ 100% passing tests
- ✅ No breaking changes to planning.py API

---

## 8. ROLLBACK PLAN

If issues arise:
1. Comment out all four boundary calls
2. All existing fetch operations continue to work
3. Catalog remains empty but functional
4. Doctor/prune report "no data" but don't fail

---

## 9. SUCCESS CRITERIA

✅ **Hard metrics**:
- All 4 boundaries called successfully on a complete fetch
- Data correctly recorded in DuckDB
- Doctor finds no inconsistencies
- All tests pass

✅ **Soft metrics**:
- Code is clear and maintainable
- Error handling is robust
- Logging is informative
- Easy to debug if issues arise

---

**Status**: Ready for implementation  
**Risk Level**: LOW - Boundaries are isolated, wrapped in try/except  
**Impact**: HIGH - Enables entire DuckDB catalog functionality

