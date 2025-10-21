# 📋 TASK 1.1 - WIRE BOUNDARIES - CORRECTED IMPLEMENTATION GUIDE

**Date**: October 21, 2025  
**Status**: Ready for implementation (Option A - thorough approach)  
**Duration**: 3-4 hours with comprehensive testing  

---

## CRITICAL INSIGHT: BOUNDARIES ARE CONTEXT MANAGERS

The boundaries use Python context managers (`@contextlib.contextmanager`), not simple functions. This is crucial:

```python
# CORRECT USAGE:
with download_boundary(conn, artifact_id, version_id, fs_relpath, size, etag) as result:
    # result: DownloadBoundaryResult
    print(f"Inserted: {result.inserted}")

# INCORRECT (will fail):
result = download_boundary(conn, ...)  # ❌ This is wrong!
```

---

## PHASE 1: UNDERSTAND THE FOUR BOUNDARIES

### Boundary 1: `download_boundary()` - Line ~1891
**Signature**:
```python
@contextlib.contextmanager
def download_boundary(
    conn: duckdb.DuckDBPyConnection,
    artifact_id: str,           # sha256 hash of archive
    version_id: str,            # version identifier
    fs_relpath: str,            # relative path on filesystem
    size: int,                  # archive size in bytes
    etag: Optional[str] = None, # HTTP ETag
) -> Generator[DownloadBoundaryResult, None, None]:
    ...
    yield result  # DownloadBoundaryResult
```

**Result Type**:
```python
@dataclass(frozen=True)
class DownloadBoundaryResult:
    artifact_id: str
    version_id: str
    fs_relpath: str
    size: int
    etag: Optional[str]
    inserted: bool  # ← Check this to verify success
```

**How it works**:
1. Caller has already written file to disk (temp → atomic rename done)
2. Context manager begins a transaction
3. Inserts artifact metadata to `artifacts` table
4. Commits transaction on success, rolls back on exception
5. Yields result so caller can verify

**Error handling**: Raises `duckdb.Error` on failure (wraps in try/except)

---

### Boundary 2: `extraction_boundary()` - Line ~1951
**Signature**:
```python
@contextlib.contextmanager
def extraction_boundary(
    conn: duckdb.DuckDBPyConnection,
    artifact_id: str,  # artifact being extracted
) -> Generator[ExtractionBoundaryResult, None, None]:
    ...
    yield result  # ExtractionBoundaryResult - MUTABLE
```

**Result Type** (MUTABLE - caller populates):
```python
@dataclass(frozen=True)  # Actually frozen, but result is yielded then modified
class ExtractionBoundaryResult:
    artifact_id: str
    files_inserted: int  # ← Caller sets this
    total_size: int     # ← Caller sets this
    audit_path: Path    # ← Caller sets this
    inserted: bool      # ← Set by boundary on commit
```

**How it works** (SPECIAL - DIFFERENT from others):
1. Context manager begins transaction
2. Yields result to caller (result.files_inserted = 0 initially)
3. **Caller populates result.files_inserted, result.total_size, result.audit_path**
4. **Caller inserts extracted file rows into DB via appender**
5. On context exit, if files_inserted > 0, commits; else rolls back
6. Runs policy gate `db_boundary_gate` before commit

**Key difference**: Caller does the actual INSERT, boundary just wraps transaction

**Pattern**:
```python
with extraction_boundary(conn, artifact_id) as result:
    # ← Caller extracts files and builds list
    # ← Caller inserts via: conn.appender("extracted_files").append(...)
    result.files_inserted = len(file_list)  # MUTABLE UPDATE
    result.total_size = sum(f.size for f in file_list)
    result.audit_path = Path(...)
    # On exit, boundary commits if files_inserted > 0
```

---

### Boundary 3: `validation_boundary()` - Line ~1963
**Signature**:
```python
@contextlib.contextmanager
def validation_boundary(
    conn: duckdb.DuckDBPyConnection,
    file_id: str,                    # file being validated
    validator: str,                  # validator name (e.g., 'rdflib')
    status: str,                     # 'pass'|'fail'|'timeout'
    details: Optional[dict] = None,  # validation details (JSON)
) -> Generator[ValidationBoundaryResult, None, None]:
    ...
    yield result  # ValidationBoundaryResult
```

**Result Type**:
```python
@dataclass(frozen=True)
class ValidationBoundaryResult:
    file_id: str
    validator: str
    status: str
    inserted: bool  # ← Check this
```

**How it works**:
1. Caller has run validator and collected results
2. Context manager begins transaction
3. Inserts validation record to `validations` table
4. Commits transaction on success
5. Yields result for caller verification

**Key point**: Inserted per validator per file (N:M relationship)

---

### Boundary 4: `set_latest_boundary()` - Line ~2094
**Signature**:
```python
@contextlib.contextmanager
def set_latest_boundary(
    conn: duckdb.DuckDBPyConnection,
    version_id: str,         # version to mark as latest
    latest_json_path: Path,  # path where LATEST.json should live
) -> Generator[SetLatestBoundaryResult, None, None]:
    ...
    yield result  # SetLatestBoundaryResult
```

**Result Type**:
```python
@dataclass(frozen=True)
class SetLatestBoundaryResult:
    version_id: str
    latest_json_path: Path
    pointer_updated: bool  # ← DB row updated
    json_written: bool     # ← LATEST.json written
```

**How it works**:
1. Caller prepares LATEST.json in temp file at `latest_json_path.parent / f"{name}.tmp"`
2. Context manager begins transaction
3. Upserts `latest_pointer` table with version_id
4. Atomically renames temp LATEST.json to final path
5. Commits transaction
6. Yields result with both flags set

**Key point**: Expects temp file already created by caller!

---

## PHASE 2: HELPER UTILITIES

Create these in `planning.py` to reduce duplication:

```python
def _get_duckdb_conn(active_config: ResolvedConfig) -> duckdb.DuckDBPyConnection:
    """Get or create DuckDB writer connection from config."""
    from DocsToKG.OntologyDownload.catalog.connection import get_writer, DuckDBConfig
    
    db_cfg = DuckDBConfig(
        path=active_config.defaults.db.path,
        threads=active_config.defaults.db.threads,
        readonly=False,
        writer_lock=active_config.defaults.db.writer_lock
    )
    return get_writer(db_cfg)


def _safe_record_boundary(
    adapter: logging.LoggerAdapter,
    boundary_name: str,
    boundary_fn,
    *args,
    **kwargs
) -> Optional[tuple]:
    """Safely call a boundary context manager with error handling.
    
    Returns: (success: bool, result: Any)
    """
    try:
        with boundary_fn(*args, **kwargs) as result:
            adapter.info(f"{boundary_name} recorded in catalog", extra={
                "stage": "catalog",
                "boundary": boundary_name,
            })
            return True, result
    except Exception as e:
        adapter.warning(
            f"failed to record {boundary_name} in catalog",
            extra={
                "stage": "catalog",
                "boundary": boundary_name,
                "error": str(e),
                "severity": "non-critical"
            }
        )
        return False, None
```

---

## PHASE 3: INTEGRATION POINTS (CORRECTED)

### Point 1: download_boundary() @ Line ~1891

**Location in planning.py**:
```python
# Around line 1878-1891
result = download_stream(
    url=pending_secure_url,
    destination=pending_destination,
    headers=candidate.plan.headers,
    ...
)

# ← INSERT BOUNDARY CALL HERE

if expected_checksum:
    attempt_record["expected_checksum"] = expected_checksum.to_mapping()
```

**Code to insert**:
```python
# Record download in catalog (non-critical if fails)
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import download_boundary
    
    conn = _get_duckdb_conn(active_config)
    
    # Compute relative path from storage root
    try:
        fs_relpath = str(destination.relative_to(STORAGE.base_path() or destination.parent))
    except (ValueError, AttributeError):
        fs_relpath = destination.name
    
    success, _ = _safe_record_boundary(
        adapter,
        "download",
        download_boundary,
        conn,
        artifact_id=result.sha256,
        version_id=pending_version,
        fs_relpath=fs_relpath,
        size=result.content_length or 0,
        etag=result.etag
    )
except Exception as e:
    adapter.debug(f"Skipping download boundary: {e}")
```

---

### Point 2: extraction_boundary() @ Line ~1951

**Location in planning.py**:
```python
# Around line 1945-1952
extracted_paths = extract_archive_safe(
    destination,
    extraction_dir,
    logger=adapter,
    max_uncompressed_bytes=active_config.defaults.http.max_uncompressed_bytes(),
)
artifacts.extend(str(path) for path in extracted_paths)

# ← INSERT BOUNDARY CALL HERE

except ConfigError as exc:
```

**Code to insert**:
```python
# Record extraction in catalog (non-critical if fails)
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import extraction_boundary
    
    conn = _get_duckdb_conn(active_config)
    
    with extraction_boundary(conn, result.sha256) as ex_result:
        # Insert extracted files into DB
        if extraction_dir and extraction_dir.exists():
            app = conn.appender("extracted_files")
            for extracted_path in extracted_paths:
                try:
                    rel_path = extracted_path.relative_to(extraction_dir)
                except ValueError:
                    rel_path = extracted_path.name
                
                st = extracted_path.stat()
                app.append([
                    # file_id, artifact_id, version_id, relpath_in_version, 
                    # format, size_bytes, mtime, cas_relpath
                    f"{result.sha256}:{rel_path}",
                    result.sha256,
                    pending_version,
                    str(rel_path),
                    extracted_path.suffix.lower() or "unknown",
                    st.st_size,
                    datetime.fromtimestamp(st.st_mtime).isoformat(),
                    None
                ])
                ex_result.files_inserted += 1
                ex_result.total_size += st.st_size
            
            app.close()
            ex_result.audit_path = extraction_dir / "extraction_audit.json"
    
    adapter.info("extraction recorded in catalog", extra={
        "stage": "catalog",
        "files": ex_result.files_inserted
    })
except Exception as e:
    adapter.debug(f"Skipping extraction boundary: {e}")
```

---

### Point 3: validation_boundary() @ Line ~1963

**Location in planning.py**:
```python
# Around line 1963-1968
validation_results = run_validators(validation_requests, adapter)
failed_validators = [
    name
    for name, result in validation_results.items()
    if not getattr(result, "ok", False)
]

# ← INSERT BOUNDARY CALL HERE

if failed_validators:
```

**Code to insert**:
```python
# Record validations in catalog (non-critical if fails)
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import validation_boundary
    
    conn = _get_duckdb_conn(active_config)
    
    # For each extracted file and validator, record validation result
    for extracted_path in (extracted_paths if extraction_dir and extraction_dir.exists() else [destination]):
        try:
            file_id = f"{result.sha256}:{extracted_path.relative_to(extraction_dir)}"
        except (ValueError, AttributeError):
            file_id = f"{result.sha256}:{extracted_path.name}"
        
        for validator_name, validation_result in validation_results.items():
            try:
                status = "pass" if getattr(validation_result, "ok", False) else "fail"
                details = getattr(validation_result, "details", None)
                
                success, _ = _safe_record_boundary(
                    adapter,
                    f"validation({validator_name})",
                    validation_boundary,
                    conn,
                    file_id=file_id,
                    validator=validator_name,
                    status=status,
                    details=details if isinstance(details, dict) else None
                )
            except Exception as e:
                adapter.debug(f"Skipping validation boundary for {validator_name}: {e}")
except Exception as e:
    adapter.debug(f"Skipping validation boundary: {e}")
```

---

### Point 4: set_latest_boundary() @ Line ~2094

**Location in planning.py**:
```python
# Around line 2094-2095
STORAGE.finalize_version(effective_spec.id, version, base_dir)

# ← INSERT BOUNDARY CALL HERE

adapter.info(
    "fetch complete",
    extra={...}
)
```

**Code to insert**:
```python
# Mark as latest in catalog (non-critical if fails)
try:
    from DocsToKG.OntologyDownload.catalog.boundaries import set_latest_boundary
    import tempfile
    
    conn = _get_duckdb_conn(active_config)
    
    # Prepare LATEST.json in temp file
    latest_json_path = base_dir.parent / "LATEST.json"
    latest_temp_path = latest_json_path.with_suffix(".json.tmp")
    latest_temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    latest_data = {
        "version": version,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sha256": result.sha256,
        "resolver": effective_spec.resolver,
        "correlation_id": correlation,
    }
    
    with open(latest_temp_path, "w") as f:
        json.dump(latest_data, f, indent=2)
    
    success, _ = _safe_record_boundary(
        adapter,
        "set_latest",
        set_latest_boundary,
        conn,
        version_id=version,
        latest_json_path=latest_json_path
    )
except Exception as e:
    adapter.debug(f"Skipping set_latest boundary: {e}")
```

---

## PHASE 4: IMPORTS REQUIRED

Add to top of `planning.py`:

```python
import contextlib
import json
from datetime import datetime, timezone
from uuid import uuid4

# Catalog boundaries (Task 1.1)
try:
    import duckdb
    from DocsToKG.OntologyDownload.catalog.boundaries import (
        download_boundary,
        extraction_boundary,
        validation_boundary,
        set_latest_boundary,
    )
    CATALOG_AVAILABLE = True
except ImportError:
    CATALOG_AVAILABLE = False
```

---

## PHASE 5: TESTING STRATEGY

### Unit Tests
```python
# tests/ontology_download/test_planning_boundaries.py
def test_download_boundary_integration():
    """Verify download_boundary called correctly after stream"""
    
def test_extraction_boundary_integration():
    """Verify extraction_boundary called and files recorded"""
    
def test_validation_boundary_integration():
    """Verify validation_boundary called for each validator"""
    
def test_set_latest_boundary_integration():
    """Verify set_latest_boundary called on completion"""
```

### Integration Tests
```python
# tests/ontology_download/test_planning_end_to_end.py
def test_fetch_one_records_to_catalog():
    """Full fetch flow with catalog recording"""
    
def test_fetch_one_extraction_in_catalog():
    """Extract + record in catalog"""
    
def test_fetch_one_validation_in_catalog():
    """Validate + record in catalog"""
```

### Smoke Tests
```bash
# With real DuckDB and small test ontology:
python -m DocsToKG.OntologyDownload.cli pull hp --max 1 --dry-run
# Then verify catalog populated
```

---

## PHASE 6: IMPLEMENTATION SEQUENCE

1. **Hour 1**: Add imports and helper utilities
2. **Hour 1-2**: Wire download_boundary() 
3. **Hour 2**: Wire extraction_boundary()
4. **Hour 2-3**: Wire validation_boundary()
5. **Hour 3**: Wire set_latest_boundary()
6. **Hour 3-4**: Create comprehensive tests
7. **Hour 4**: Verify, lint, run smoke tests

---

## KEY TAKEAWAYS

✅ **Context managers** - Use `with` statement  
✅ **Connection required** - Must pass DuckDB connection  
✅ **Transaction safety** - Boundaries handle BEGIN/COMMIT/ROLLBACK  
✅ **Non-critical** - Wrap all in try/except, never fail fetch  
✅ **Extraction is special** - Caller populates mutable result  
✅ **Error handling** - Use adapter.debug/info/warning for logging  

---

**Status**: Ready for implementation Phase 1  
**Risk**: LOW - Fully wrapped, non-blocking  
**Impact**: HIGH - Enables catalog functionality

