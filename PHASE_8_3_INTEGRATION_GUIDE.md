# Phase 8.3: Gate Integration into Core Flows - IMPLEMENTATION GUIDE

**Status**: ✅ READY FOR INTEGRATION  
**Date**: October 21, 2025  
**Scope**: Wire all 6 gates into 4 core OntologyDownload flows

---

## Integration Points Overview

All 6 gates are now ready to be wired into the core OntologyDownload workflow:

```
┌─────────────────────────────────────────────────────────┐
│ CLI Entry (fetch_one / plan_one)                        │
│                                                          │
│  ▼                                                        │
│ [config_gate] ← Validate settings on startup             │
│  ▼                                                        │
│ planning._resolve_plan_with_fallback()                   │
│  ▼                                                        │
│ [url_gate] ← Per-URL validation (pre-request)            │
│  ▼                                                        │
│ download_stream()                                        │
│  ▼                                                        │
│ extraction & [extraction_gate]                           │
│  ▼                                                        │
│ [filesystem_gate] ← Path validation during extract      │
│  ▼                                                        │
│ manifests._write_manifest()                              │
│  ▼                                                        │
│ [db_boundary_gate] ← TX choreography post-extract       │
│  ▼                                                        │
│ Result returned                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Integration Pattern

### Pattern 1: Gate Result Handling

```python
from DocsToKG.OntologyDownload.policy.gates import <gate_name>

# Call gate with appropriate arguments
result = <gate_name>(...)

# Check result
if isinstance(result, PolicyReject):
    # Gate rejected - log event, raise appropriate exception
    adapter.error("gate rejected", extra={"gate": "<gate_name>", "error_code": result.error_code})
    raise <DomainException>(f"Gate rejected: {result.error_code}")

# Gate passed - continue
adapter.debug("gate passed", extra={"gate": "<gate_name>", "elapsed_ms": result.elapsed_ms})
```

### Pattern 2: Safe Integration (Fallback)

```python
try:
    from DocsToKG.OntologyDownload.policy.gates import <gate_name>
    result = <gate_name>(...)
    if isinstance(result, PolicyReject):
        raise <DomainException>(...)
except ImportError:
    # Gate module not available - continue without gate (dev/testing)
    adapter.warning("gate not available, skipping", extra={"gate": "<gate_name>"})
```

---

## Integration Location 1: `planning.py` - Configuration Gate

### Where: `fetch_one()` function, line 1746

**Current Code**:
```python
def fetch_one(spec, *, config=None, ...):
    ensure_python_version()
    active_config = config or get_default_config(copy=True)
    # ... setup logging ...
    adapter.info("planning fetch", extra={"stage": "plan"})
```

**Integration**:
```python
    adapter.info("planning fetch", extra={"stage": "plan"})
    
    # GATE 1: Validate configuration
    from DocsToKG.OntologyDownload.policy.gates import config_gate
    config_result = config_gate(active_config)
    if isinstance(config_result, PolicyReject):
        adapter.error("config gate rejected", extra={"stage": "plan", "error_code": config_result.error_code})
        raise ConfigError(f"Configuration validation failed: {config_result.error_code}")
    
    adapter.debug(f"config gate passed ({config_result.elapsed_ms:.2f}ms)")
```

**Status**: ✅ Already integrated (added in Phase 8.3 start)

---

## Integration Location 2: `planning.py` - URL Gate

### Where: `_populate_plan_metadata()` function, line 1159

**Current Code**:
```python
try:
    secure_url = validate_url_security(planned.plan.url, http_config)
except (ConfigError, PolicyError) as exc:
    adapter.error("metadata probe blocked by URL policy", ...)
    raise
```

**Integration**:
```python
try:
    secure_url = validate_url_security(planned.plan.url, http_config)
    
    # GATE 2: Validate URL against network policy
    from DocsToKG.OntologyDownload.policy.gates import url_gate
    url_result = url_gate(
        secure_url,
        allowed_hosts=http_config.allowed_hosts,
        allowed_ports=http_config.allowed_ports if hasattr(http_config, 'allowed_ports') else None
    )
    if isinstance(url_result, PolicyReject):
        adapter.error("url gate rejected", extra={"url": secure_url, "error_code": url_result.error_code})
        raise PolicyError(f"URL policy violation: {url_result.error_code}")
    
except (ConfigError, PolicyError) as exc:
    adapter.error("metadata probe blocked by URL policy", ...)
    raise
```

**Implementation Status**: Pending (requires `validate_url_security` integration)

---

## Integration Location 3: `io/extraction_policy.py` - Extraction Gate

### Where: Pre-scan validation before archive extraction

**Current Pattern**:
```python
def extract_archive(archive_path, destination, max_ratio=100.0, ...):
    # Validation logic here
    with zipfile.ZipFile(archive_path) as zf:
        # Process entries
```

**Integration**:
```python
def extract_archive(archive_path, destination, max_ratio=100.0, ...):
    # GATE 3: Validate extraction parameters (zip bomb detection)
    from DocsToKG.OntologyDownload.policy.gates import extraction_gate
    
    # Get archive stats before extraction
    with zipfile.ZipFile(archive_path) as zf:
        entries_total = len(zf.filelist)
        bytes_declared = sum(info.file_size for info in zf.filelist)
    
    # Gate validation
    extraction_result = extraction_gate(
        entries_total=entries_total,
        bytes_declared=bytes_declared,
        max_total_ratio=max_ratio,
    )
    if isinstance(extraction_result, PolicyReject):
        logger.error(f"extraction gate rejected: {extraction_result.error_code}")
        raise ExtractionError(f"Archive policy violation: {extraction_result.error_code}")
    
    logger.debug(f"extraction gate passed ({extraction_result.elapsed_ms:.2f}ms)")
    
    # Proceed with extraction
    with zipfile.ZipFile(archive_path) as zf:
        # Process entries
```

**Implementation Status**: Pending

---

## Integration Location 4: `io/filesystem.py` - Filesystem Gate

### Where: Before extracting entries to disk

**Current Pattern**:
```python
def extract_entries(archive_path, destination, entries, ...):
    for entry in entries:
        # Write entry to disk
```

**Integration**:
```python
def extract_entries(archive_path, destination, entries, ...):
    # GATE 4: Validate filesystem paths (traversal prevention)
    from DocsToKG.OntologyDownload.policy.gates import filesystem_gate
    
    # Collect entry paths
    entry_paths = [entry.filename for entry in entries]
    
    # Gate validation
    fs_result = filesystem_gate(
        root_path=str(destination),
        entry_paths=entry_paths,
        allow_symlinks=False,
    )
    if isinstance(fs_result, PolicyReject):
        logger.error(f"filesystem gate rejected: {fs_result.error_code}")
        raise IOError(f"Filesystem policy violation: {fs_result.error_code}")
    
    logger.debug(f"filesystem gate passed ({fs_result.elapsed_ms:.2f}ms)")
    
    # Proceed with extraction
    for entry in entries:
        # Write entry to disk
```

**Implementation Status**: Pending

---

## Integration Location 5: `catalog/boundaries.py` - DB Boundary Gate

### Where: Before committing extracted files to database

**Current Pattern**:
```python
def commit_manifest(manifest, connection):
    # Insert manifest records
    connection.commit()
```

**Integration**:
```python
def commit_manifest(manifest, connection, fs_success=True):
    # GATE 6: Validate transaction boundaries (no torn writes)
    from DocsToKG.OntologyDownload.policy.gates import db_boundary_gate
    
    # Gate validation
    db_result = db_boundary_gate(
        operation="pre_commit",
        tables_affected=["extracted_files", "manifests", "versions"],
        fs_success=fs_success,
    )
    if isinstance(db_result, PolicyReject):
        logger.error(f"db_boundary gate rejected: {db_result.error_code}")
        connection.rollback()
        raise DBError(f"Transaction boundary violation: {db_result.error_code}")
    
    logger.debug(f"db_boundary gate passed ({db_result.elapsed_ms:.2f}ms)")
    
    # Proceed with commit
    connection.commit()
```

**Implementation Status**: Pending

---

## Storage Gate Integration (Optional)

### Where: In `settings.STORAGE.mirror_cas_artifact()` or similar

**Integration Pattern**:
```python
def mirror_cas_artifact(src, dst, operation="move"):
    # GATE 5: Validate storage operation
    from DocsToKG.OntologyDownload.policy.gates import storage_gate
    
    storage_result = storage_gate(
        operation=operation,
        src_path=src,
        dst_path=dst,
        check_traversal=True,
    )
    if isinstance(storage_result, PolicyReject):
        logger.error(f"storage gate rejected: {storage_result.error_code}")
        raise StorageError(f"Storage policy violation: {storage_result.error_code}")
    
    logger.debug(f"storage gate passed ({storage_result.elapsed_ms:.2f}ms)")
    
    # Proceed with operation
    if operation == "move":
        shutil.move(src, dst)
    elif operation == "copy":
        shutil.copy(src, dst)
```

**Implementation Status**: Optional (if CAS mirroring is used)

---

## Testing Strategy for Integration

### Unit Tests (per gate integration point)

```python
def test_fetch_one_config_gate_validates_on_startup():
    """Config gate validates config at fetch_one start."""
    # Create invalid config
    # Call fetch_one
    # Assert config_gate was called
    # Assert ConfigError raised

def test_url_gate_validates_urls():
    """URL gate validates each URL before download."""
    # Mock url_gate to reject
    # Call fetch_one with disallowed host
    # Assert rejection logged
    # Assert PolicyError raised

def test_filesystem_gate_prevents_traversal():
    """Filesystem gate prevents path traversal during extraction."""
    # Create archive with ../../../ paths
    # Call extract_entries
    # Assert filesystem_gate called
    # Assert IOError raised
```

### Integration Tests

```python
def test_gates_integrated_e2e():
    """All gates integrated in full fetch pipeline."""
    # E2E scenario with multiple gates
    # Verify events emitted
    # Verify metrics recorded
```

---

## Error Handling Pattern

When a gate rejects, follow this pattern:

1. **Log the rejection** with gate name and error code
2. **Emit event** (automatic via gate telemetry)
3. **Record metric** (automatic via gate telemetry)
4. **Raise domain-specific exception**:
   - `ConfigError` for config_gate
   - `PolicyError` or `URLPolicyException` for url_gate
   - `ExtractionError` for extraction_gate
   - `IOError` or `FilesystemPolicyException` for filesystem_gate
   - `StorageError` or `StoragePolicyException` for storage_gate
   - `DBError` or `DbBoundaryException` for db_boundary_gate

---

## Import Organization

For each file, add imports at the top:

```python
# In planning.py
from DocsToKG.OntologyDownload.policy.gates import (
    config_gate,
    url_gate,
)
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

# In io/filesystem.py
from DocsToKG.OntologyDownload.policy.gates import filesystem_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

# In extraction_policy.py
from DocsToKG.OntologyDownload.policy.gates import extraction_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

# In catalog/boundaries.py
from DocsToKG.OntologyDownload.policy.gates import db_boundary_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject
```

---

## Implementation Checklist

- [x] Planning.py config_gate integration (done)
- [ ] Planning.py url_gate integration
- [ ] extraction_policy.py extraction_gate integration
- [ ] io/filesystem.py filesystem_gate integration
- [ ] catalog/boundaries.py db_boundary_gate integration
- [ ] storage operations storage_gate integration (optional)
- [ ] Unit tests for each integration point
- [ ] Integration tests (E2E scenarios)
- [ ] Error handling verification
- [ ] Event emission validation
- [ ] Metrics recording validation

---

## Phase 8.4: Testing (Follow-up)

After integration, implement comprehensive tests:

1. **Per-gate unit tests** - Verify gate called with correct args
2. **Rejection tests** - Verify proper error handling
3. **Event tests** - Verify events emitted
4. **Metrics tests** - Verify metrics recorded
5. **E2E tests** - Full pipeline with gates
6. **Cross-platform** - Windows, macOS path handling
7. **Chaos** - Crash recovery, state consistency

---

## Summary

**Phase 8.3 Ready**: All 6 gates are implemented, instrumented with telemetry, and ready for integration into core flows.

**Integration Points**: 5 critical points + 1 optional (storage)

**Next Steps**:
1. Wire config_gate into fetch_one (✅ done)
2. Wire url_gate into _populate_plan_metadata
3. Wire extraction_gate into extract_archive
4. Wire filesystem_gate into extract_entries
5. Wire db_boundary_gate into commit_manifest
6. Create comprehensive test suite

**Estimated Time**: 3-4 hours for full integration + testing
