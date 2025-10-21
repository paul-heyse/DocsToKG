# 🚀 DuckDB Phase 1 Implementation Plan

**Date**: October 21, 2025  
**Duration**: Week 1, ~8 hours  
**Goal**: Close all CRITICAL + HIGH priority gaps

---

## CRITICAL FINDING: Boundaries NOT Called in planning.py

**Audit Result**: `grep -r "download_boundary|extraction_boundary|validation_boundary|set_latest"` returned **ZERO matches** in planning.py

**Impact**: The entire DuckDB transaction choreography is **NOT INTEGRATED** into the core download flow.

**What This Means**:
- Artifacts are downloaded but NOT recorded in DuckDB
- Extraction happens but extracted files NOT in catalog
- Validations run but results NOT in DuckDB
- No "latest" pointer management
- **Doctor/Prune cannot work** without this data
- **No audit trail** of what happened

---

## PHASE 1 IMPLEMENTATION ROADMAP

### Task 1.1: Wire Boundaries into planning.py (2.5 hours)

**Location**: `src/DocsToKG/OntologyDownload/planning.py`

**Changes Required**:

#### 1.1.1 Import boundaries at top of fetch_one()
```python
from DocsToKG.OntologyDownload.catalog.boundaries import (
    download_boundary,
    extraction_boundary,
    validation_boundary,
    set_latest_boundary,
)
```

#### 1.1.2 Call download_boundary() after successful download
**Location**: After successful HTTP download (~line 1930)
```python
# After download completes successfully
from_bytes = destination.stat().st_size
result = download_boundary(
    artifact_id=sha256,
    version_id=version,
    service=spec.id,
    source_url=plan.url,
    fs_relpath=str(destination.relative_to(LOCAL_ONTOLOGY_DIR)),
    size_bytes=from_bytes,
    etag=etag,
    last_modified=last_modified,
    content_type=content_type,
    status="fresh",
)
# Now database knows about artifact
```

#### 1.1.3 Call extraction_boundary() after extraction completes
**Location**: After extract_archive_safe() completes (~line 1951)
```python
# After extraction succeeds
extracted_files = [
    {
        "file_id": compute_sha256(path),
        "artifact_id": sha256,
        "version_id": version,
        "relpath_in_version": str(path.relative_to(extraction_dir)),
        "format": infer_format(path),  # rdf, ttl, owl, obo, etc.
        "size_bytes": path.stat().st_size,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime),
    }
    for path in extracted_paths
]
ex_result = extraction_boundary(
    version_id=version,
    files=extracted_files,
)
```

#### 1.1.4 Call validation_boundary() after validators complete
**Location**: After run_validators() completes (~line 1963)
```python
# After validation runs
validation_records = [
    {
        "validation_id": f"{file_id}:{validator_name}:{now}",
        "file_id": file_id,
        "validator": validator_name,
        "passed": result.ok,
        "details_json": json.dumps(result.details),
        "run_at": datetime.now(timezone.utc),
    }
    for file_id, validator_name, result in validation_results.items()
]
val_result = validation_boundary(
    version_id=version,
    validations=validation_records,
)
```

#### 1.1.5 Call set_latest_boundary() after all validation passes
**Location**: After successful fetch_one() completion (~line 2020)
```python
# Before returning FetchResult
if all_validations_passed:
    latest_result = set_latest_boundary(
        version_id=version,
        service=spec.id,
        by=f"fetch_one:{spec.id}",
    )
```

---

### Task 1.2: Complete Missing CLI Commands (2 hours)

**Location**: `src/DocsToKG/OntologyDownload/cli/db_cmd.py` or new `cli/db_commands.py`

**Commands to Implement**:

1. **`db files --version <v> [--format FORMAT]`**
   - List extracted files for a version
   - Filter by format (ttl, rdf, owl, obo, etc.)
   - Show: relpath, size, mtime

2. **`db stats --version <v>`**
   - Statistics for a version
   - Total files, total bytes, validation summary
   - Format breakdown

3. **`db delta <v1> <v2>`**
   - Compare two versions
   - Show: new files, removed files, renamed files
   - Validation differences

4. **`db doctor [--fix]`**
   - Run health checks
   - Detect DB↔FS mismatches
   - Optionally auto-repair

5. **`db prune --keep N [--service S] [--dry-run]`**
   - Identify orphaned files
   - Delete old versions keeping last N
   - Show reclaimed space

6. **`db backup [--output DIR]`**
   - Create timestamped backup of database
   - Copy to backup directory

---

### Task 1.3: Wire Observability Events (2 hours)

**Location**: `src/DocsToKG/OntologyDownload/catalog/boundaries.py` and catalog modules

**Changes Required**:

#### 1.3.1 Add event emission to download_boundary()
```python
from DocsToKG.OntologyDownload.observability.events import emit_event

emit_event(
    "db.tx.download_boundary",
    level="INFO",
    payload={
        "boundary": "download",
        "artifact_id": artifact_id,
        "version_id": version_id,
        "service": service,
        "size_bytes": size_bytes,
        "status": status,
    },
)
```

#### 1.3.2 Add event emission to extraction_boundary()
```python
emit_event(
    "db.tx.extraction_boundary",
    level="INFO",
    payload={
        "boundary": "extract",
        "version_id": version_id,
        "files_count": len(files),
        "total_size": sum(f["size_bytes"] for f in files),
    },
)
```

#### 1.3.3 Add event emission to validation_boundary()
```python
emit_event(
    "db.tx.validation_boundary",
    level="INFO",
    payload={
        "boundary": "validate",
        "version_id": version_id,
        "validations_count": len(validations),
        "passed": sum(1 for v in validations if v["passed"]),
        "failed": sum(1 for v in validations if not v["passed"]),
    },
)
```

#### 1.3.4 Add event emission to set_latest_boundary()
```python
emit_event(
    "db.tx.latest_boundary",
    level="INFO",
    payload={
        "boundary": "latest",
        "version_id": version_id,
        "service": service,
        "by": by,
    },
)
```

#### 1.3.5 Add event emission to doctor/prune operations
- Emit on issue detection
- Emit on repair actions
- Emit on orphan detection
- Emit on deletion

---

### Task 1.4: Wire Settings Integration (1.5 hours)

**Location**: `src/DocsToKG/OntologyDownload/settings.py` and `database.py`

**Changes Required**:

#### 1.4.1 Define DuckDBSettings in settings.py
```python
from pydantic import BaseModel, Field
from pathlib import Path

class DuckDBSettings(BaseModel):
    """DuckDB catalog configuration."""
    path: Path = Field(
        default_factory=lambda: Path.home() / ".catalog" / "ontofetch.duckdb",
        description="Path to DuckDB file"
    )
    threads: int = Field(
        default=8,
        description="Number of threads for DuckDB"
    )
    readonly: bool = Field(default=False, description="Open in read-only mode")
    writer_lock: bool = Field(default=True, description="Use writer lock")
```

#### 1.4.2 Define StorageSettings in settings.py
```python
class StorageSettings(BaseModel):
    """Storage backend configuration."""
    root: Path = Field(
        default_factory=lambda: Path.home() / "ontologies",
        description="Storage root directory"
    )
    latest_name: str = Field(
        default="LATEST.json",
        description="Name of latest pointer file"
    )
    write_latest_mirror: bool = Field(
        default=True,
        description="Write JSON mirror of latest pointer"
    )
```

#### 1.4.3 Add to ResolvedConfig
```python
class ResolvedConfig:
    # ... existing fields ...
    db: DuckDBSettings = Field(default_factory=DuckDBSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
```

#### 1.4.4 Wire into config_hash
```python
def compute_config_hash(config: ResolvedConfig) -> str:
    """Hash all config including DB settings."""
    config_dict = {
        # ... existing fields ...
        "db": {
            "path": str(config.db.path),
            "threads": config.db.threads,
            "readonly": config.db.readonly,
            "writer_lock": config.db.writer_lock,
        },
        "storage": {
            "root": str(config.storage.root),
            "latest_name": config.storage.latest_name,
            "write_latest_mirror": config.storage.write_latest_mirror,
        },
    }
    return hashlib.sha256(json.dumps(config_dict).encode()).hexdigest()
```

---

### Task 1.5: Create Integration Tests (1.5 hours)

**Location**: `tests/ontology_download/test_catalog_integration.py`

**Test Coverage**:

#### 1.5.1 E2E Download → Extract → Validate → Latest Flow
```python
def test_e2e_full_pipeline():
    """Test complete flow: download → extract → validate → latest."""
    # Setup
    spec = FetchSpec(id="TEST", resolver="direct", target_formats=("owl",))
    
    # Download
    result = fetch_one(spec)
    assert result.status == "success"
    
    # Verify in database
    repo = Repo(db_config)
    versions = repo.list_versions(service="TEST")
    assert len(versions) > 0
    
    # Verify latest
    latest = repo.get_latest("default")
    assert latest == result.version_id
```

#### 1.5.2 Doctor Detection Tests
```python
def test_doctor_detects_missing_file():
    """Doctor should detect when file is missing from FS."""
    # Insert row, then delete file
    # Run doctor
    # Verify issue detected
```

#### 1.5.3 Prune Tests
```python
def test_prune_keeps_latest_n():
    """Prune should keep only N versions."""
    # Create 5 versions
    # Prune keeping 2
    # Verify only 2 remain
```

#### 1.5.4 Delta Tests
```python
def test_delta_shows_differences():
    """Delta should show file additions/removals."""
    # Create v1 with files A, B
    # Create v2 with files B, C
    # Delta should show: +C, -A, unchanged B
```

---

## IMPLEMENTATION SEQUENCE

1. **Task 1.4: Settings** (1.5 hrs) ← START HERE
   - Define DuckDBSettings + StorageSettings
   - Wire into config_hash
   - Ensures database initialized properly

2. **Task 1.2: CLI Commands** (2 hrs)
   - Can be implemented independently
   - Useful for manual verification

3. **Task 1.1: Wire Boundaries** (2.5 hrs) ← CRITICAL
   - Integrates planning.py with database
   - Core of the fix

4. **Task 1.3: Observability** (2 hrs)
   - Adds event emission
   - Important for audit trail

5. **Task 1.5: Tests** (1.5 hrs)
   - Validates the integration
   - E2E coverage

---

## SUCCESS CRITERIA

✅ **Phase 1 Done When**:
- [x] No boundary calls found in planning.py (confirmed)
- [ ] Settings fully wired with config_hash
- [ ] Download boundary called after HTTP success
- [ ] Extraction boundary called after archive extraction
- [ ] Validation boundary called after validator runs
- [ ] Latest boundary called on success
- [ ] All events emitted with `{run_id, config_hash}`
- [ ] All 6 CLI commands working
- [ ] All integration tests passing (100%)
- [ ] Zero linting violations
- [ ] 100% type-safe

---

## ESTIMATED TIME BREAKDOWN

| Task | Hours | Status |
|------|-------|--------|
| 1.4 Settings | 1.5 | pending |
| 1.2 CLI Commands | 2.0 | pending |
| 1.1 Boundaries | 2.5 | pending |
| 1.3 Observability | 2.0 | pending |
| 1.5 Tests | 1.5 | pending |
| **TOTAL** | **~9** | pending |

---

**Next Action**: Implement Task 1.4 (Settings Integration) first as foundation for all others.

