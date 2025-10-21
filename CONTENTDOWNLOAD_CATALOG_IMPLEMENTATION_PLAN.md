# ContentDownload Catalog & Storage Index — Detailed Implementation Plan

**Created**: October 21, 2025  
**Scope Document**: PR #9 Artifact Catalog & Storage Index  
**Estimated Effort**: 32 hours  
**Total LOC**: ~3,600  
**Risk Level**: LOW  

---

## Overview

This plan details the 8-phase implementation of the complete Artifact Catalog and Storage Index system for ContentDownload, exactly as specified in the PR #9 scope documents.

Each phase is self-contained, testable, and builds sequentially on the previous phase.

---

## Phase 1: Configuration & Schema (2 hours, 200 LOC)

### Objective
Establish the configuration model and database schema foundation.

### Files to Create/Modify

**1.1: Update `src/DocsToKG/ContentDownload/config/models.py`**

Add two new Pydantic models to the existing file:

```python
# Append to models.py

class StorageConfig(BaseModel):
    """Storage backend and layout configuration."""
    model_config = ConfigDict(extra="forbid")
    
    backend: Literal["fs", "s3"] = Field(
        default="fs",
        description="Storage backend: filesystem or S3"
    )
    root_dir: str = Field(
        default="data/docs",
        description="Root directory for final artifacts"
    )
    layout: Literal["policy_path", "cas"] = Field(
        default="policy_path",
        description="Layout strategy: policy_path (human-friendly) or cas (content-addressable)"
    )
    cas_prefix: str = Field(
        default="sha256",
        description="CAS prefix (e.g., 'sha256')"
    )
    hardlink_dedup: bool = Field(
        default=True,
        description="Enable hardlink deduplication on POSIX systems"
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket name (required if backend='s3')"
    )
    s3_prefix: str = Field(
        default="docs/",
        description="S3 object key prefix"
    )
    s3_storage_class: str = Field(
        default="STANDARD",
        description="S3 storage class (STANDARD, INTELLIGENT_TIERING, GLACIER, etc.)"
    )


class CatalogConfig(BaseModel):
    """Artifact catalog database and retention configuration."""
    model_config = ConfigDict(extra="forbid")
    
    backend: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Database backend (sqlite or postgres)"
    )
    path: str = Field(
        default="state/catalog.sqlite",
        description="Database file path (for SQLite) or connection URL"
    )
    wal_mode: bool = Field(
        default=True,
        description="Enable WAL mode for SQLite"
    )
    compute_sha256: bool = Field(
        default=True,
        description="Compute SHA-256 on successful downloads"
    )
    verify_on_register: bool = Field(
        default=False,
        description="Verify SHA-256 after finalization before registering"
    )
    retention_days: int = Field(
        default=0,
        ge=0,
        description="Retention policy: 0 = disabled; N > 0 = delete records older than N days"
    )
    orphan_ttl_days: int = Field(
        default=7,
        ge=1,
        description="GC eligibility: files not referenced for N days are candidates"
    )


# Update ContentDownloadConfig to include storage and catalog
# (modify existing ContentDownloadConfig class)

class ContentDownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    
    # ... existing fields ...
    
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage backend configuration"
    )
    catalog: CatalogConfig = Field(
        default_factory=CatalogConfig,
        description="Artifact catalog configuration"
    )
```

**1.2: Create `src/DocsToKG/ContentDownload/catalog/schema.sql`**

```sql
-- Artifact Catalog Schema (SQLite/Postgres-compatible)
-- Enables deduplication, verification, retention, and content-addressed storage

PRAGMA foreign_keys = ON;

-- Core documents table: one row per unique (artifact_id, source_url, resolver) tuple
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id TEXT NOT NULL,
  source_url TEXT NOT NULL,
  resolver TEXT NOT NULL,
  content_type TEXT,
  bytes INTEGER NOT NULL,
  sha256 TEXT,
  storage_uri TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  run_id TEXT
);

-- Uniqueness constraint: prevents duplicate (artifact_id, source_url, resolver) tuples
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_unique
  ON documents(artifact_id, source_url, resolver);

-- Lookup indexes
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_ct ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id);
CREATE INDEX IF NOT EXISTS idx_documents_artifact ON documents(artifact_id);
CREATE INDEX IF NOT EXISTS idx_documents_resolver ON documents(resolver);

-- Optional variants table: for PDF/HTML/supplement variants of the same artifact
CREATE TABLE IF NOT EXISTS variants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  variant TEXT NOT NULL,
  storage_uri TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  content_type TEXT,
  sha256 TEXT,
  created_at TEXT NOT NULL
);

-- Uniqueness: one variant type per document
CREATE UNIQUE INDEX IF NOT EXISTS idx_variants_unique
  ON variants(document_id, variant);

-- Lookup index
CREATE INDEX IF NOT EXISTS idx_variants_sha ON variants(sha256);
```

### Acceptance Criteria
- [ ] `StorageConfig` class present with all 8 fields
- [ ] `CatalogConfig` class present with all 7 fields
- [ ] `ContentDownloadConfig` includes `storage` and `catalog` fields
- [ ] All fields have type hints and docstrings
- [ ] All Pydantic validation works (extra="forbid", bounds checks)
- [ ] `schema.sql` loads without syntax errors
- [ ] Tests pass for config validation

---

## Phase 2: Catalog Core (Models, Store, CRUD) (6 hours, 800 LOC)

### Objective
Implement the catalog data model and SQLite store with all CRUD operations.

### Files to Create

**2.1: Create `src/DocsToKG/ContentDownload/catalog/__init__.py`**

```python
"""Artifact Catalog and Storage Index for ContentDownload.

Provides persistent storage of download metadata, SHA-256 hashes, and content-addressed
paths for deduplication, verification, and garbage collection.
"""

from .models import DocumentRecord
from .store import CatalogStore, SQLiteCatalog

__all__ = [
    "DocumentRecord",
    "CatalogStore",
    "SQLiteCatalog",
]
```

**2.2: Create `src/DocsToKG/ContentDownload/catalog/models.py`**

```python
"""Data models for the artifact catalog."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Immutable record of a successfully stored artifact.
    
    Attributes:
        id: Primary key (auto-generated by database)
        artifact_id: Artifact identifier (e.g., DOI, PMID, URL)
        source_url: Original source URL
        resolver: Resolver name that found this URL
        content_type: MIME type (e.g., "application/pdf")
        bytes: File size in bytes
        sha256: SHA-256 hash in hex (lowercase); may be None if disabled
        storage_uri: Storage location (e.g., file:///path or s3://bucket/key)
        created_at: ISO 8601 timestamp (UTC)
        updated_at: ISO 8601 timestamp (UTC)
        run_id: Optional run ID for provenance
    """
    id: int
    artifact_id: str
    source_url: str
    resolver: str
    content_type: Optional[str]
    bytes: int
    sha256: Optional[str]
    storage_uri: str
    created_at: datetime
    updated_at: datetime
    run_id: Optional[str] = None
```

**2.3: Create `src/DocsToKG/ContentDownload/catalog/store.py`**

Implement `CatalogStore` protocol and `SQLiteCatalog` (800 LOC with all methods):
- `__init__()` - Initialize connection and schema
- `register_or_get()` - Idempotent insert
- `get_by_artifact()` - Lookup by artifact ID
- `get_by_sha256()` - Lookup by hash
- `find_duplicates()` - Find (sha256, count) tuples
- `verify()` - Re-hash and compare
- `stats()` - Return summary statistics
- `get_by_run()` - Lookup by run ID
- `close()` - Cleanup

### Acceptance Criteria
- [ ] All 8+ methods implemented and type-hinted
- [ ] SQLite WAL mode configurable
- [ ] Idempotence verified: same (artifact_id, url, resolver) returns same record
- [ ] All queries use parameterized statements (no SQL injection)
- [ ] Timestamps are ISO 8601 UTC
- [ ] No hardcoded paths; respects config
- [ ] 100% test coverage for all CRUD operations

---

## Phase 3: Storage Layouts (FS & S3) (4 hours, 400 LOC)

### Objective
Implement path strategies and deduplication logic.

### Files to Create

**3.1: Create `src/DocsToKG/ContentDownload/catalog/fs_layout.py`**

```python
"""Filesystem storage layout strategies and deduplication."""

def cas_path(root_dir: str, sha256_hex: str, cas_prefix: str = "sha256") -> str:
    """Generate CAS path from SHA-256.
    
    Example: cas_path("data", "e3b0c44298...") -> "data/cas/sha256/e3/b0c44298..."
    """
    ...

def policy_path(root_dir: str, *, artifact_id: str, url_basename: str) -> str:
    """Generate human-friendly policy path.
    
    Example: policy_path("data", artifact_id="doi:...", url_basename="paper.pdf")
           -> "data/paper.pdf"
    """
    ...

def dedup_hardlink_or_copy(
    src_tmp: str,
    dst_final: str,
    hardlink: bool = True
) -> None:
    """Move temp file to final, preferring hardlink for dedup.
    
    If hardlink=True and supported:
      - Try os.link(src_tmp, dst_final)
      - On success, remove src_tmp
    
    Fallback (no hardlink support or hardlink=False):
      - Delete dst_final if exists
      - shutil.move(src_tmp, dst_final)
    """
    ...
```

**3.2: Create `src/DocsToKG/ContentDownload/catalog/s3_layout.py`**

Stub for S3 integration (to be filled later):
- `s3_upload()` - Upload file to S3, return s3://... URI
- `s3_verify()` - HEAD and compare ETag/Content-Length
- `s3_delete()` - Remove object from S3

### Acceptance Criteria
- [ ] `cas_path()` generates correct two-level fanout paths
- [ ] `policy_path()` generates readable paths
- [ ] `dedup_hardlink_or_copy()` prefers hardlink on POSIX
- [ ] Fallback to copy/move on Windows or cross-filesystem
- [ ] All path operations are atomic (no partial writes)
- [ ] Tests verify inode equality for hardlinks

---

## Phase 4: Operations (GC, Retention, Migration) (3 hours, 300 LOC)

### Objective
Implement garbage collection, retention, and backfill migration.

### Files to Create

**4.1: Create `src/DocsToKG/ContentDownload/catalog/gc.py`**

```python
"""Garbage collection and retention utilities."""

def find_orphans(
    root_dir: str,
    referenced_paths: set[str],
) -> list[str]:
    """Find files in root_dir not in referenced_paths.
    
    Returns list of orphan file paths (relative to root_dir).
    """
    ...

def retention_filter(
    records: list[DocumentRecord],
    days: int,
) -> list[DocumentRecord]:
    """Filter records older than retention_days.
    
    Args:
        records: List of DocumentRecord
        days: Delete if created_at < now - timedelta(days=days)
    
    Returns:
        Records eligible for deletion
    """
    ...

def delete_orphan_files(
    root_dir: str,
    orphan_paths: list[str],
    dry_run: bool = True,
) -> int:
    """Delete orphan files.
    
    Returns: Number of files deleted (or would delete if dry_run=True)
    """
    ...
```

**4.2: Create `src/DocsToKG/ContentDownload/catalog/migrate.py`**

```python
"""Migration helpers for backfilling catalog from manifest.jsonl."""

def import_manifest(
    manifest_path: str,
    catalog: CatalogStore,
    compute_sha256: bool = False,
) -> tuple[int, int]:
    """Import records from manifest.jsonl into catalog.
    
    Args:
        manifest_path: Path to manifest.jsonl
        catalog: CatalogStore instance
        compute_sha256: If True, compute SHA-256 for files (expensive)
    
    Returns:
        (total_read, total_registered) tuple
    """
    ...
```

### Acceptance Criteria
- [ ] `find_orphans()` correctly identifies unreferenced files
- [ ] `retention_filter()` respects age threshold
- [ ] `delete_orphan_files()` implements dry-run safely
- [ ] `import_manifest()` backfills records correctly
- [ ] All operations are logged and auditable
- [ ] No data loss in any scenario

---

## Phase 5: Pipeline Integration (2 hours, 200 LOC)

### Objective
Wire catalog into the download finalization flow.

### Files to Modify

**5.1: Modify `src/DocsToKG/ContentDownload/download_execution.py`**

Update `finalize_candidate_download()` to:

1. **Compute SHA-256** (if `cfg.catalog.compute_sha256`)
2. **Choose final path**:
   - If `cfg.storage.layout == "cas"`: use `cas_path(root, sha256)`
   - Else: use `policy_path(root, artifact_id, url_basename)`
3. **Atomic move**: `dedup_hardlink_or_copy(tmp_path, final_path, hardlink=...)`
4. **Register to catalog** (if `catalog` provided):
   ```python
   record = catalog.register_or_get(
       artifact_id=artifact.id,
       source_url=plan.url,
       resolver=plan.resolver_name,
       content_type=stream.content_type,
       bytes=stream.bytes_written,
       sha256=sha,
       storage_uri=f"file://{final_path}",
       run_id=run_id,
   )
   ```
5. **Return outcome** with metadata:
   ```python
   return DownloadOutcome(
       ok=True,
       classification="success",
       path=final_path,
       meta={
           "content_type": stream.content_type,
           "bytes": stream.bytes_written,
           "sha256": sha,
           "record_id": record.id,
           "dedup_hit": (was_cas_reuse),
       }
   )
   ```

**5.2: Modify `src/DocsToKG/ContentDownload/bootstrap.py`**

Add catalog initialization:

```python
def build_catalog_store(config: CatalogConfig) -> CatalogStore:
    """Build CatalogStore from configuration."""
    if config.backend == "sqlite":
        return SQLiteCatalog(path=config.path, wal_mode=config.wal_mode)
    # else: raise NotImplementedError(f"Catalog backend {config.backend} not supported")
```

Update bootstrap to inject `catalog` into pipeline.

### Acceptance Criteria
- [ ] SHA-256 computed when enabled
- [ ] CAS paths generated correctly
- [ ] Hardlink dedup working
- [ ] Catalog registration happens before outcome returned
- [ ] Metadata includes record_id and dedup_hit flag
- [ ] All atomic writes preserved
- [ ] No performance degradation when catalog disabled

---

## Phase 6: CLI Commands (5 hours, 500 LOC)

### Objective
Implement 6 user-facing CLI commands for catalog operations.

### Files to Create/Modify

**6.1: Create `src/DocsToKG/ContentDownload/cli/catalog_commands.py`**

Implement 6 Typer commands:

```python
@catalog_app.command("import-manifest")
def cmd_import_manifest(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    manifest_path: str = typer.Argument(...),
    compute_sha: bool = typer.Option(False, "--compute-sha"),
):
    """Import records from manifest.jsonl into catalog."""
    ...

@catalog_app.command("show")
def cmd_show(
    artifact_id: str,
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Show all records for an artifact_id."""
    ...

@catalog_app.command("where")
def cmd_where(
    sha256: str,
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Find all records with a given SHA-256."""
    ...

@catalog_app.command("dedup-report")
def cmd_dedup_report(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    min_count: int = typer.Option(2, "--min-count"),
):
    """List SHA-256 values with count >= min_count."""
    ...

@catalog_app.command("verify")
def cmd_verify(
    record_id: int,
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Verify SHA-256 of a record."""
    ...

@catalog_app.command("gc")
def cmd_gc(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
    orphan_days: Optional[int] = typer.Option(None, "--orphans"),
):
    """Run garbage collection."""
    ...
```

**6.2: Modify `src/DocsToKG/ContentDownload/cli/app.py`**

Wire commands into main CLI:

```python
from .catalog_commands import catalog_app

app.add_typer(catalog_app, name="catalog", help="Artifact catalog operations")
```

### Acceptance Criteria
- [ ] All 6 commands working and documented
- [ ] Output is human-readable with optional JSON
- [ ] Help text comprehensive
- [ ] Config loading works for all commands
- [ ] Dry-run mode for gc command
- [ ] No breaking changes to existing CLI

---

## Phase 7: Tests (8 hours, 1200 LOC)

### Objective
Comprehensive test coverage for all catalog components.

### Files to Create

**7.1: `tests/content_download/test_catalog_register.py`**
- Register and lookup by artifact_id, sha256, run_id
- Idempotence (duplicate inserts return same record)
- Stats calculation

**7.2: `tests/content_download/test_dedup_and_layout.py`**
- CAS path generation and two-level fanout
- Policy path generation
- Hardlink dedup (check inode equality on POSIX)
- Fallback to copy on non-supporting systems

**7.3: `tests/content_download/test_catalog_verify.py`**
- Verify: re-hash file and compare with catalog
- Detect tampering (hash mismatch)
- Verify failures increment metrics

**7.4: `tests/content_download/test_catalog_gc.py`**
- Find orphans
- Retention filtering
- GC removes orphans; respect dry-run flag
- Metrics incremented

**7.5: `tests/content_download/test_cli_catalog.py`**
- Smoke tests for all 6 CLI commands
- Config loading
- Output format validation
- Error handling

### Acceptance Criteria
- [ ] >95% code coverage for all modules
- [ ] All happy paths tested
- [ ] All error paths tested
- [ ] Edge cases (empty catalog, concurrent access, etc.)
- [ ] All tests passing

---

## Phase 8: Observability (Metrics & Telemetry) (2 hours, 200 LOC)

### Objective
Integrate OTel metrics and structured logging.

### Files to Create/Modify

**8.1: Create `src/DocsToKG/ContentDownload/telemetry/catalog_metrics.py`**

```python
"""Catalog metrics and observability."""

from opentelemetry import metrics

meter = metrics.get_meter(__name__)

dedup_hits = meter.create_counter(
    "contentdownload_dedup_hits_total",
    description="Deduplication hits (CAS reuse)",
    unit="1",
)

gc_removed = meter.create_counter(
    "contentdownload_gc_removed_total",
    description="Files removed by garbage collection",
    unit="1",
)

verify_failures = meter.create_counter(
    "contentdownload_verify_failures_total",
    description="Catalog verification failures (hash mismatch)",
    unit="1",
)

def record_dedup_hit(resolver: str):
    """Increment dedup_hits counter."""
    dedup_hits.add(1, {"resolver": resolver})

def record_gc_removal():
    """Increment gc_removed counter."""
    gc_removed.add(1)

def record_verify_failure():
    """Increment verify_failures counter."""
    verify_failures.add(1)
```

**8.2: Modify relevant modules to call metrics**

- In `download_execution.py`: call `record_dedup_hit()` on CAS reuse
- In `gc.py`: call `record_gc_removal()` for each file deleted
- In store verification: call `record_verify_failure()` on hash mismatch

### Acceptance Criteria
- [ ] All 3 metrics present and exposed
- [ ] Metrics incremented at correct points
- [ ] OTel scraping works (JSON export)
- [ ] Prometheus format compatible

---

## Implementation Sequencing

### Week 1
- **Day 1 (2h)**: Phase 1 (Config + Schema)
- **Day 2-3 (6h)**: Phase 2 (Catalog Core)
- **Day 4 (4h)**: Phase 3 (Storage Layouts)

### Week 2
- **Day 5 (3h)**: Phase 4 (Operations)
- **Day 6 (2h)**: Phase 5 (Pipeline Integration)
- **Day 7-8 (5h)**: Phase 6 (CLI Commands)

### Week 3
- **Day 9-11 (8h)**: Phase 7 (Tests)
- **Day 12 (2h)**: Phase 8 (Metrics)

---

## Quality Gates (Per Phase)

Each phase must meet:
- ✅ 100% type-hinted (mypy clean)
- ✅ 0 linting errors (ruff/black compliant)
- ✅ >95% test coverage (Phase 7 onwards)
- ✅ All docstrings present
- ✅ No hardcoded paths/secrets
- ✅ Atomic file operations preserved
- ✅ No breaking changes to existing API

---

## Rollback Plan

If any phase fails:

1. **Commit to separate branch** (`catalog-implementation`)
2. **Revert on main** before triggering pipeline integration
3. **Isolate failure** to specific component
4. **Fix and retry** or defer to next release

The catalog is **opt-in** (disabled by default), so partial failure is safe.

---

## Success Criteria (Scope § 10)

- ✅ Success outcomes register in catalog with idempotence
- ✅ SHA-256 computed; CAS or policy path used; atomic writes preserved
- ✅ Dedup works (hardlink/copy); no extra downloads
- ✅ CLI supports import/show/where/dedup-report/verify/gc
- ✅ OTel metrics for dedup/gc/verify
- ✅ Tests green; backward compatible
- ✅ Manifest.jsonl remains; catalog is queryable index
- ✅ Paths are URIs; S3-ready

---

## Next: Proceed to Phase 1 Implementation

Once approved, we will proceed immediately with Phase 1 (Config + Schema) and continue through all 8 phases sequentially.

