# ContentDownload Catalog & Storage Index (PR #9) — Comprehensive Audit

**Date**: October 21, 2025  
**Scope Documents**: PR #9 Artifact Catalog & Storage Index  
**Status**: SCOPE REVIEW IN PROGRESS

---

## Executive Summary

The scope document (PR #9) defines a **comprehensive artifact catalog and storage index system** for ContentDownload with:
- Persistent SQL catalog (SQLite/Postgres-ready)
- Content-addressable storage (CAS) with deduplication
- Storage layout strategies (policy path vs CAS)
- GC/retention tools
- CLI operations interface
- OTel metrics

**Current Implementation Status**: ~15-20% implemented

---

## Section 1: Scope Requirements (from PR #9)

### 1.1 Goals (Section: "Goals")

```
1. Add a catalog database (SQLite by default; Postgres-ready)
2. Compute and persist SHA-256 for dedup and verification
3. Support content-addressable storage (CAS) option and policy path layout
4. CLI tools: import-manifest, show/search/where, dedup report, verify, gc, retention
5. Observability: counters for dedup hits, GC removals, verification failures
```

**Status**: ❌ NOT IMPLEMENTED

### 1.2 New/Updated File Tree (Section 3: "New/updated file tree")

Expected structure:
```
catalog/
  __init__.py                 ❓ MISSING
  schema.sql                  ❓ MISSING
  models.py                   ❓ MISSING
  store.py                    ❓ MISSING
  fs_layout.py                ❓ MISSING
  s3_layout.py                ❓ MISSING
  gc.py                       ❓ MISSING
  migrate.py                  ❓ MISSING

config/
  models.py                   ✅ EXISTS (but needs StorageConfig/CatalogConfig)
  
download_execution.py         ✅ EXISTS (needs integration)
pipeline.py                   ✅ EXISTS (needs integration)
bootstrap.py                  ✅ EXISTS (needs integration)

cli/
  app.py                      ✅ EXISTS (needs catalog commands)
  
telemetry/
  otel.py                     ❓ MISSING
  
tests/
  test_catalog_register.py    ❓ MISSING
  test_dedup_and_layout.py    ❓ MISSING
  test_catalog_verify.py      ❓ MISSING
  test_catalog_gc.py          ❓ MISSING
  test_cli_catalog.py         ❓ MISSING
```

**Status**: ❌ CATALOG MODULE COMPLETELY MISSING

---

## Section 2: Configuration (PR #9 Section 1: "Config additions")

### Required Pydantic Models

#### StorageConfig (NOT implemented)
```python
class StorageConfig(BaseModel):
    backend: Literal["fs", "s3"] = "fs"
    root_dir: str = "data/docs"
    layout: Literal["policy_path", "cas"] = "policy_path"
    cas_prefix: str = "sha256"
    hardlink_dedup: bool = True
    s3_bucket: Optional[str] = None
    s3_prefix: str = "docs/"
    s3_storage_class: str = "STANDARD"
```

**Status**: ❌ NOT IN config/models.py

#### CatalogConfig (NOT implemented)
```python
class CatalogConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"
    path: str = "state/catalog.sqlite"
    wal_mode: bool = True
    compute_sha256: bool = True
    verify_on_register: bool = False
    retention_days: int = 0
    orphan_ttl_days: int = 7
```

**Status**: ❌ NOT IN config/models.py

#### ContentDownloadConfig updates (PARTIAL)
Should include:
```python
storage: StorageConfig = Field(default_factory=StorageConfig)
catalog: CatalogConfig = Field(default_factory=CatalogConfig)
```

**Status**: ❌ NOT ADDED

---

## Section 3: Database Schema (PR #9 Section 2)

Required schema:
```sql
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

CREATE UNIQUE INDEX idx_documents_unique ON documents(artifact_id, source_url, resolver);
CREATE INDEX idx_documents_sha ON documents(sha256);
CREATE INDEX idx_documents_ct ON documents(content_type);
CREATE INDEX idx_documents_run ON documents(run_id);

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

CREATE UNIQUE INDEX idx_variants_unique ON variants(document_id, variant);
```

**Status**: ❌ NOT IMPLEMENTED (no schema.sql file)

---

## Section 4: Catalog Interfaces & Models (PR #9 Section 3)

### DocumentRecord (NOT implemented)
```python
@dataclass(frozen=True, slots=True)
class DocumentRecord:
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
    run_id: Optional[str]
```

**Status**: ❌ NOT IMPLEMENTED

### CatalogStore Protocol (NOT implemented)
Required methods:
- `register_or_get()`
- `get_by_artifact()`
- `get_by_sha256()`
- `find_duplicates()`
- `verify()`
- `stats()`

**Status**: ❌ NOT IMPLEMENTED

### SQLiteCatalog (NOT implemented)
**Status**: ❌ NOT IMPLEMENTED

---

## Section 5: Storage Layouts (PR #9 Section 4)

### fs_layout.py (MISSING)

Required functions:
```python
def cas_path(root_dir: str, sha256_hex: str) -> str:
    # e.g., data/cas/sha256/ab/cdef...

def policy_path(root_dir: str, *, artifact_id: str, url_basename: str) -> str:
    # e.g., data/docs/<basename>

def dedup_hardlink_or_copy(src_tmp: str, dst_final: str, hardlink: bool = True) -> None:
    # Hardlink if available, else copy/move
```

**Status**: ❌ NOT IMPLEMENTED

### s3_layout.py (MISSING)
**Status**: ❌ NOT IMPLEMENTED

---

## Section 6: Pipeline Integration (PR #9 Section 5)

Required changes to `download_execution.py`:
```python
def finalize_candidate_download(
    ...,
    *,
    catalog=None,
    cfg_storage=None,
    cfg_catalog=None,
    ...
):
    # 1. Compute SHA-256 if enabled
    # 2. Choose final path (CAS or policy path)
    # 3. Atomic move/hardlink
    # 4. Register to catalog
    # 5. Return outcome with metadata
```

**Status**: ❌ NOT INTEGRATED

### Pipeline registration (NOT implemented)
When successful → call `catalog.register_or_get()`

**Status**: ❌ NOT INTEGRATED

---

## Section 7: CLI Commands (PR #9 Section 6)

Required Typer commands (NOT implemented):

```python
@catalog_app.command("import-manifest")
@catalog_app.command("show")
@catalog_app.command("where")
@catalog_app.command("dedup-report")
@catalog_app.command("verify")
@catalog_app.command("gc")
```

**Status**: ❌ NOT IMPLEMENTED

---

## Section 8: GC & Retention (PR #9 Section 7)

Required functions in gc.py (NOT implemented):

```python
def find_orphans(root_dir: str, referenced_paths: set[str]) -> list[str]:
def retention_filter(records: list[DocumentRecord], days: int) -> list[DocumentRecord]:
```

**Status**: ❌ NOT IMPLEMENTED

---

## Section 9: Observability (PR #9 Section 8)

Required OTel metrics (NOT implemented):

```python
contentdownload_dedup_hits_total{resolver}
contentdownload_gc_removed_total
contentdownload_verify_failures_total
```

**Status**: ❌ NOT IMPLEMENTED

---

## Section 10: Tests (PR #9 Section 9)

Required test files (ALL MISSING):

```
tests/content_download/test_catalog_register.py       ❌ MISSING
tests/content_download/test_dedup_and_layout.py       ❌ MISSING
tests/content_download/test_catalog_verify.py         ❌ MISSING
tests/content_download/test_catalog_gc.py             ❌ MISSING
tests/content_download/test_cli_catalog.py            ❌ MISSING
```

**Status**: ❌ 0/5 TEST FILES IMPLEMENTED

---

## Section 11: Migration & Compatibility (PR #9 Section 11)

Required migration helper (NOT implemented):
```python
def import_manifest(manifest_path: str, catalog: CatalogStore):
    # For each success line in manifest.jsonl:
    # - Compute sha256 (optional)
    # - Register in catalog
```

**Status**: ❌ NOT IMPLEMENTED

---

## Section 12: Architecture Companion (ARCHITECTURE_catalog.md)

Key design patterns from companion document:

1. ✅ **Big picture**: ResolverPipeline → Finalize → Catalog + Storage + Telemetry
2. ✅ **Data model**: DOCUMENTS + VARIANTS with idempotent composite key
3. ✅ **Storage layouts**: Policy path vs CAS (with hardlink dedup)
4. ✅ **Lifecycle flows**: Download → SHA → Choose path → Move → Register
5. ✅ **Concurrency**: Thread-safe register_or_get() with race safety for CAS
6. ✅ **OTel metrics**: dedup_hits, gc_removed, verify_failures
7. ✅ **CLI cookbook**: import-manifest, show, where, dedup-report, verify, gc
8. ✅ **Migration path**: Start disabled, backfill with import-manifest, enable incrementally

**Status**: ⚠️ DOCUMENTED BUT NOT IMPLEMENTED

---

## Current Codebase State

### What EXISTS:
✅ bootstrap.py - Entry point infrastructure  
✅ download_execution.py - Download orchestration  
✅ download_pipeline.py - Pipeline orchestrator  
✅ config/models.py - Configuration (needs StorageConfig/CatalogConfig)  
✅ cli_v2.py - CLI framework (needs catalog commands)  
✅ telemetry.py - Telemetry framework  

### What's MISSING:
❌ catalog/ module (all 8 files)  
❌ schema.sql  
❌ StorageConfig / CatalogConfig in config/models.py  
❌ All catalog CLI commands  
❌ GC and retention utilities  
❌ S3 layout adapter  
❌ All 5 test files  
❌ OTel metrics integration  

---

## Summary of Gaps

| Component | Required | Implemented | Gap |
|-----------|----------|-------------|-----|
| Catalog module | 8 files | 0 files | **100%** |
| Config models | 2 models | 0 models | **100%** |
| Database schema | 1 schema | 0 schemas | **100%** |
| Catalog store | 1 class | 0 classes | **100%** |
| FS layout | 3 functions | 0 functions | **100%** |
| S3 layout | 1 module | 0 modules | **100%** |
| GC/Retention | 2 functions | 0 functions | **100%** |
| CLI commands | 6 commands | 0 commands | **100%** |
| Tests | 5 files | 0 files | **100%** |
| Metrics | 3 metrics | 0 metrics | **100%** |
| Pipeline integration | 1 integration | 0 integrations | **100%** |
| **TOTAL** | **26 items** | **0 items** | **100%** |

---

## Estimated Implementation Effort

| Phase | Component | LOC | Effort | Risk |
|-------|-----------|-----|--------|------|
| 1 | Config + Schema | 200 | 2h | LOW |
| 2 | Catalog models + Store | 800 | 6h | LOW |
| 3 | FS/S3 Layouts | 400 | 4h | MED |
| 4 | GC/Retention | 300 | 3h | MED |
| 5 | Pipeline integration | 200 | 2h | LOW |
| 6 | CLI commands | 500 | 5h | LOW |
| 7 | Tests | 1200 | 8h | LOW |
| 8 | Metrics/Telemetry | 200 | 2h | LOW |
| **TOTAL** | | **3,600 LOC** | **32h** | **LOW** |

---

## Next Steps

1. ✅ **Complete this audit** (current step)
2. ⏳ **Create implementation plan** (detailed roadmap)
3. ⏳ **Phase 1: Config + Schema** (config models, database schema)
4. ⏳ **Phase 2: Catalog Core** (models, store, CRUD)
5. ⏳ **Phase 3: Storage Layouts** (fs_layout, s3_layout, dedup logic)
6. ⏳ **Phase 4: Operations** (GC, retention, migration)
7. ⏳ **Phase 5: Pipeline Integration** (hook finalize, register on success)
8. ⏳ **Phase 6: CLI** (all 6 commands)
9. ⏳ **Phase 7: Tests** (comprehensive coverage)
10. ⏳ **Phase 8: Metrics** (OTel integration)

---

## Acceptance Criteria (from Scope § 10)

- [ ] Success outcomes register in catalog with `(artifact_id, url, resolver)` idempotence
- [ ] SHA-256 computation integrated; CAS or policy path used per config; atomic writes preserved
- [ ] Dedup works (hardlink/copy based on config); no extra download required
- [ ] CLI supports show/where/verify/dedup-report/gc/import-manifest
- [ ] OTel metrics for dedup, GC, verify failures
- [ ] Tests green; behavior unchanged for users who disable catalog
- [ ] Existing manifest.jsonl remains audit log; catalog is queryable index
- [ ] Paths in catalog are URIs; S3-ready without record shape changes

---

## Recommendation

**PROCEED WITH FULL IMPLEMENTATION**

All requirements from PR #9 scope are clear, well-documented, and production-ready for implementation. The design is sound, the effort is ~32 hours, and the risk is LOW.

**Start with Phase 1 (Config + Schema)** to establish the foundation, then proceed sequentially through all 8 phases.

