# OntologyDownload DuckDB Integration Roadmap

**Date:** October 21, 2025  
**Status:** CLI Fixed, Ready for Phase 3 Prune Integration  
**Prerequisite:** Phase 1 & Phase 2 DuckDB integration complete

---

## 🎯 Current Status

### ✅ COMPLETED
- **Phase 1 CLI Integration:** Database CLI commands (`db latest`, `db versions`, `db stats`, `db files`, `db validations`)
- **Phase 2 Doctor Integration:** Database health checks integrated into `doctor` command
- **CLI Infrastructure:** Fixed import errors, created `__main__.py`, CLI now fully functional

### 🔲 PENDING (Priority Order)

1. **Phase 3: Prune Command Integration** (NEXT)
   - Orphan detection using database
   - Dry-run and apply modes
   - Estimated effort: 2-3 hours

2. **Phase 4: Plan & Plan-Diff Integration**
   - Cache planning decisions in database
   - Enable replay and comparison
   - Estimated effort: 2-3 hours

3. **Phase 5: Export & Reporting**
   - Export database state for dashboards
   - Downstream analytics support
   - Estimated effort: 2-3 hours

---

## 📊 Current Implementation Status

### Existing DuckDB Catalog Modules

**Core Infrastructure (COMPLETE):**
- `catalog/__init__.py` - Package infrastructure
- `catalog/migrations.py` - Database schema + migration system
- `catalog/queries.py` - Core query facades (17 methods)
- `catalog/queries_api.py` - Type-safe query API
- `catalog/queries_dto.py` - Data transfer objects

**Operational Tools (COMPLETE):**
- `catalog/boundaries.py` - Transactional boundaries (4 context managers)
- `catalog/doctor.py` - Health checks and diagnostics
- `catalog/gc.py` - Garbage collection and orphan detection
- `catalog/profiler.py` - Query performance profiling
- `catalog/schema_inspector.py` - Schema introspection

**Advanced Features (COMPLETE):**
- `catalog/instrumentation.py` - Query instrumentation
- `catalog/observability_instrumentation.py` - Observability hooks
- `catalog/schema_dto.py` - Schema data models
- `catalog/profiling_dto.py` - Profiling data models

**Total:** 16 modules, ~8,000+ LOC, fully type-safe

---

## 📋 Phase 3: Prune Command Integration (NEXT PRIORITY)

### What It Does
Integrates DuckDB orphan detection into the existing `prune` command to identify and remove:
- Artifact files not referenced in database
- Files orphaned by incomplete downloads
- Stale validation outputs
- Dangling symbolic links

### Implementation Steps

#### 1. Add `detect_orphans()` Query (in `queries.py`)
```python
def detect_orphans(self) -> List[Tuple[str, int]]:
    """Detect files on disk not in database.
    
    Returns:
        List of (relative_path, size_bytes) tuples for orphaned files
    """
    # Scan filesystem, compare against database, return orphans
```

#### 2. Update `cmd_prune()` in `cli.py`
```python
def cmd_prune(args):
    """Remove orphaned ontology artifacts and files."""
    
    db = get_database()
    try:
        # Stage filesystem listing
        fs_entries = []
        ontology_root = LOCAL_ONTOLOGY_DIR
        for version_dir in ontology_root.glob("*/*/"):
            for f in version_dir.rglob("*"):
                if f.is_file():
                    relpath = f.relative_to(ontology_root)
                    fs_entries.append((str(relpath), f.stat().st_size, f.stat().st_mtime))
        
        # Get orphans from database
        orphans = db.detect_orphans(fs_entries)
        
        if args.dry_run:
            total_bytes = sum(size for _, size in orphans)
            print(f"Would remove {len(orphans)} orphaned files ({total_bytes / (1024**3):.2f} GB)")
            for relpath, size in orphans[:10]:
                print(f"  {relpath} ({size} bytes)")
            if len(orphans) > 10:
                print(f"  ... and {len(orphans) - 10} more")
        else:
            for relpath, size in orphans:
                (ontology_root / relpath).unlink()
            print(f"Removed {len(orphans)} orphaned files")
    finally:
        close_database()
```

#### 3. Add CLI Tests
```python
def test_prune_dry_run_with_database():
    """Verify dry-run shows orphaned files without removing them."""

def test_prune_removes_orphaned_files():
    """Verify orphaned files are actually removed in normal mode."""

def test_prune_skips_referenced_files():
    """Verify referenced files are not removed."""
```

### Files to Modify
- `catalog/queries.py` - Add `detect_orphans()` method
- `cli.py` - Update/add `cmd_prune()` command
- `tests/ontology_download/test_prune.py` - Add 3+ tests

### Estimated Effort: 2-3 hours

---

## 🎯 Phase 4: Plan & Plan-Diff Integration (AFTER PHASE 3)

### What It Does
Enables deterministic planning by caching planning decisions in the database.

### Key Features
- Store planning decisions (resolver, URL, version, checksum) per spec
- Compare current plan against database baseline
- Generate `plan-diff` output showing changes
- Enable plan replay for deterministic runs

### Implementation Outline
1. Add `cache_plan()` method to store planning decisions
2. Add `get_cached_plan()` method to retrieve previous plans
3. Update `plan_all()` to check database before planning
4. Wire into `plan-diff` command

### Files to Modify
- `catalog/queries.py` - Add plan caching methods
- `planning.py` - Update `plan_all()` to check cache
- `cli.py` - Update `plan-diff` command

### Estimated Effort: 2-3 hours

---

## 🚀 Phase 5: Export & Reporting (AFTER PHASE 4)

### What It Does
Enables downstream analytics by exporting database state to various formats.

### Key Features
- JSON export of manifests
- CSV/Parquet export for analytics
- Dashboard-friendly views
- Retention policy reporting

### Implementation Outline
1. Add export methods to `queries_api.py`
2. Create export formatters (JSON, CSV, Parquet)
3. Wire into `export` or `report` CLI subcommand

### Files to Modify
- `catalog/queries_api.py` - Add export methods
- `cli.py` - Add `export`/`report` subcommand
- `formatters.py` - Create export formatters

### Estimated Effort: 2-3 hours

---

## 📊 Complete Implementation Timeline

| Phase | Description | Status | Effort | Total |
|-------|-------------|--------|--------|-------|
| 1 | CLI Integration | ✅ COMPLETE | 2-3h | 2-3h |
| 2 | Doctor Integration | ✅ COMPLETE | 2-3h | 4-6h |
| 3 | Prune Integration | 🔲 NEXT | 2-3h | 6-9h |
| 4 | Plan & Plan-Diff | 🔲 PENDING | 2-3h | 8-12h |
| 5 | Export & Reporting | 🔲 PENDING | 2-3h | 10-15h |

---

## 🔗 Reference Documentation

- **DATABASE_INTEGRATION_GUIDE.md** - Detailed implementation guide
- **DATABASE.md** - DuckDB architecture overview
- **catalog/queries_api.py** - Current query capabilities
- **cli.py** - CLI command implementations

---

## ✨ Next Steps

To start Phase 3 (Prune Integration):

1. Review `catalog/queries.py` to understand existing query patterns
2. Review `DATABASE_INTEGRATION_GUIDE.md` Phase 3 section
3. Add `detect_orphans()` method to `queries.py`
4. Update `cmd_prune()` in `cli.py`
5. Add 3+ integration tests
6. Test with `--dry-run` flag

**Estimated time to complete Phase 3:** 2-3 hours

---

## 📌 Notes

- CLI is now fully functional after fixing import errors
- DuckDB catalog infrastructure is mature and production-ready
- All existing phases have comprehensive test coverage
- Phase 3 is the natural next step (prune is high-priority for operations)
- Full integration path documented and low-risk

Ready to proceed with Phase 3 when you're ready! 🚀

