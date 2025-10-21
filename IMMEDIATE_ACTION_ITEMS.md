# IMMEDIATE ACTION ITEMS — OntologyDownload Scope Implementation

**Generated**: 2025-10-21
**Status**: VALIDATION COMPLETE | READY TO IMPLEMENT
**Owner**: AI Agent + Development Team

---

## ✅ COMPLETED THIS SESSION

1. ✅ **Scope Validation Document** — Created comprehensive analysis
   - File: `SCOPE_VALIDATION_AND_IMPLEMENTATION_PLAN.md` (2,000+ lines)
   - Status: COMPLETE

2. ✅ **Summary Document** — Quick reference guide
   - File: `SCOPE_VALIDATION_SUMMARY.txt` (300 lines)
   - Status: COMPLETE

3. ✅ **Architecture Diagrams** — Detailed integration flows
   - File: `SCOPE_ARCHITECTURE_DIAGRAM.md` (500+ lines)
   - Status: COMPLETE

4. ✅ **Memory Updated** — Scope validation recorded for future sessions
   - Memory ID: 10146057
   - Status: COMPLETE

---

## 📋 NEXT STEPS (Sequential Order)

### **DAY 1: Setup & Pillar 5A Foundation (Phase 1 of 10.5 days)**

#### Task 1: Create Package Structure
```bash
# Create catalog package directory
mkdir -p src/DocsToKG/OntologyDownload/catalog
touch src/DocsToKG/OntologyDownload/catalog/__init__.py
touch src/DocsToKG/OntologyDownload/catalog/connection.py
touch src/DocsToKG/OntologyDownload/catalog/migrations.py
touch src/DocsToKG/OntologyDownload/catalog/queries.py
touch src/DocsToKG/OntologyDownload/catalog/boundaries.py
touch src/DocsToKG/OntologyDownload/catalog/bulk_ingest.py
touch src/DocsToKG/OntologyDownload/catalog/doctor.py
touch src/DocsToKG/OntologyDownload/catalog/gc.py
touch src/DocsToKG/OntologyDownload/catalog/backup.py

# Create analytics package directory
mkdir -p src/DocsToKG/OntologyDownload/analytics
touch src/DocsToKG/OntologyDownload/analytics/__init__.py
touch src/DocsToKG/OntologyDownload/analytics/pipelines.py
touch src/DocsToKG/OntologyDownload/analytics/reports.py
touch src/DocsToKG/OntologyDownload/analytics/renderers.py
touch src/DocsToKG/OntologyDownload/analytics/io.py

# Create analytics CLI command file
touch src/DocsToKG/OntologyDownload/cli/analytics_cmd.py

# Create tests
mkdir -p tests/ontology_download/catalog
mkdir -p tests/ontology_download/analytics
touch tests/ontology_download/catalog/__init__.py
touch tests/ontology_download/catalog/test_migrations.py
touch tests/ontology_download/catalog/test_queries.py
touch tests/ontology_download/catalog/test_boundaries.py
touch tests/ontology_download/catalog/test_doctor_prune.py
touch tests/ontology_download/analytics/__init__.py
touch tests/ontology_download/analytics/test_pipelines.py
touch tests/ontology_download/analytics/test_reports.py
touch tests/ontology_download/analytics/test_renderers.py
```

#### Task 2: Start Phase 5A — Migration Runner
**Acceptance Criteria:**
- [ ] `catalog/migrations.py` complete (300 LOC)
- [ ] Idempotent migration runner tested
- [ ] 20 tests for migrations (passing)
- [ ] Query façades sketched

**Deliverables:**
- `catalog/migrations.py` — Idempotent runner
- `tests/ontology_download/catalog/test_migrations.py` — 20 unit tests

**Estimated Time**: 4–6 hours

---

### **DAYS 2–3: Query Facades & Boundaries (Phase 5A continuation)**

#### Task 3: Query Facades
**Files**: `catalog/queries.py` (250 LOC, 15 tests)

**Required Functions**:
```python
def list_versions(db) -> List[VersionRow]
def get_latest(db) -> Optional[VersionRow]
def list_files(db, version_id: str) -> List[FileRow]
def list_validations(db, version_id: str) -> List[ValidationRow]
def get_artifact_stats(db, version_id: str) -> Dict[str, int]
```

**Acceptance Criteria**:
- [ ] All functions return typed DTOs
- [ ] SQL encapsulated (no raw queries to callers)
- [ ] 15 tests passing
- [ ] Fixtures cover empty/multi-record cases

**Estimated Time**: 3–4 hours

#### Task 4: Transaction Context Managers
**Files**: `catalog/boundaries.py` (250 LOC, 15 tests)

**Required Context Managers**:
```python
@contextmanager
def download_boundary(db: Database, version_id: str, artifact_id: str)
    # INSERT artifacts, emit event

@contextmanager
def extraction_boundary(db: Database, artifact_id: str)
    # BULK INSERT extracted_files (Arrow), record audit JSON

@contextmanager
def validation_boundary(db: Database)
    # INSERT validations, emit event

@contextmanager
def set_latest_boundary(db: Database, version_id: str, latest_json_path: Path)
    # UPSERT latest_pointer + atomic LATEST.json write
```

**Acceptance Criteria**:
- [ ] All 4 boundaries atomic (FS + DB sync)
- [ ] Rollback on FS/DB failure tested
- [ ] 15 tests passing
- [ ] Context manager cleanup guaranteed

**Estimated Time**: 4–5 hours

---

### **DAYS 4–5: Doctor & Prune (Phase 5B)**

#### Task 5: Doctor Workflow
**Files**: `catalog/doctor.py` (350 LOC, 20 tests)

**Required Functions**:
```python
def doctor(db: Database, dry_run: bool = True) -> Dict[str, List]
    # Returns:
    # {
    #   "fs_orphans": [{"artifact_id": "...", "fs_relpath": "..."}],
    #   "db_orphans": [{"db_row": {...}, "reason": "missing_from_fs"}],
    #   "latest_mismatch": {...},
    #   "actions": ["DELETE artifact_id:...", "INSERT stub:..."]
    # }

def apply_fixes(db: Database, actions: List[str]) -> Dict[str, int]
    # Returns: {"deleted": 5, "inserted": 2, "fixed": 1}
```

**Acceptance Criteria**:
- [ ] Detects all 3 mismatch types
- [ ] Dry-run safe (print-only)
- [ ] Apply safe (user confirms)
- [ ] 20 tests passing
- [ ] Audit trail recorded

**Estimated Time**: 4–6 hours

#### Task 6: Prune Workflow
**Files**: `catalog/gc.py` (350 LOC, 20 tests)

**Required Functions**:
```python
def list_orphans(db: Database) -> List[OrphanRow]
    # Returns paths + sizes for candidates

def prune(db: Database, dry_run: bool = True, batch_size: int = 100) -> Dict[str, int]
    # Returns: {"deleted_files": 10, "bytes_freed": 5242880, "batches": 2}

def record_prune_event(db: Database, run_id: str, candidates: List, applied: bool)
    # Insert to events table for audit trail
```

**Acceptance Criteria**:
- [ ] Orphan detection correct
- [ ] Batch deletion safe
- [ ] Dry-run → apply workflow
- [ ] 20 tests passing
- [ ] Audit trail complete

**Estimated Time**: 4–6 hours

#### Task 7: Backup/Restore
**Files**: `catalog/backup.py` (150 LOC, 10 tests)

**Required Functions**:
```python
def backup(db: Database, target_dir: Path) -> Path
    # CHECKPOINT + copy .duckdb + schemas

def restore(source_dir: Path, target_db_path: Path) -> bool
    # Copy back + run migrate
```

**Acceptance Criteria**:
- [ ] Backup captures all schema versions
- [ ] Restore integrity verified
- [ ] 10 tests passing

**Estimated Time**: 2–3 hours

---

### **DAYS 5–6: Polars Analytics (Phase 6A)**

#### Task 8: Pipeline Builders
**Files**: `analytics/pipelines.py` (400 LOC, 25 tests)

**Required Functions**:
```python
def scan_audit_json(path: Path) -> pl.LazyFrame
    # Schema: path_rel, size, sha256, format, mtime

def fetch_version_arrow(db: Database, version_id: str) -> pa.Table
    # SELECT extracted_files WHERE version_id = ?

def normalize_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame
    # Cast to Int64, Categorical, UTC times

def stream_large_audit(path: Path, max_rows: int = None) -> Generator[pl.LazyFrame]
    # Streaming chunks for memory efficiency
```

**Acceptance Criteria**:
- [ ] Lazy evaluation verified
- [ ] Predicate pushdown confirmed
- [ ] Streaming works for 1M+ rows
- [ ] 25 tests passing

**Estimated Time**: 5–6 hours

#### Task 9: Core Reports
**Files**: `analytics/reports.py` (500 LOC, 30 tests)

**Required Reports**:
1. **Latest Summary**
   - Output: DataFrame with (format, count, size_bytes, pass_rate)

2. **Version Growth (A→B)**
   - Output: DataFrame with ADDED/REMOVED/MODIFIED/RENAMED rows

3. **Validation Health**
   - Output: DataFrame with (validator, pass_rate, regressed, fixed)

4. **Hotspots**
   - Output: DataFrame with (path|format, size_bytes|failure_rate) sorted by power law

**Acceptance Criteria**:
- [ ] All 4 reports produce consistent schemas
- [ ] Aggregations correct on fixtures
- [ ] 30 tests passing
- [ ] Power law properties verified

**Estimated Time**: 6–8 hours

#### Task 10: Renderers
**Files**: `analytics/renderers.py` (200 LOC, 15 tests)

**Required Functions**:
```python
def render_as_table(df: pl.DataFrame) -> str
    # Rich ASCII table

def render_as_json(df: pl.DataFrame, format: str = "jsonl") -> str
    # JSONL or compact JSON

def render_as_parquet(df: pl.DataFrame, path: Path) -> None
    # Write to Parquet
```

**Acceptance Criteria**:
- [ ] No data loss in any format
- [ ] Types preserved
- [ ] 15 tests passing

**Estimated Time**: 2–3 hours

---

### **DAYS 7–8: CLI Integration (Phase 6B)**

#### Task 11: Analytics CLI Commands
**Files**: `cli/analytics_cmd.py` (300 LOC, 15 tests)

**Required Subcommands**:
```bash
ontofetch report latest --version <v> --format table|json|parquet [--profile]
ontofetch report growth --a <v1> --b <v2> --by format|path --top 50 [--profile]
ontofetch report validation --version <v> --by validator|format [--profile]
ontofetch report hotspots --version <v> --top 20 [--profile]
```

**Acceptance Criteria**:
- [ ] All commands parse correctly
- [ ] Output formats work (table|json|parquet)
- [ ] `--profile` prints lazy plan
- [ ] 15 tests passing

**Estimated Time**: 3–4 hours

#### Task 12: End-to-End Integration Tests
**Files**: `tests/ontology_download/analytics/test_e2e.py` (150 LOC, 15 tests)

**Test Scenarios**:
1. Download + extract + validate → query latest report
2. Growth detection across 2 versions
3. Validation health tracking
4. Hotspots identification

**Acceptance Criteria**:
- [ ] Full pipeline → report queries work
- [ ] All 4 report types produce output
- [ ] 15 tests passing

**Estimated Time**: 2–3 hours

---

### **DAYS 8–9: Testing & Documentation**

#### Task 13: Comprehensive Testing
**Checklist**:
- [ ] All unit tests passing (300 tests)
- [ ] All integration tests passing (50 tests)
- [ ] Performance benchmarks met (< targets)
- [ ] Mypy full pass
- [ ] Ruff zero violations

**Estimated Time**: 4–6 hours

#### Task 14: Documentation & AGENTS.md Update
**Updates Required**:
- [ ] AGENTS.md Pillar 5 section
- [ ] AGENTS.md Pillar 6 section
- [ ] README.md new CLI commands
- [ ] Architecture diagrams (already done)
- [ ] Code examples in docs/

**Estimated Time**: 2–3 hours

---

### **DAY 10: Review & Deployment Prep**

#### Task 15: Code Review & Integration
**Checklist**:
- [ ] All 5 PRs merged to main
- [ ] Git tags created (v6.0.0-alpha, v6.0.0-rc1, v6.0.0)
- [ ] CI/CD green on all platforms
- [ ] Documentation deployed

**Estimated Time**: 2–3 hours

---

## 📊 PROJECT STATISTICS (Target Post-Implementation)

| Metric | Value |
|--------|-------|
| Total LOC | 12,898 |
| Total Tests | 759 |
| Test Pass Rate | 100% |
| Mypy Errors | 0 |
| Lint Violations | 0 |
| Phases Completed | 5.5–5.9 + 6.0–6.2 |
| PRs to Merge | 5 |
| Deployment Version | v6.0.0 |

---

## 🎯 SUCCESS CRITERIA

✅ All acceptance criteria met
✅ 100% test passing
✅ 100% type-safe
✅ 0 linting violations
✅ Performance targets met
✅ Documentation complete

---

## 📞 DEPENDENCIES & BLOCKERS

**None Identified** — All required packages already in `.venv`:
- ✅ DuckDB
- ✅ Polars
- ✅ PyArrow
- ✅ httpx + Hishel
- ✅ All existing DocsToKG modules

---

## 🚀 GO/NO-GO DECISION

**STATUS**: ✅ **GO** — Implementation Ready

- Scope validated
- Design finalized
- Architecture diagrams complete
- All integration points identified
- Risk mitigation in place
- No blockers identified

**Ready to Start**: Immediately

---

*End of Action Items*
