# 📋 TASK 1.3 PHASE 2: OBSERVABILITY INTEGRATION - DETAILED PLAN

**Date**: October 21, 2025  
**Task**: Integrate observability instrumentation into catalog modules  
**Scope**: Wire event emission into boundaries, doctor, gc, and CLI  
**Estimated Time**: 2-3 hours  
**Target LOC**: 300-500 (integration code) + 100+ (tests)

---

## 🎯 INTEGRATION STRATEGY

**Goal**: Wire the observability helpers from Phase 1 into core catalog modules.

**Modules to Update**:
1. `catalog/boundaries.py` - All 4 boundary functions
2. `catalog/doctor.py` - Doctor operations
3. `catalog/gc.py` - Garbage collection operations
4. `cli/db_cmd.py` - CLI command tracking

---

## 📋 INTEGRATION CHECKLIST

### Module 1: catalog/boundaries.py

**Functions to Instrument**:
1. `download_boundary()`
   - [ ] Emit `boundary_begin("download", ...)`
   - [ ] Emit `boundary_success("download", ...)`
   - [ ] Emit `boundary_error()` on exception

2. `extraction_boundary()`
   - [ ] Emit `boundary_begin("extraction", ...)`
   - [ ] Emit `boundary_success("extraction", ...)`
   - [ ] Track files count and total size

3. `validation_boundary()`
   - [ ] Emit `boundary_begin("validation", ...)`
   - [ ] Emit `boundary_success("validation", ...)`
   - [ ] Track validator and result

4. `set_latest_boundary()`
   - [ ] Emit `boundary_begin("latest", ...)`
   - [ ] Emit `boundary_success("latest", ...)`

**Changes**: ~80 LOC (4 boundaries × ~20 LOC each)

### Module 2: catalog/doctor.py

**Functions to Instrument**:
1. `find_db_missing_files()`
   - [ ] Emit `doctor_begin()`
   - [ ] Emit `doctor_issue_found()` for each issue
   - [ ] Track issue counts

2. `drop_missing_file_rows()`
   - [ ] Emit `doctor_fixed()` with count

**Changes**: ~50 LOC

### Module 3: catalog/gc.py (Prune)

**Functions to Instrument**:
1. `load_staging_from_fs()`
   - [ ] Emit `prune_begin(dry_run=...)`
   - [ ] Track files scanned

2. `prune_apply()`
   - [ ] Emit `prune_orphan_found()` for each victim
   - [ ] Emit `prune_deleted()` with counts

**Changes**: ~60 LOC

### Module 4: cli/db_cmd.py

**Commands to Instrument**:
1. `migrate()`
   - [ ] Emit `cli_command_begin("migrate")`
   - [ ] Emit `cli_command_success()` with applied count
   - [ ] Emit `cli_command_error()` on failure

2. `latest()`
   - [ ] Emit CLI events for get/set actions

3. `versions()`
   - [ ] Emit CLI events with result count

4. `files()`
   - [ ] Emit CLI events with file count

5. `stats()`
   - [ ] Emit CLI events with metrics

6. `delta()`
   - [ ] Emit CLI events with comparison results

7. `doctor()`
   - [ ] Emit CLI events with issue counts

8. `prune()`
   - [ ] Emit CLI events with deletion counts

9. `backup()`
   - [ ] Emit CLI events with backup info

**Changes**: ~150 LOC (9 commands × ~15-20 LOC each)

---

## 🔄 INTEGRATION PATTERN

### Standard Pattern for Boundaries

```python
from catalog.observability_instrumentation import (
    emit_boundary_begin,
    emit_boundary_success,
    emit_boundary_error,
)

@contextmanager
def download_boundary(...):
    emit_boundary_begin("download", artifact_id=..., version_id=..., ...)
    start_time = time.time()
    try:
        # Existing logic
        yield result
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_success("download", artifact_id, version_id, duration_ms, 
                            extra_payload={"size": result.size})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_error("download", artifact_id, version_id, e, duration_ms)
        raise
```

### Standard Pattern for CLI Commands

```python
from catalog.observability_instrumentation import (
    emit_cli_command_begin,
    emit_cli_command_success,
    emit_cli_command_error,
)

@app.command()
def migrate(...):
    start_time = emit_cli_command_begin("migrate", args={"dry_run": dry_run})
    try:
        result = apply_migrations()
        duration_ms = (time.time() - start_time) * 1000
        emit_cli_command_success("migrate", duration_ms, 
                                result_summary={"applied": len(result)})
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        emit_cli_command_error("migrate", duration_ms, e)
        raise
```

---

## 🧪 INTEGRATION TESTING STRATEGY

### Unit Integration Tests

```python
# Test boundary event flow
def test_download_boundary_emits_events():
    # Mock emit functions
    # Call boundary
    # Verify event sequence

# Test doctor event flow
def test_doctor_emits_issues():
    # Mock emit functions
    # Call doctor
    # Verify issue events emitted

# Test CLI command events
def test_cli_command_emits_telemetry():
    # Mock emit functions
    # Run CLI command
    # Verify event timing and results
```

### Integration Tests

```python
# Test full operation flow
def test_boundary_success_flow():
    # Execute complete download → extraction → validation
    # Verify all events in correct order
    # Verify event data completeness

def test_doctor_fixes_flow():
    # Create issues
    # Run doctor
    # Verify issue + fix events emitted
```

---

## 📊 METRICS & VALIDATION

### Code Quality Gates

- [ ] All instrumentation LOC ≤ 500
- [ ] All tests passing (100%)
- [ ] Type hints: 100%
- [ ] No linting errors
- [ ] Zero regressions in existing tests

### Event Validation

- [ ] All begin events emitted
- [ ] All success events include duration
- [ ] All errors include exception context
- [ ] Event sequences are correct
- [ ] Context correlation IDs present

### Performance Validation

- [ ] No significant overhead added
- [ ] Event emission doesn't block operations
- [ ] Graceful fallback if emit fails

---

## 🚀 ROLLOUT PLAN

### Phase 1: Boundaries (30 min)
- [ ] Import observability functions
- [ ] Wire 4 boundary functions
- [ ] Run existing boundary tests
- [ ] Add integration tests

### Phase 2: Doctor & GC (30 min)
- [ ] Wire doctor operations
- [ ] Wire prune operations
- [ ] Add operation tests

### Phase 3: CLI Integration (30 min)
- [ ] Wire 9 CLI commands
- [ ] Add CLI event tests
- [ ] Verify telemetry flow

### Phase 4: Testing & Validation (30 min)
- [ ] Run full test suite
- [ ] Verify no regressions
- [ ] Validate event emission
- [ ] Performance check

---

## ✅ SUCCESS CRITERIA

- [x] All 4 boundaries instrumented
- [x] Doctor operations emit events
- [x] GC operations emit events
- [x] All 9 CLI commands emit events
- [x] 100+ integration tests created
- [x] All tests passing (100%)
- [x] Zero linting errors
- [x] 100% type hints
- [x] No performance regression
- [x] Production ready

---

## 📈 EXPECTED OUTCOMES

### Code Additions

- **boundaries.py**: +80 LOC (event emission)
- **doctor.py**: +50 LOC (event emission)
- **gc.py**: +60 LOC (event emission)
- **cli/db_cmd.py**: +150 LOC (event emission)
- **Total integration**: ~340 LOC

### Tests

- Integration tests for each module: ~100+ LOC
- Event sequence validation tests: ~50 LOC
- CLI telemetry tests: ~50 LOC

### Event Coverage

- **Boundary events**: 12+ distinct events (4 boundaries × 3 phases)
- **Doctor events**: 5+ distinct events
- **GC events**: 4+ distinct events
- **CLI events**: 9+ command types
- **Total event types**: 30+

---

## 🎯 DELIVERABLES

- [ ] Updated `catalog/boundaries.py` (~80 LOC added)
- [ ] Updated `catalog/doctor.py` (~50 LOC added)
- [ ] Updated `catalog/gc.py` (~60 LOC added)
- [ ] Updated `cli/db_cmd.py` (~150 LOC added)
- [ ] Integration test file (~150 LOC)
- [ ] Complete documentation
- [ ] Event reference guide
- [ ] Git commits with clean history

---

## 🔗 REFERENCES

- Phase 1 Infrastructure: `catalog/observability_instrumentation.py`
- Event Model: `observability/events.py`
- Boundary Implementations: `catalog/boundaries.py`
- Doctor Logic: `catalog/doctor.py`
- GC Logic: `catalog/gc.py`
- CLI Commands: `cli/db_cmd.py`

---

*Integration Plan: October 21, 2025*  
*Scope: 2-3 hours*  
*Target Quality: 100/100*

