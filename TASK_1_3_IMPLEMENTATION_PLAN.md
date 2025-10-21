# 📋 TASK 1.3: OBSERVABILITY WIRING - IMPLEMENTATION PLAN

**Date**: October 21, 2025  
**Task**: Wire event emission to all boundaries and catalog operations  
**Scope**: Comprehensive telemetry for observability and debugging  
**Estimated Time**: 4-6 hours  
**Target LOC**: 500-800 (production) + 300-400 (tests)

---

## 🎯 OVERVIEW

**Goal**: Instrument all DuckDB catalog operations with structured event emission for comprehensive observability, debugging, and monitoring.

**Key Areas**:
1. Download boundary events
2. Extraction boundary events
3. Validation boundary events
4. Set latest boundary events
5. Doctor operation events
6. Garbage collection events
7. CLI command telemetry
8. Error and exception tracking

---

## 📋 IMPLEMENTATION PHASES

### Phase 1: Event Infrastructure Review (30 min)

**Objectives**:
- Understand existing event model (`observability/events.py`)
- Review event emission API (`emit_event()`)
- Check event sink configuration
- Validate context correlation

**Deliverables**:
- Event model documentation
- Integration patterns identified
- Context correlation strategy

### Phase 2: Boundary Instrumentation (120 min)

**Objectives**:
- Wire `download_boundary` events
- Wire `extraction_boundary` events
- Wire `validation_boundary` events
- Wire `set_latest_boundary` events

**For each boundary**:
- Begin event (with parameters)
- Success event (with results)
- Error event (with exception details)
- Completion event (with timing)

**Deliverables**:
- 300+ LOC instrumentation code
- Consistent event structure
- Performance metadata

### Phase 3: Catalog Operations Telemetry (90 min)

**Objectives**:
- Wire doctor operation events
- Wire garbage collection events
- Wire migration events
- Wire query performance events

**Events per operation**:
- Operation start (with context)
- Progress updates (for long operations)
- Result summary
- Error handling
- Performance metrics

**Deliverables**:
- 200+ LOC telemetry code
- Comprehensive operation tracking
- Performance profiling

### Phase 4: CLI Integration & Testing (120 min)

**Objectives**:
- Wire CLI command telemetry
- Create E2E instrumentation tests
- Validate event emission
- Performance verification

**Deliverables**:
- CLI event wiring
- 200+ LOC test code
- Integration test suite
- Performance baseline

---

## 🏗️ TECHNICAL ARCHITECTURE

### Event Model Review

```python
# Expected event structure (from observability/events.py)
Event(
    ts: datetime,
    type: str,  # e.g., "boundary.download.begin", "boundary.extract.success"
    level: str,  # "INFO", "WARN", "ERROR"
    ids: dict,   # {run_id, config_hash, boundary, ...}
    payload: dict  # Operation-specific data
)
```

### Event Categories

**Boundary Events**:
- `boundary.download.begin` - Download started
- `boundary.download.success` - Download completed
- `boundary.download.error` - Download failed
- `boundary.extraction.begin` - Extraction started
- `boundary.extraction.success` - Extraction completed
- `boundary.validation.begin` - Validation started
- `boundary.validation.success` - Validation completed
- `boundary.latest.begin` - Latest pointer update started
- `boundary.latest.success` - Latest pointer updated

**Catalog Events**:
- `catalog.doctor.begin` - Doctor check started
- `catalog.doctor.issue_found` - Issue detected
- `catalog.doctor.fixed` - Issue fixed
- `catalog.prune.begin` - Prune started
- `catalog.prune.orphan_found` - Orphan detected
- `catalog.prune.deleted` - Orphan deleted

**CLI Events**:
- `cli.command.begin` - Command started
- `cli.command.success` - Command completed
- `cli.command.error` - Command failed

**Performance Events**:
- `perf.query_slow` - Query exceeded threshold
- `perf.operation_slow` - Operation exceeded threshold

---

## 📊 EVENT PAYLOAD SPECIFICATIONS

### Boundary Events

**download_boundary.begin**:
```python
{
    "boundary": "download",
    "artifact_id": str,
    "version_id": str,
    "service": str,
    "source_url": str,  # May be sanitized
    "expected_size": int,
}
```

**download_boundary.success**:
```python
{
    "boundary": "download",
    "artifact_id": str,
    "actual_size": int,
    "duration_ms": float,
    "etag": str,
    "content_type": str,
}
```

**extraction_boundary.success**:
```python
{
    "boundary": "extraction",
    "artifact_id": str,
    "files_count": int,
    "total_size": int,
    "formats": dict,  # {format: count}
    "duration_ms": float,
}
```

**validation_boundary.success**:
```python
{
    "boundary": "validation",
    "file_id": str,
    "validator": str,
    "passed": bool,
    "duration_ms": float,
    "details": dict,
}
```

### Catalog Operation Events

**doctor.issue_found**:
```python
{
    "operation": "doctor",
    "issue_type": str,  # "missing_file", "orphan_record", "mismatch"
    "severity": str,  # "warning", "error"
    "affected_records": int,
    "details": dict,
}
```

**prune.orphan_found**:
```python
{
    "operation": "prune",
    "path": str,
    "size_bytes": int,
    "age_days": int,
}
```

---

## 🔄 INTEGRATION POINTS

### In `catalog/boundaries.py`

```python
# For each boundary function:

@contextmanager
def download_boundary(...):
    emit("boundary.download.begin", payload={...})
    try:
        # Existing logic
        yield result
        emit("boundary.download.success", payload={...})
    except Exception as e:
        emit("boundary.download.error", level="ERROR", payload={...})
        raise
```

### In `catalog/doctor.py`

```python
def find_db_missing_files(...):
    emit("catalog.doctor.begin")
    issues = []
    for issue in detected_issues:
        emit("catalog.doctor.issue_found", payload={...})
        issues.append(issue)
    emit("catalog.doctor.complete", payload={"total_issues": len(issues)})
    return issues
```

### In `catalog/gc.py`

```python
def prune_apply(...):
    emit("catalog.prune.begin")
    deleted = 0
    for victim in victims:
        emit("catalog.prune.orphan_found", payload={...})
        delete_file(victim)
        deleted += 1
    emit("catalog.prune.complete", payload={"deleted_count": deleted})
    return deleted
```

### In `cli/db_cmd.py`

```python
@app.command()
def command_name(...):
    emit("cli.command.begin", payload={"command": "name", "args": {...}})
    try:
        result = do_work()
        emit("cli.command.success", payload={"result_summary": {...}})
        return result
    except Exception as e:
        emit("cli.command.error", level="ERROR", payload={"error": str(e)})
        raise
```

---

## 🧪 TESTING STRATEGY

### Unit Tests

```python
# Test event emission for each boundary
def test_download_boundary_emits_begin_event():
    # Verify "boundary.download.begin" emitted
    
def test_download_boundary_emits_success_event():
    # Verify "boundary.download.success" emitted

def test_extraction_boundary_emits_events():
    # Verify complete event sequence
    
def test_error_handling_emits_error_event():
    # Verify error events emit on exceptions
```

### Integration Tests

```python
# Test event flow through complete operation
def test_full_download_extraction_event_flow():
    # Verify all events emitted in correct order
    
def test_doctor_operation_events():
    # Verify doctor telemetry
    
def test_prune_operation_events():
    # Verify prune telemetry
```

### Validation Tests

```python
# Verify event structure and content
def test_event_payload_structure():
    # Validate all required fields present
    
def test_event_context_correlation():
    # Verify run_id and config_hash in all events
    
def test_event_timestamps_monotonic():
    # Verify timestamps increase monotonically
```

---

## 🎯 SUCCESS CRITERIA

- [x] All boundaries instrumented with events
- [x] All catalog operations emit telemetry
- [x] All CLI commands emit telemetry
- [x] 200+ LOC tests (>80% passing)
- [x] Zero linting errors
- [x] 100% type hints
- [x] Event structure validated
- [x] Context correlation verified
- [x] Performance metrics collected
- [x] Error events properly formatted
- [x] Production quality code
- [x] Comprehensive documentation

---

## 📈 METRICS & QUALITY GATES

| Metric | Target | Gate |
|--------|--------|------|
| Instrumentation LOC | 500-800 | >= 500 |
| Test LOC | 200-400 | >= 200 |
| Test Pass Rate | >= 80% | >= 80% |
| Type Hints | 100% | 100% |
| Linting Errors | 0 | 0 |
| Event Coverage | 100% | All boundaries |
| Documentation | Complete | All code |

---

## 🚀 ROLLOUT PHASES

### Week 1: Core Instrumentation
- Phase 1-2: Event infrastructure & boundaries (4 hours)

### Week 2: Catalog Operations
- Phase 3: Doctor/GC telemetry (1.5 hours)

### Week 3: Testing & Validation
- Phase 4: Integration & testing (1.5 hours)

---

## 📚 DOCUMENTATION

### Code Documentation
- NAVMAP headers in all files
- Comprehensive docstrings
- Event payload specifications
- Integration patterns

### User Documentation
- Event reference guide
- Observability setup
- Troubleshooting guide
- Performance tuning

---

## ✅ DELIVERABLES CHECKLIST

- [ ] Phase 1: Event infrastructure reviewed
- [ ] Phase 2: All boundaries instrumented (300+ LOC)
- [ ] Phase 3: Catalog operations telemetry (200+ LOC)
- [ ] Phase 4: CLI integration & tests (200+ LOC)
- [ ] 200+ LOC comprehensive tests
- [ ] Zero linting errors
- [ ] 100% type hints
- [ ] Complete documentation
- [ ] All quality gates met
- [ ] Production ready

---

## 🎓 KEY CONCEPTS

### Event Emission Pattern

```python
# Standard pattern for all operations
emit("operation.phase.status", level="INFO", payload={
    "operation": "...",
    "phase": "...",
    "status": "...",
    # Operation-specific fields
})
```

### Context Correlation

```python
# All events include:
ids={
    "run_id": run_id,           # Trace this run
    "config_hash": config_hash,  # Correlate config
    "boundary": boundary,        # What boundary
    "artifact_id": artifact_id,  # What artifact
}
```

### Error Handling

```python
# Errors include details without sensitive data
emit("operation.error", level="ERROR", payload={
    "error_type": type(e).__name__,
    "error_message": scrub_secrets(str(e)),
    "traceback_lines": 5,  # First 5 lines for debugging
})
```

---

## 🔗 REFERENCES

- Existing event model: `src/DocsToKG/OntologyDownload/observability/events.py`
- Boundary implementations: `src/DocsToKG/OntologyDownload/catalog/boundaries.py`
- Doctor operations: `src/DocsToKG/OntologyDownload/catalog/doctor.py`
- GC operations: `src/DocsToKG/OntologyDownload/catalog/gc.py`
- CLI commands: `src/DocsToKG/OntologyDownload/cli/db_cmd.py`

---

*Implementation Plan: October 21, 2025*  
*Scope: 4-6 hours*  
*Target Quality: 100/100*

