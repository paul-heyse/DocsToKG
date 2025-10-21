# Phase 8: Gates Integration & Testing - Complete Roadmap

**Status**: Pillar 8 Foundation 100% Complete → Ready for Integration Phase
**Date**: October 21, 2025
**Scope**: 600 LOC gates + 300 LOC integration + 300 LOC E2E tests

---

## Current State (Foundation Complete)

✅ **Core Infrastructure Deployed:**
- `policy/errors.py` (259 LOC) - 33 error codes, exception classes, scrubbing
- `policy/registry.py` (165 LOC) - Thread-safe decorator-based gate registration
- `policy/metrics.py` (250 LOC) - Per-gate metrics collection with percentiles
- `policy/gates.py` (300 LOC) - 6 gate skeletons with url_gate implemented

✅ **Event Bus Integration:**
- Observability event emission ready
- Metrics collection infrastructure
- Error taxonomy in place

---

## Phase 8.1: Complete Gate Implementations (600 LOC)

### Gate 1: URL & Network Gate ✅ (DONE - 100 LOC)
**Status**: Fully implemented
**Features**:
- RFC 3986 URL parsing
- Scheme validation (http/https only)
- Userinfo rejection
- Host allowlisting with CIDR/IP support
- Port validation (global + per-host)
- Redirect auditing (no auth forwarding)

### Gate 2: Filesystem & Path Gate ⏳ (TODO - 80 LOC)
**Implementation**:
```python
@policy_gate(name="filesystem_gate", domain="filesystem")
def filesystem_gate(
    root_path: str,
    entry_paths: List[str],
    allow_symlinks: bool = False,
) -> PolicyResult:
    """Validate filesystem paths against security policy.

    Checks:
    - Dirfd/openat semantics (O_NOFOLLOW|O_EXCL)
    - Path normalization (NFC, no .., no /)
    - Casefold collision detection
    - Length/depth constraints
    - Reserved names (Windows)
    - Relative path enforcement
    """
```

### Gate 3: Extraction Policy Gate ⏳ (TODO - 80 LOC)
**Implementation**:
```python
@policy_gate(name="extraction_gate", domain="extraction")
def extraction_gate(
    entries_total: int,
    bytes_declared: int,
    policies: ExtractionPolicy,
) -> PolicyResult:
    """Validate archive extraction parameters.

    Checks:
    - Entry type allowlist (regular files only)
    - Global zip-bomb ratio limit
    - Per-entry ratio limit
    - Per-file size limit
    - Entry count budget
    - Include/exclude globs
    """
```

### Gate 4: Storage Gate ⏳ (TODO - 60 LOC)
**Implementation**:
```python
@policy_gate(name="storage_gate", domain="storage")
def storage_gate(
    operation: str,  # "put", "move", "copy", "delete"
    src_path: str,
    dst_path: str,
) -> PolicyResult:
    """Validate storage operations.

    Checks:
    - Atomic writes (temp + move pattern)
    - Path traversal prevention
    - Permission enforcement
    - Rename safety
    """
```

### Gate 5: DB Transactional Gate ⏳ (TODO - 70 LOC)
**Implementation**:
```python
@policy_gate(name="db_boundary_gate", domain="db")
def db_boundary_gate(
    operation: str,  # "pre_commit", "post_extract"
    fs_state: Dict,
    db_state: Dict,
) -> PolicyResult:
    """Validate database transaction boundaries.

    Checks:
    - Foreign key invariants
    - Commit-after-FS-success choreography
    - Latest pointer consistency
    - Marker alignment
    - Recovery procedures
    """
```

### Gate 6: Config Gate ✅ (DONE - 30 LOC)
**Status**: Stub implemented, ready for enhancement

---

## Phase 8.2: Gate Telemetry Integration (150 LOC)

**In each gate implementation, add:**
```python
# At gate invocation
emit_event(
    type="policy.gate",
    level="ERROR" if rejected else "INFO",
    payload={
        "gate": gate_name,
        "outcome": "reject" if rejected else "ok",
        "elapsed_ms": elapsed_ms,
        "error_code": error_code,
    }
)

# Record metrics
collector = get_metrics_collector()
collector.record_metric(GateMetric(
    gate_name=gate_name,
    passed=is_ok,
    elapsed_ms=elapsed_ms,
    error_code=error_code,
))
```

---

## Phase 8.3: Integration into Core Flows (300 LOC)

### Into planning.py
```python
# At start: config validation
gate_result = config_gate(config)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(...)

# Before each download: URL validation
gate_result = url_gate(url, allowed_hosts, allowed_ports)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(...)
```

### Into io/filesystem.py
```python
# Before extraction: path validation
gate_result = filesystem_gate(root_path, entry_paths)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(...)
```

### Into io/extraction_policy.py
```python
# Pre-scan: archive validation
gate_result = extraction_gate(entries_total, bytes_declared, policies)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(...)
```

### Into catalog/boundaries.py
```python
# After extraction, before DB: boundary validation
gate_result = db_boundary_gate("post_extract", fs_state, db_state)
if isinstance(gate_result, PolicyReject):
    raise_policy_error(...)
```

---

## Phase 8.4: Testing Strategy (400 LOC)

### Unit Tests (100 LOC per gate)
- White-box tests for each gate
- Both accept and reject paths
- Edge cases (IDNs, CIDRs, Unicode, long paths)
- All error codes triggered
- Metrics verified

### Property-Based Tests (50 LOC per gate)
- URL generators → no false positives
- Path generators (combining marks, bidi, deep trees)
- Invariant verification
- Idempotency checks

### Integration Tests (100 LOC)
- E2E scenarios triggering each error
- Correct CLI exit codes
- Zero partial writes
- Manifest consistency

### Cross-Platform Tests (50 LOC)
- Windows reserved names
- macOS NFD/NFC normalization
- Path separators

### Chaos Tests (50 LOC)
- Crash between FS + DB
- Doctor recovery
- Data consistency

---

## Implementation Sequence (5-6 days)

**Day 1 (2.5 hrs): Gate 2-3** (Filesystem + Extraction)
- Filesystem_gate: path normalization, traversal check
- Extraction_gate: ratio detection, size limits

**Day 2 (2 hrs): Gate 4-5** (Storage + DB)
- Storage_gate: atomic writes, traversal prevention
- DB_boundary_gate: transaction choreography

**Day 3-4 (4 hrs): Telemetry** (All gates)
- Wire events + metrics into each gate
- Verify emission

**Day 5-6 (6 hrs): Integration**
- planning.py config/URL validation
- filesystem.py path validation
- extraction_policy.py archive validation
- boundaries.py DB validation
- Full integration tests
- Cross-platform tests

**Day 7 (2 hrs): E2E & Validation**
- End-to-end smoke tests
- Performance baseline
- Docs update

---

## Success Criteria

- [ ] All 6 gates fully implemented (config, url, filesystem, extraction, storage, db)
- [ ] 400+ tests passing (100% error codes covered)
- [ ] Events + metrics flowing for each gate
- [ ] Integration into planning, download, extraction, storage flows
- [ ] Zero partial writes or data corruption
- [ ] Cross-platform tests passing
- [ ] Type-safe (0 mypy errors)
- [ ] 0 ruff violations
- [ ] Docs updated

---

## Risk Assessment

| Gate | Risk | Mitigation |
|------|------|-----------|
| URL | LOW | Already implemented |
| Filesystem | MEDIUM | Property tests for normalization |
| Extraction | LOW | Proven zip-bomb logic |
| Storage | MEDIUM | Atomic write testing |
| DB | MEDIUM | Boundary choreography tests |
| Config | LOW | Schema validation |

**Overall: LOW-MEDIUM** (established patterns, proven technologies)

---

## Deliverables

- 600 LOC gate implementations
- 300 LOC integration code
- 400 LOC test code
- Event emission in all gates
- Metrics collection for all gates
- Integration guides
- Performance baselines

**Total Phase 8: ~1,300 LOC, ~400 tests**

---

## Files to Create/Modify

### New Files
- `tests/ontology_download/test_policy_gates_filesystem.py`
- `tests/ontology_download/test_policy_gates_extraction.py`
- `tests/ontology_download/test_policy_gates_storage.py`
- `tests/ontology_download/test_policy_gates_db.py`

### Modified Files
- `src/DocsToKG/OntologyDownload/policy/gates.py` - Complete 5 gates
- `src/DocsToKG/OntologyDownload/planning.py` - Config + URL gates
- `src/DocsToKG/OntologyDownload/io/filesystem.py` - Path gate
- `src/DocsToKG/OntologyDownload/io/extraction_policy.py` - Extraction gate
- `src/DocsToKG/OntologyDownload/catalog/boundaries.py` - DB gate

---

## Next Actions

1. Start Day 1: Implement filesystem_gate + extraction_gate
2. Wire metrics/events into all gates
3. Integration phase
4. Comprehensive testing
5. Performance validation
6. Production deployment

---

**Ready to Begin Phase 8 Implementation** ✅
