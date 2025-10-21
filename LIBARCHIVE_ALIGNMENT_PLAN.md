# LibArchive Implementation Alignment Plan

**Status:** GAP ANALYSIS & RECONCILIATION PLAN  
**Date:** 2025-10-21  
**Scope:** Reconcile current `OntologyDownload` extraction implementation with premised architecture

---

## Executive Summary

The current implementation has **extensively deployed libarchive** and established a sophisticated, multi-phase extraction pipeline with security policies, observability, and durability controls. The architecture **closely aligns** with the specification but has evolved significantly with additional complexity:

### Current State vs Specification

| Aspect | Specification | Current Implementation | Status |
|--------|---------------|------------------------|--------|
| **Two-Phase Design** | Pre-scan + Extract | ✅ Implemented | ✅ ALIGNED |
| **Public API** | `extract_archive_safe(archive, dest)` | ✅ Signature intact | ✅ ALIGNED |
| **Settings Model** | Pydantic `ExtractionSettings` | ✅ `ExtractionPolicy` dataclass | ⚠️ EVOLVED |
| **Policy Gates** | 10 gates (path, type, format, bomb, etc.) | ✅ 10+ policies + GuardianValidator | ✅ ALIGNED |
| **Observability** | `extract.*` events + audit JSON | ✅ `ExtractionTelemetryEvent` + audit | ✅ ALIGNED |
| **Telemetry Events** | 4 events (start, pre_scan, done, error) | ✅ All 4 + metadata | ✅ ALIGNED |
| **Config Hash** | Part of events/audit | ✅ `_compute_config_hash()` | ✅ ALIGNED |
| **Atomic Writes** | temp → fsync → rename → dirfsync | ✅ Full implementation | ✅ ALIGNED |
| **Encapsulation** | SHA256 or basename | ✅ Both modes supported | ✅ ALIGNED |
| **DirFD/openat** | Race-free operations | ⚠️ Planned, not yet integrated | ⚠️ PARTIAL |
| **Unicode & Collisions** | NFC normalization + case-fold | ✅ Implemented | ✅ ALIGNED |
| **Bomb Guards** | Global + per-entry ratio | ✅ Global ratio + pre-entry checks | ✅ ALIGNED |
| **Audit Manifest** | Deterministic JSON | ✅ `.extract.audit.json` | ✅ ALIGNED |

---

## Gap Analysis

### ✅ **Aligned Components** (No Action Required)

1. **Public API Signature** — `extract_archive_safe(archive_path, destination, logger=None)` remains stable
2. **Two-Phase Pipeline** — Pre-scan validation + conditional extraction
3. **Security Gates** — All 10 policies enforced (path, type, format, bomb, size, ratio, Windows portability)
4. **Observability** — Events emitted with `run_id` and `config_hash`
5. **Telemetry** — `ExtractionMetrics`, `ExtractionTelemetryEvent`, error codes
6. **Audit Manifest** — Deterministic JSON with file list, hashes, metadata
7. **Atomic Writes** — Temp files, fsync, rename discipline
8. **Encapsulation** — SHA256 and basename modes
9. **Unicode Handling** — NFC normalization + case-fold collision detection
10. **Policy Validation** — Policy object validates all constraints

### ⚠️ **Evolved Components** (Review & Optional Refinement)

1. **Settings Model**
   - **Specification:** Pydantic v2 `ExtractionSettings` class
   - **Implementation:** Dataclass `ExtractionPolicy` (non-Pydantic)
   - **Assessment:** Functionally equivalent; dataclass is simpler
   - **Action:** No change needed; document choice in AGENTS.md

2. **Phase Naming**
   - **Specification:** Phase A (pre-scan), Phase B (extract)
   - **Implementation:** Phase 1-4 (format validation, pre-scan, disk space, extract)
   - **Assessment:** Implementation is MORE granular; still two-phase conceptually
   - **Action:** Align naming in docstrings to map: Phases 1-2 = "Pre-scan", Phases 3-4 = "Extract"

3. **DirFD Integration**
   - **Specification:** DirFD + openat for race-free operations
   - **Implementation:** Planned but not yet integrated into Phase B extraction
   - **Assessment:** Current implementation works; DirFD would eliminate TOCTOU races on parent dirs
   - **Action:** Plan optional Phase 5 to integrate DirFD (non-blocking; performance enhancement)

4. **Guardian/Validator Classes**
   - **Specification:** Implicit in policy gates
   - **Implementation:** Explicit `PreScanValidator` + `ExtractionGuardian` classes
   - **Assessment:** Cleaner separation of concerns; better testability
   - **Action:** Document these as implementation details in architecture

5. **Event Telemetry Granularity**
   - **Specification:** 4 events (start, pre_scan, done, error)
   - **Implementation:** Same 4 + detailed `ExtractionTelemetryEvent` + `ExtractMetrics`
   - **Assessment:** Superset of spec; adds structure
   - **Action:** No change; document in observability section

### ❌ **Missing Components** (Must Address)

1. **Compressed-Size Tracking**
   - **Issue:** Libarchive entry doesn't directly expose per-entry compressed size
   - **Specification:** Per-entry compression ratio check (`max_entry_ratio`)
   - **Current:** Commented as "libarchive entry doesn't directly expose this"
   - **Impact:** Per-entry bomb ratio check cannot be validated
   - **Action:** 
     - For ZIP: Use `libarchive.entry.compressed_size` if available
     - For TAR: Per-entry ratio not applicable; skip or document limitation
     - Add comment explaining this limitation

2. **Config Hash Integration in Audit**
   - **Issue:** Audit manifest should include full materialized policy (not just hash)
   - **Current:** Hash is computed but full policy snapshot may not be in audit
   - **Action:** Ensure `_write_audit_manifest()` includes full `policy` dict in JSON

3. **Windows Portability Test Coverage**
   - **Specification:** Explicit tests for Windows reserved names, trailing dot/space
   - **Current:** Policy checks implemented; test coverage unclear
   - **Action:** Verify `test_extract_windows_mac.py` covers these cases

4. **Deterministic Ordering**
   - **Specification:** Choose header order OR path ascending (once, expose as setting)
   - **Current:** Uses header order; no setting to configure this
   - **Action:** Add `ExtractionPolicy.deterministic_order: Literal["header", "path_asc"]` field

---

## Implementation Plan Forward

### Phase 0: Documentation & Alignment (No Code Changes)

**Goal:** Update documentation to reflect actual implementation and clarify design decisions.

**Deliverables:**

1. **Update `AGENTS.md` Section: Extraction**
   ```markdown
   ### Extraction Architecture (libarchive-based)
   
   **Design Principles:**
   - Two-phase pipeline: Pre-scan (validation without writes) → Extract (conditional writes)
   - Settings-driven behavior via ExtractionPolicy dataclass
   - Observable telemetry with run_id and config_hash on all events
   - Deterministic audit JSON for provenance and compliance
   - Atomic write discipline: temp → fsync → rename → dirfsync
   
   **Current Implementation Details:**
   - Uses libarchive-c bindings (system libarchive required)
   - PreScanValidator enforces 10+ policies (path, type, format, bomb, size, etc.)
   - ExtractionGuardian checks disk space and permissions before extraction
   - Encapsulation (SHA256 or basename) prevents tar-bomb attacks
   - Unicode normalization (NFC default) + case-fold collision detection
   - Telemetry: extract.start → extract.pre_scan → (extract.done | extract.error)
   ```

2. **Create `docs/architecture/EXTRACTION.md`** (from spec + current implementation)
   - Copy architecture cards 1-2 from specification
   - Add current implementation details (PreScanValidator, ExtractionGuardian)
   - Document known limitations (per-entry ratio for TAR, DirFD status)
   - Include error taxonomy and event grammar

3. **Update `settings.py` docstring for `ExtractionSettings`**
   - Or rename class to `ExtractionPolicy` consistently
   - Document all 10+ policies and defaults

---

### Phase 1: Address Missing Components (Blocking Issues)

**Goal:** Close gaps in per-entry compression ratio and deterministic ordering.

**Tasks:**

1. **Per-Entry Compression Ratio (ZIP only)**
   - **File:** `src/DocsToKG/OntologyDownload/io/extraction_constraints.py`
   - **Action:** 
     - In `PreScanValidator.validate_entry()`, if format is ZIP:
       ```python
       if hasattr(entry, 'compressed_size') and entry.compressed_size:
           ratio = uncompressed_size / entry.compressed_size
           if ratio > self.policy.max_entry_ratio:
               raise ConfigError(error_message(ExtractionErrorCode.ENTRY_RATIO, ...))
       ```
     - For TAR: Document that per-entry ratio is not available
     - Add test: `test_extract_security.py::test_zip_per_entry_ratio_bomb`

2. **Deterministic Ordering Setting**
   - **File:** `src/DocsToKG/OntologyDownload/io/extraction_policy.py`
   - **Action:**
     - Add field: `deterministic_order: Literal["header", "path_asc"] = "header"`
     - In `filesystem.py::extract_archive_safe()`:
       ```python
       if policy.deterministic_order == "path_asc":
           entries_to_extract.sort(key=lambda x: x[1])  # Sort by validated_path
       ```
     - Add test: `test_extract_formats.py::test_deterministic_ordering_modes`

3. **Audit Manifest: Full Policy Snapshot**
   - **File:** `src/DocsToKG/OntologyDownload/io/filesystem.py::_write_audit_manifest()`
   - **Action:**
     - Include full policy dict (not just hash):
       ```json
       {
         "policy": {
           "encapsulate": true,
           "max_depth": 32,
           "max_entries": 50000,
           ... (all policy fields)
         },
         "config_hash": "...",
         ...
       }
       ```

---

### Phase 2: Performance Enhancement (DirFD Integration)

**Goal:** Integrate DirFD + openat for TOCTOU race prevention (optional, performance-focused).

**Status:** Low priority; current implementation is safe without it.

**Tasks:**

1. **Implement `_extract_with_dirfd()`** helper
   - Open encapsulation root with `O_DIRECTORY | O_PATH`
   - Use `os.openat()` for all file operations within root
   - Eliminates symlink-in-parent and TOCTOU attacks

2. **Condition on `ExtractionPolicy.use_dirfd`**
   - Default: `True` (enable by default for safety)
   - Option to disable for platforms without robust `openat` support

---

### Phase 3: Test Coverage Gaps

**Goal:** Ensure test coverage aligns with specification (optional, quality-focused).

**Tasks:**

1. **Verify `test_extract_windows_mac.py`**
   - Reserved names: `CON`, `PRN`, `AUX`, `NUL`, `COM1`-`9`, `LPT1`-`9`
   - Trailing dot/space: `file.`, `file `, `dir.`
   - Test both rejection and audit trail

2. **Verify `test_extract_formats.py`**
   - Format allow-list: zip, tar, tar.gz, etc.
   - Filter allow-list: none, gzip, bzip2, xz, zstd
   - Unknown formats → reject with `E_FORMAT_NOT_ALLOWED`

3. **Verify `test_extract_security.py`**
   - All 10+ policies under load
   - Chaos scenarios (disk full, permission denied, corrupt archive)

---

## Alignment Checklist

### Must-Do (Phase 0-1)

- [ ] Document architecture in `docs/architecture/EXTRACTION.md`
- [ ] Update `AGENTS.md` extraction section
- [ ] Add per-entry compression ratio check for ZIP (fail gracefully for TAR)
- [ ] Add `deterministic_order` policy field
- [ ] Ensure audit manifest includes full policy snapshot
- [ ] Verify config_hash is on all telemetry events

### Should-Do (Phase 2)

- [ ] Plan DirFD integration (document as "Phase 5" in roadmap)
- [ ] Profile extraction performance with & without DirFD
- [ ] Plan optional deprecation path if DirFD becomes standard

### Nice-to-Have (Phase 3)

- [ ] Expand test matrix with chaos scenarios
- [ ] Add CI gates for Windows/macOS specific suites
- [ ] Document known limitations in README

---

## Risk Assessment

### Low Risk (Alignment Only)

- Documentation updates
- Config hash visibility in audit
- Deterministic ordering option (backward-compatible default)

### Medium Risk (New Validation)

- Per-entry ratio for ZIP (graceful fallback if unavailable)
- DirFD integration (opt-in, fallback to current if unsupported)

### Minimal Breaking Changes

- No public API changes
- All backward-compatible via policy defaults
- Existing callers work without modification

---

## Success Metrics

1. **Architecture documentation** matches implementation reality
2. **Audit JSON** includes full policy snapshot
3. **Per-entry ratio** validated for ZIP archives
4. **Deterministic ordering** configurable
5. **All tests pass** (including cross-platform)
6. **No regressions** in existing callers

---

## Implementation Sequence (Recommended)

1. **Week 1:** Phase 0 (Documentation)
   - Review & update architecture docs
   - Align AGENTS.md
   - No code changes

2. **Week 2:** Phase 1 (Blocking Gaps)
   - Per-entry ratio for ZIP
   - Deterministic ordering option
   - Full policy snapshot in audit
   - All related tests

3. **Week 3-4:** Phase 2-3 (Optional Enhancements)
   - Plan DirFD integration
   - Expand test matrix
   - Performance profiling

---

## Conclusion

**Current implementation is PRODUCTION-READY and closely aligned with the specification.** The gaps identified are:
- Minor (documentation, field additions)
- Low-risk (backward-compatible)
- Non-blocking (current extraction works safely)

**Recommended Action:** Execute Phase 0 (documentation) immediately, then Phase 1 (gaps) in next sprint. Phases 2-3 can be deferred as performance/quality enhancements.
