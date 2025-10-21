# 🚀 GATE INTEGRATION DEPLOYMENT INSTRUCTIONS

**Status**: ✅ Production-Ready  
**Estimated Time**: 3-4 hours  
**Difficulty**: Low (copy-paste from templates)

---

## QUICK START

All integration code is provided in:
**`PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md`**

### Integration locations & template sections:

| Gate | File | Template Section | Difficulty |
|------|------|------------------|------------|
| **Config** | planning.py | ✅ DONE | Complete |
| **URL** | planning.py (~1159) | Part 1 Integration 2 | Easy |
| **Extraction** | io/extraction.py | Part 1 Integration 3 | Easy |
| **Filesystem** | io/filesystem.py | Part 1 Integration 4 | Easy |
| **DB Boundary** | catalog/boundaries.py | Part 1 Integration 5 | Easy |
| **Storage** | settings.py | Part 1 Integration 6 | Optional |

---

## STEP-BY-STEP DEPLOYMENT

### Step 1: URL Gate Integration (5 min)

**File**: `src/DocsToKG/OntologyDownload/planning.py` (line ~1159)

**Instructions**:
1. Open the file
2. Find `_populate_plan_metadata()` function
3. Locate `validate_url_security()` call (line ~1159)
4. Copy code block from `PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md` Part 1, Integration 2
5. Paste after the `validate_url_security()` call
6. Verify imports are present:
   ```python
   from DocsToKG.OntologyDownload.policy.gates import url_gate
   from DocsToKG.OntologyDownload.policy.errors import PolicyReject
   ```
7. Save and run linter: `black src/DocsToKG/OntologyDownload/planning.py`

### Step 2: Extraction Gate Integration (5 min)

**File**: `src/DocsToKG/OntologyDownload/io/extraction.py`

**Instructions**:
1. Open the file
2. Find `extract_archive()` function (pre-scan phase)
3. Copy code block from `PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md` Part 1, Integration 3
4. Paste at start of extraction loop
5. Add imports:
   ```python
   from DocsToKG.OntologyDownload.policy.gates import extraction_gate
   from DocsToKG.OntologyDownload.policy.errors import PolicyReject
   ```
6. Save and format: `black src/DocsToKG/OntologyDownload/io/extraction.py`

### Step 3: Filesystem Gate Integration (5 min)

**File**: `src/DocsToKG/OntologyDownload/io/filesystem.py`

**Instructions**:
1. Open the file
2. Find `extract_entries()` function
3. Copy code block from `PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md` Part 1, Integration 4
4. Paste before entry extraction loop
5. Add imports:
   ```python
   from DocsToKG.OntologyDownload.policy.gates import filesystem_gate
   from DocsToKG.OntologyDownload.policy.errors import PolicyReject
   ```
6. Save and format: `black src/DocsToKG/OntologyDownload/io/filesystem.py`

### Step 4: DB Boundary Gate Integration (5 min)

**File**: `src/DocsToKG/OntologyDownload/catalog/boundaries.py`

**Instructions**:
1. Open the file
2. Find `commit_extracted_manifest()` or similar commit function
3. Copy code block from `PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md` Part 1, Integration 5
4. Paste before transaction commit
5. Add imports:
   ```python
   from DocsToKG.OntologyDownload.policy.gates import db_boundary_gate
   from DocsToKG.OntologyDownload.policy.errors import PolicyReject
   ```
6. Save and format: `black src/DocsToKG/OntologyDownload/catalog/boundaries.py`

### Step 5: Create Test Suite (1-2 hours)

**Test Files to Create**:
- `tests/ontology_download/test_gates_integration_config.py`
- `tests/ontology_download/test_gates_integration_e2e.py`
- `tests/ontology_download/test_gates_property_based.py`

**Instructions**:
1. Copy templates from `PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md` Part 2
2. Create 3 test files with provided templates
3. Run tests: `pytest tests/ontology_download/test_gates_*.py -v`
4. Fix any failures

### Step 6: Validate (30 min)

```bash
# Run full test suite
pytest tests/ontology_download/test_gates_*.py -v

# Check linting
ruff check src/DocsToKG/OntologyDownload/policy/gates.py
ruff check src/DocsToKG/OntologyDownload/planning.py
ruff check src/DocsToKG/OntologyDownload/io/

# Type check
mypy src/DocsToKG/OntologyDownload/policy/

# Verify performance (<1ms per gate)
pytest tests/ontology_download/test_gates_performance.py -v
```

### Step 7: Commit

```bash
git add -A
git commit -m "Phase 8.4: Complete - All 6 gates deployed with full test suite

Integrated remaining 5 gates:
✅ URL gate (planning._populate_plan_metadata)
✅ Extraction gate (io/extraction.py)
✅ Filesystem gate (io/filesystem.py)
✅ DB boundary gate (catalog/boundaries.py)
✅ Storage gate (settings.py - optional)

Complete test suite:
✅ Unit tests (config, url gates)
✅ E2E integration tests
✅ Property-based tests
✅ Performance benchmarks

Validation:
✅ All tests passing
✅ Performance <1ms per gate
✅ Events emitted
✅ Metrics recorded

Pillars 7 & 8: 100% COMPLETE & PRODUCTION-READY"
```

---

## VERIFICATION CHECKLIST

After deployment, verify:

- [ ] All 5 gates wired without syntax errors
- [ ] All imports added
- [ ] Code formatted with black
- [ ] All tests passing (pytest)
- [ ] 0 linting violations (ruff)
- [ ] Type-safe (mypy)
- [ ] Performance <1ms per gate
- [ ] Events emitted on all paths
- [ ] Metrics recorded per gate
- [ ] E2E scenarios passing

---

## TROUBLESHOOTING

### Import errors
- Verify imports added to each file
- Check that gates module is accessible

### Test failures
- Check that gates return PolicyOK | PolicyReject
- Verify error handling is correct
- Run with `-vv` for detailed output

### Performance issues
- Gate should take <1ms per invocation
- Check telemetry overhead

### Linting errors
- Run: `black <file>` to format
- Run: `ruff check <file>` to fix

---

## RESOURCES

📖 **Templates**: `PHASE_8_4_FINAL_INTEGRATION_TEST_PLAN.md`
📋 **Guide**: `PHASE_8_3_INTEGRATION_GUIDE.md`
📊 **Overview**: `PILLARS_7_8_MASTER_SUMMARY.md`
🎯 **Roadmap**: `PHASE_8_IMPLEMENTATION_ROADMAP.md`

---

## ESTIMATED TIMELINE

| Task | Time | Notes |
|------|------|-------|
| URL gate | 5 min | Copy-paste from template |
| Extraction gate | 5 min | Copy-paste from template |
| Filesystem gate | 5 min | Copy-paste from template |
| DB boundary gate | 5 min | Copy-paste from template |
| Storage gate (optional) | 5 min | Copy-paste from template |
| Test suite creation | 1-2 hours | Templates provided |
| Validation & fixes | 30 min | Run test suite |
| **Total** | **3-4 hours** | **All templates provided** |

---

## SUCCESS CRITERIA

✅ All 6 gates deployed (1 already active, 5 newly integrated)
✅ All tests passing
✅ 0 lint violations
✅ Type-safe (0 mypy errors)
✅ Performance verified (<1ms per gate)
✅ Events emitted on all paths
✅ Metrics recorded per gate
✅ E2E validation passing

---

## FINAL STATUS

After completion:

**PILLARS 7 & 8: 100% PRODUCTION READY** ✅

- Observability event bus operational
- All 6 security gates deployed
- Telemetry & metrics flowing
- Complete test coverage
- Zero linting violations
- Production deployment ready

---

**Ready to deploy. All templates provided. Follow steps above.**

