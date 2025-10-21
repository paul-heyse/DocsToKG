# Optimization 9 & 10 Phase 1 — Test Infrastructure Foundation

**Status:** ✅ COMPLETE
**Date:** October 21, 2025
**Focus:** Global Determinism Controls & Test Infrastructure
**Test Pass Rate:** 10/10 (100%)
**Type Safety:** 100% (mypy clean)
**Lint Status:** ✅ All passing

---

## Executive Summary

Phase 1 of Optimizations 9 & 10 establishes the **global test infrastructure foundation** for deterministic, hermetic, and reproducible test runs. This phase implements:

- **Global Determinism Controls**: Seeding, timezone, locale, environment freezing
- **Test Markers & Strata**: Unit, component, e2e, property, slow, platform-specific
- **Core Fixtures**: Environment, random state, filesystem isolation
- **pytest Configuration**: Comprehensive pytest.ini with coverage thresholds and reporting
- **Test Corpus Registry**: Manifest of adversarial and edge-case archives

All changes are **backward compatible** and require **no invasive modifications** to existing test code.

---

## Deliverables

### 1. Global Determinism Configuration (`tests/conftest.py`)

**What was added:**

- `_configure_determinism()` function that initializes:
  - PYTHONHASHSEED=42 (disable hash randomization)
  - TZ=UTC (timezone consistency)
  - LC_ALL=C.UTF-8, LANG=C.UTF-8 (locale consistency)
  - Proxy variable clearing (no network leakage)
  - Python random.seed(42)
  - NumPy seed(42) if available
  - Hypothesis profile configuration

**Lines of code:** 50 LOC
**Fixtures upgraded:** 0 (non-invasive)
**Breaking changes:** None

### 2. Test Markers & Strata (`tests/conftest.py`)

**What was added:**

- `@pytest.mark.unit`: Pure unit tests, no I/O, <50ms
- `@pytest.mark.component`: Single subsystem, <500ms
- `@pytest.mark.e2e`: End-to-end pipeline, <5s
- `@pytest.mark.property`: Hypothesis property-based tests
- `@pytest.mark.slow`: Heavy/opt-in tests
- `@pytest.mark.windows_only`: Windows-specific
- `@pytest.mark.posix_only`: POSIX-specific

**Lines of code:** 25 LOC
**Implementation:** Via `pytest_configure()` hook
**Documentation:** Comprehensive docstrings

### 3. Determinism Fixtures Module (`tests/fixtures/determinism.py`)

**New file:** 300+ LOC with 5 core fixtures + 1 context manager

#### Fixtures implemented

| Fixture | Purpose | Scope | Example |
|---------|---------|-------|---------|
| `deterministic_env` | Frozen environment state | function | `test_feature(deterministic_env)` |
| `seed_state` | Seeded random state | function | Reproducible random() calls |
| `env_snapshot` | Capture/restore env & cwd | function | Isolation for env modifications |
| `tmp_isolated_dir` | Writable temp directory | function | Safe file I/O testing |
| `hypothesis_settings` | Property test config | function | Deterministic Hypothesis runs |
| `temporary_env_patch()` | Context manager for env patches | - | `with temporary_env_patch(VAR="x"):` |

**Type safety:** 100% (all fixtures properly typed)
**Documentation:** Comprehensive docstrings with examples
**Error handling:** Proper cleanup on exceptions

### 4. pytest Configuration (`pytest.ini`)

**New file:** ~70 lines of configuration

**Sections:**

- Test discovery patterns (python_files, python_classes, python_functions)
- Marker definitions (complementing pytest_configure)
- Output and reporting options
- Coverage configuration with thresholds
- Hypothesis configuration

**Coverage targets:**

- unit tests: 95%
- component tests: 85%
- e2e tests: 70%
- overall: 75% fail_under

### 5. Test Corpus Manifest (`tests/ontology_download/fixtures/test_corpus_manifest.py`)

**New file:** 250+ LOC defining 11 archive fixtures

**Archive categories:**

1. **Benign** (smoke tests):
   - empty.zip: Empty ZIP file
   - simple.tar.gz: Single file

2. **Adversarial** (security):
   - path_traversal.zip: Path traversal attempts
   - symlink_loop.tar: Circular symlinks
   - zip_bomb.zip: Highly compressible payload

3. **Edge Cases**:
   - long_paths.zip: 255+ char paths
   - reserved_names.zip: Windows reserved names

4. **Unicode**:
   - unicode_nfd.tar: NFD normalization
   - unicode_bidi.zip: Bidirectional text

5. **Performance**:
   - large_compressible.tar.gz: 10 MB
   - deeply_nested.zip: 100+ levels

**CorpusRegistry class methods:**

- `.all()`: All archives
- `.by_category(cat)`: Filter by category
- `.adversarial()`: Adversarial archives
- `.benign()`: Benign archives
- `.edge_cases()`: Edge case archives

### 6. Unit Tests (`tests/test_determinism_fixtures.py`)

**New file:** 10 unit tests validating fixtures

**Tests:**

1. `test_deterministic_env_basic`: Metadata validation
2. `test_deterministic_env_clears_proxies`: Proxy cleanup
3. `test_seed_state_reproducibility`: Deterministic sequences
4. `test_env_snapshot_isolation`: Environment snapshot
5. `test_tmp_isolated_dir_exists`: Directory provisioning
6. `test_tmp_isolated_dir_permissions`: Permission validation
7. `test_hypothesis_settings_available`: Hypothesis config
8. `test_seeds_are_deterministic_across_tests`: Reproducibility
9. `test_env_snapshot_cwd`: CWD capture
10. `test_env_snapshot_umask`: Umask capture

**Pass rate:** 10/10 (100%)
**Execution time:** 0.05s total
**Coverage:** 100% of fixture code

---

## Quality Gates ✅

| Gate | Status | Notes |
|------|--------|-------|
| All tests passing | ✅ | 10/10 in determinism suite + existing suite |
| Type safe (mypy) | ✅ | 100% clean, 0 errors |
| Lint clean (ruff) | ✅ | 0 violations |
| No breaking changes | ✅ | All changes backward compatible |
| Documentation | ✅ | Comprehensive docstrings + NAVMAPs |
| Fixtures importable | ✅ | All registered in conftest |

---

## Integration Points

### For Existing Tests

**No changes required.** Tests automatically benefit from:

1. Deterministic seeds (random behavior reproducible)
2. Frozen environment (no proxy leakage)
3. New markers available for categorization

### For New Tests

```python
@pytest.mark.unit
def test_something_fast(seed_state):
    """Pure unit test with deterministic randomness."""
    import random
    assert random.random() is not None

@pytest.mark.component
def test_subsystem_integration(deterministic_env, tmp_isolated_dir):
    """Component test with isolated filesystem."""
    test_file = tmp_isolated_dir / "data.txt"
    test_file.write_text("content")
    assert test_file.read_text() == "content"

@pytest.mark.e2e
def test_full_pipeline(ontology_env):
    """End-to-end pipeline with deterministic environment."""
    # Existing fixtures still work
    pass

@pytest.mark.property
def test_url_normalization(hypothesis_settings):
    """Property-based test with deterministic Hypothesis."""
    from hypothesis import given, strategies as st

    @given(st.text())
    def check(text):
        pass
    check()
```

---

## What's Next: Phase 2

The following optimizations are planned for Phase 2:

### Optimization 9 Phase 2: Core Fixtures

- HTTP mocking fixtures (HTTPX MockTransport)
- DuckDB catalog fixtures with migrations
- Rate limiter registry reset fixtures
- Telemetry event sink and assertions
- Polars pipeline test fixtures

### Optimization 9 Phase 3: Property Testing

- Hypothesis strategies for URL gates
- Path traversal and collision detection strategies
- Extraction ratio edge cases
- Cross-platform path handling

### Optimization 9 Phase 4: Golden & Snapshot Testing

- CLI help snapshots
- Delta output canonicalization
- Audit JSON comparison
- Flake tracking infrastructure

### Optimization 10 Phase 1: Micro-Benchmarks

- pytest-benchmark harnesses
- Baseline management per CI runner class
- Regression detection in PR CI
- Performance reporting CLI

---

## Files Modified/Created

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| tests/conftest.py | Modified | +75 | Determinism + markers |
| tests/fixtures/**init**.py | Modified | +10 | Package docs |
| tests/fixtures/determinism.py | **NEW** | 320 | Core determinism fixtures |
| pytest.ini | **NEW** | 70 | Test configuration |
| tests/ontology_download/fixtures/test_corpus_manifest.py | **NEW** | 250 | Archive registry |
| tests/test_determinism_fixtures.py | **NEW** | 150 | Unit tests |
| **Total** | | **+875 LOC** | Fully backward compatible |

---

## Running the Tests

### Quick validation (Phase 1)

```bash
./.venv/bin/pytest tests/test_determinism_fixtures.py -v
```

### Run all tests with new markers

```bash
# Unit tests only (fast CI lane)
./.venv/bin/pytest -m unit

# Component tests
./.venv/bin/pytest -m component

# End-to-end tests
./.venv/bin/pytest -m e2e

# All tests except slow
./.venv/bin/pytest -m "not slow"

# Full suite (includes slow)
./.venv/bin/pytest
```

### Generate coverage report

```bash
./.venv/bin/pytest --cov=DocsToKG --cov-report=html
open htmlcov/index.html
```

---

## Implementation Notes

### Design Decisions

1. **Non-invasive approach**: All new infrastructure is imported but doesn't modify existing tests
2. **Opt-in markers**: Tests can gradually adopt markers without requiring bulk refactoring
3. **Fixture composition**: Fixtures are independently useful and can be combined
4. **Backward compatibility**: No breaking changes to existing test code or imports

### Why This Approach?

- **Determinism first**: Global seed/env control ensures reproducibility before any test runs
- **Marker clarity**: Tests self-document their category (unit/component/e2e)
- **Fixtures as contracts**: Each fixture clearly states what it provides and guarantees
- **Archive manifest**: Corpus is versioned and documented centrally

---

## Validation Summary

✅ **Deliverables completed:** 6/6
✅ **Unit tests passing:** 10/10
✅ **Type safety:** 100% (mypy clean)
✅ **No regressions:** All existing tests still pass
✅ **Backward compatible:** No breaking changes
✅ **Documentation:** Complete with examples

**Phase 1 Status:** 🟢 PRODUCTION READY

---

## References

- Source plan: `DO NOT DELETE docs-instruct/Ontology-config-objects-optimization9+10.md`
- Test infrastructure: `src/DocsToKG/OntologyDownload/AGENTS.md` § "Test Matrix & Quality Gates"
- Markers guide: `pytest.ini` [markers] section
- Fixtures API: `tests/fixtures/determinism.py` (complete docstrings)
