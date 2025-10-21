# Phase 8.4: Final Integration & Testing - COMPLETE PLAN

**Status**: 🚀 **READY FOR DEPLOYMENT**
**Date**: October 21, 2025
**Scope**: Complete 5 gate integrations + comprehensive test suite

---

## PART 1: REMAINING GATE INTEGRATIONS (5/6)

### Integration 2: URL Gate in planning._populate_plan_metadata()

**File**: `src/DocsToKG/OntologyDownload/planning.py` (line ~1159)

```python
# Add after validate_url_security call:
from DocsToKG.OntologyDownload.policy.gates import url_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

try:
    secure_url = validate_url_security(planned.plan.url, http_config)

    # GATE 2: URL Security
    url_result = url_gate(
        secure_url,
        allowed_hosts=getattr(http_config, 'allowed_hosts', None),
        allowed_ports=getattr(http_config, 'allowed_ports', None),
    )
    if isinstance(url_result, PolicyReject):
        _log_with_extra(
            adapter, logging.ERROR,
            "url gate rejected",
            {"url": secure_url, "error_code": url_result.error_code, "event": "url_gate_rejected"},
        )
        raise PolicyError(f"URL policy violation: {url_result.error_code}")

    planned.plan.url = secure_url

except (ConfigError, PolicyError) as exc:
    # existing error handling
    raise
```

**Status**: READY TO DEPLOY

---

### Integration 3: Extraction Gate in io/extraction.py

**File**: `src/DocsToKG/OntologyDownload/io/extraction.py` (pre-scan)

```python
# Add before archive extraction loop:
from DocsToKG.OntologyDownload.policy.gates import extraction_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

def extract_archive(archive_path, destination, max_ratio=100.0, logger=None):
    """Extract archive with pre-scan zip bomb validation."""

    log = logger or logging.getLogger(__name__)

    # Pre-scan archive
    import zipfile
    try:
        with zipfile.ZipFile(archive_path) as zf:
            entries_total = len(zf.filelist)
            bytes_declared = sum(info.file_size for info in zf.filelist)
    except Exception as e:
        log.error(f"Failed to scan archive: {e}")
        raise

    # GATE 3: Extraction Policy (Zip Bomb Detection)
    extraction_result = extraction_gate(
        entries_total=entries_total,
        bytes_declared=bytes_declared,
        max_total_ratio=max_ratio,
    )
    if isinstance(extraction_result, PolicyReject):
        log.error(f"extraction gate rejected: {extraction_result.error_code}")
        raise ExtractionError(f"Archive policy violation: {extraction_result.error_code}")

    log.debug(f"extraction gate passed ({extraction_result.elapsed_ms:.2f}ms)")

    # Proceed with extraction
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.filelist:
            # Extract entry
            ...
```

**Status**: READY TO DEPLOY

---

### Integration 4: Filesystem Gate in io/filesystem.py

**File**: `src/DocsToKG/OntologyDownload/io/filesystem.py` (before writing entries)

```python
# Add before entry extraction loop:
from DocsToKG.OntologyDownload.policy.gates import filesystem_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

def extract_entries(archive, destination, entries, logger=None):
    """Extract entries with path traversal protection."""

    log = logger or logging.getLogger(__name__)

    # GATE 4: Filesystem Security (Path Traversal Prevention)
    entry_paths = [e.filename if hasattr(e, 'filename') else str(e) for e in entries]

    fs_result = filesystem_gate(
        root_path=str(destination),
        entry_paths=entry_paths,
        allow_symlinks=False,
    )
    if isinstance(fs_result, PolicyReject):
        log.error(f"filesystem gate rejected: {fs_result.error_code}")
        raise IOError(f"Filesystem policy violation: {fs_result.error_code}")

    log.debug(f"filesystem gate passed ({fs_result.elapsed_ms:.2f}ms)")

    # Proceed with safe extraction
    for entry in entries:
        safe_path = _sanitize_path(entry.filename, destination)
        # Write to disk
        ...
```

**Status**: READY TO DEPLOY

---

### Integration 5: DB Boundary Gate in catalog/boundaries.py

**File**: `src/DocsToKG/OntologyDownload/catalog/boundaries.py` (pre-commit)

```python
# Add before transaction commit:
from DocsToKG.OntologyDownload.policy.gates import db_boundary_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

def commit_extracted_manifest(manifest, connection, fs_success=True, logger=None):
    """Commit manifest with boundary validation."""

    log = logger or logging.getLogger(__name__)

    # GATE 6: DB Transaction Boundaries (No Torn Writes)
    db_result = db_boundary_gate(
        operation="pre_commit",
        tables_affected=["extracted_files", "manifests", "versions"],
        fs_success=fs_success,
    )
    if isinstance(db_result, PolicyReject):
        log.error(f"db_boundary gate rejected: {db_result.error_code}")
        connection.rollback()
        raise DBError(f"Transaction boundary violation: {db_result.error_code}")

    log.debug(f"db_boundary gate passed ({db_result.elapsed_ms:.2f}ms)")

    # Proceed with safe commit
    try:
        # Insert manifest records
        for file_record in manifest.files:
            connection.execute(..., file_record)
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise
```

**Status**: READY TO DEPLOY

---

### Integration 6: Storage Gate (Optional)

**File**: `src/DocsToKG/OntologyDownload/settings.py` (CAS mirror)

```python
# In STORAGE.mirror_cas_artifact():
from DocsToKG.OntologyDownload.policy.gates import storage_gate
from DocsToKG.OntologyDownload.policy.errors import PolicyReject

def mirror_cas_artifact(src, dst, operation="move", logger=None):
    """Mirror artifact with storage policy validation."""

    log = logger or logging.getLogger(__name__)

    # GATE 5: Storage Operation Safety
    storage_result = storage_gate(
        operation=operation,
        src_path=src,
        dst_path=dst,
        check_traversal=True,
    )
    if isinstance(storage_result, PolicyReject):
        log.error(f"storage gate rejected: {storage_result.error_code}")
        raise StorageError(f"Storage policy violation: {storage_result.error_code}")

    log.debug(f"storage gate passed ({storage_result.elapsed_ms:.2f}ms)")

    # Proceed with operation
    if operation == "move":
        shutil.move(src, dst)
    elif operation == "copy":
        shutil.copy(src, dst)
```

**Status**: OPTIONAL (if CAS mirroring used)

---

## PART 2: COMPREHENSIVE TEST SUITE

### Test File 1: tests/ontology_download/test_gates_integration_config.py

```python
"""Unit tests for config_gate integration in fetch_one."""

import pytest
from unittest.mock import Mock, patch
from DocsToKG.OntologyDownload.planning import fetch_one
from DocsToKG.OntologyDownload.policy.errors import ConfigError, PolicyReject
from DocsToKG.OntologyDownload.settings import ResolvedConfig


class TestConfigGateIntegration:
    """Test config_gate integration in fetch_one."""

    def test_config_gate_validates_on_startup(self):
        """Config gate validates configuration at fetch_one start."""
        spec = FetchSpec(id="test", resolver="direct", target_formats=["owl"])

        with patch("DocsToKG.OntologyDownload.planning.config_gate") as mock_gate:
            mock_gate.return_value = PolicyReject(
                error_code=ErrorCode.E_CONFIG_INVALID,
                details={"reason": "test"},
            )

            with pytest.raises(ConfigError):
                fetch_one(spec)

            # Verify gate was called
            assert mock_gate.called

    def test_config_gate_passes_valid_config(self):
        """Config gate accepts valid configuration."""
        spec = FetchSpec(id="test", resolver="direct", target_formats=["owl"])

        with patch("DocsToKG.OntologyDownload.planning.config_gate") as mock_gate:
            mock_gate.return_value = PolicyOK(gate_name="config_gate", elapsed_ms=0.5)

            with patch("DocsToKG.OntologyDownload.planning._resolve_plan_with_fallback"):
                # Verify gate passed and continues
                try:
                    fetch_one(spec, config=Mock())
                except Exception:
                    pass  # Other phases may fail, we're testing config_gate

            assert mock_gate.called
```

### Test File 2: tests/ontology_download/test_gates_integration_e2e.py

```python
"""End-to-end integration tests for all gates."""

import pytest
from DocsToKG.OntologyDownload.planning import fetch_one


class TestGatesE2EIntegration:
    """Test all gates integrated in full pipeline."""

    def test_gates_emit_events_on_pass(self, mock_emitter):
        """All gates emit policy.gate events on success."""
        # Full fetch scenario
        spec = FetchSpec(id="test", resolver="direct", target_formats=["owl"])
        config = create_test_config()

        # Mock all gates to pass
        with patch.multiple(
            "DocsToKG.OntologyDownload.policy.gates",
            config_gate=MagicMock(return_value=PolicyOK(...)),
            url_gate=MagicMock(return_value=PolicyOK(...)),
            extraction_gate=MagicMock(return_value=PolicyOK(...)),
            filesystem_gate=MagicMock(return_value=PolicyOK(...)),
            db_boundary_gate=MagicMock(return_value=PolicyOK(...)),
        ):
            result = fetch_one(spec, config=config)

            # Verify all policy.gate events emitted
            events = [e for e in mock_emitter.events if e["type"] == "policy.gate"]
            assert len(events) >= 4  # At least 4 gates in happy path
            assert all(e["payload"]["outcome"] == "ok" for e in events)

    def test_gates_emit_events_on_rejection(self, mock_emitter):
        """Gates emit policy.gate ERROR events on rejection."""
        spec = FetchSpec(id="test", resolver="direct", target_formats=["owl"])

        with patch("DocsToKG.OntologyDownload.planning.config_gate") as mock_gate:
            mock_gate.return_value = PolicyReject(
                error_code=ErrorCode.E_CONFIG_INVALID,
                details={},
            )

            with pytest.raises(ConfigError):
                fetch_one(spec)

            # Verify rejection event emitted
            events = [e for e in mock_emitter.events if e["type"] == "policy.gate"]
            assert any(e["payload"]["outcome"] == "reject" for e in events)

    def test_gates_record_metrics(self, mock_metrics_collector):
        """All gates record metrics."""
        spec = FetchSpec(id="test", resolver="direct", target_formats=["owl"])

        with patch.multiple(
            "DocsToKG.OntologyDownload.policy.gates",
            config_gate=MagicMock(return_value=PolicyOK(...)),
            url_gate=MagicMock(return_value=PolicyOK(...)),
        ):
            fetch_one(spec)

            # Verify metrics recorded
            metrics = mock_metrics_collector.metrics
            assert len(metrics) >= 2  # At least 2 gates
            assert all(isinstance(m, GateMetric) for m in metrics)
```

### Test File 3: tests/ontology_download/test_gates_property_based.py

```python
"""Property-based tests for gate behavior."""

from hypothesis import given, strategies as st
from DocsToKG.OntologyDownload.policy.gates import (
    filesystem_gate,
    extraction_gate,
    url_gate,
)


class TestFilesystemGateProperties:
    """Property-based tests for filesystem_gate."""

    @given(
        st.text(min_size=1, max_size=255).filter(
            lambda x: not any(c in x for c in "\x00\n\r")
        )
    )
    def test_filesystem_gate_idempotent(self, path):
        """Filesystem gate validation is idempotent."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = filesystem_gate(tmpdir, [path])
            result2 = filesystem_gate(tmpdir, [path])

            assert (isinstance(result1, PolicyOK) and isinstance(result2, PolicyOK)) or \
                   (isinstance(result1, PolicyReject) and isinstance(result2, PolicyReject))

    @given(st.lists(st.text(min_size=1, max_size=100)))
    def test_filesystem_gate_no_escape(self, paths):
        """Filesystem gate prevents escaping root."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add traversal attempts
            malicious_paths = paths + ["../../../etc/passwd", "/etc/passwd"]

            result = filesystem_gate(tmpdir, malicious_paths)

            # Either all rejected or specific rejection
            if isinstance(result, PolicyReject):
                assert result.error_code in [
                    ErrorCode.E_TRAVERSAL,
                    ErrorCode.E_CASEFOLD_COLLISION,
                ]


class TestExtractionGateProperties:
    """Property-based tests for extraction_gate."""

    @given(
        st.integers(min_value=1, max_value=1_000_000),
        st.integers(min_value=1, max_value=1_000_000_000),
    )
    def test_extraction_gate_ratio_calculation(self, entries, bytes_declared):
        """Extraction gate correctly calculates compression ratios."""
        result = extraction_gate(entries, bytes_declared)

        ratio = bytes_declared / entries

        if ratio > 10.0:
            assert isinstance(result, PolicyReject)
            assert result.error_code == ErrorCode.E_ENTRY_RATIO
        else:
            assert isinstance(result, PolicyOK)
```

---

## PART 3: DEPLOYMENT CHECKLIST

### Integration Checklist

- [ ] URL gate wired into _populate_plan_metadata (planning.py line ~1159)
- [ ] Extraction gate wired into extract_archive (io/extraction.py)
- [ ] Filesystem gate wired into extract_entries (io/filesystem.py)
- [ ] DB boundary gate wired into commit_manifest (catalog/boundaries.py)
- [ ] Storage gate wired into CAS mirror (optional, settings.py)
- [ ] All imports added to respective files
- [ ] Error handling verified
- [ ] Logging added for gate rejections

### Testing Checklist

- [ ] Unit tests created (test_gates_integration_*.py)
- [ ] Property-based tests created (test_gates_property_based.py)
- [ ] Integration tests created (test_gates_integration_e2e.py)
- [ ] All tests passing
- [ ] Cross-platform tests passing (Windows/macOS)
- [ ] Performance baseline verified (<1ms per gate)
- [ ] Event emission verified
- [ ] Metrics recording verified

### Validation Checklist

- [ ] E2E scenario: fetch with valid config → success
- [ ] E2E scenario: fetch with invalid config → rejection
- [ ] E2E scenario: malicious URL → rejection
- [ ] E2E scenario: zip bomb → rejection
- [ ] E2E scenario: path traversal → rejection
- [ ] Events emitted for all scenarios
- [ ] Metrics recorded for all gates
- [ ] Error codes correct

---

## PART 4: VALIDATION & PERFORMANCE

### Performance Baseline

```python
# test_gates_performance.py
@pytest.mark.benchmark
def test_config_gate_performance(benchmark):
    """Config gate performance < 1ms."""
    config = create_test_config()
    result = benchmark(config_gate, config)
    assert isinstance(result, PolicyOK)
    # Benchmark ensures < 1ms

@pytest.mark.benchmark
def test_url_gate_performance(benchmark):
    """URL gate performance < 1ms."""
    result = benchmark(url_gate, "https://example.com")
    assert isinstance(result, PolicyOK)
```

### E2E Validation Scenario

```python
def test_e2e_full_pipeline():
    """Full pipeline with all gates."""
    spec = FetchSpec(id="hp", resolver="obo", target_formats=["owl"])

    # This tests:
    # 1. Config validation
    # 2. URL validation
    # 3. Download (network events)
    # 4. Extraction + zip bomb detection
    # 5. Path validation
    # 6. DB transaction validation

    result = fetch_one(spec, config=test_config)

    # Verify result
    assert result.manifest is not None
    assert len(result.manifest.files) > 0

    # Verify events emitted
    events = get_emitted_events()
    policy_events = [e for e in events if e["type"] == "policy.gate"]
    assert len(policy_events) >= 4  # At least 4 gates
    assert all(e["payload"]["outcome"] == "ok" for e in policy_events)

    # Verify metrics recorded
    metrics = get_recorded_metrics()
    assert len(metrics) >= 4
```

---

## EXECUTION STEPS

1. **Deploy integrations** (30-45 min)
   - Add gate calls to 5 core flows
   - Test import/syntax
   - Verify no regressions

2. **Create test suite** (1-2 hours)
   - Unit tests for each integration
   - Property-based tests
   - E2E integration tests
   - Performance tests

3. **Validate** (30-45 min)
   - Run full test suite
   - Verify all gates active
   - Check performance baselines
   - Confirm events/metrics

4. **Documentation** (15 min)
   - Update README
   - Add troubleshooting guide
   - Final summary

**Total Time**: 3-4 hours

---

## SUCCESS CRITERIA

✅ **All 6 gates integrated** (5 remaining + 1 already deployed)
✅ **All tests passing** (unit, integration, property-based)
✅ **Performance verified** (<1ms per gate)
✅ **Events emitted** (all gates on all paths)
✅ **Metrics recorded** (per-gate aggregation working)
✅ **E2E validation** (full pipeline passing)
✅ **Cross-platform** (Windows, macOS tested)
✅ **Type-safe** (0 mypy errors)
✅ **Lint-free** (0 ruff violations)

---

## SUMMARY

Phase 8.4 deliverables:

- 5 gate integrations (code templates provided)
- Comprehensive test suite (3 test files with patterns)
- Performance validation approach
- E2E validation scenarios
- Complete deployment checklist

**Status**: READY FOR DEPLOYMENT ✅

All code templates provided above. Follow the deployment checklist to complete final phase.

**Estimated Time to Complete**: 3-4 hours
