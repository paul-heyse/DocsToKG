# === NAVMAP v1 ===
# {
#   "module": "tests.test_regression_detection",
#   "purpose": "Tests for regression detection and CI integration",
#   "sections": [
#     {"id": "baseline-store-tests", "name": "BaselineStore Tests", "anchor": "baseline-store-tests", "kind": "section"},
#     {"id": "regression-detector-tests", "name": "RegressionDetector Tests", "anchor": "regression-detector-tests", "kind": "section"},
#     {"id": "ci-helpers-tests", "name": "CI Helpers Tests", "anchor": "ci-helpers-tests", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Tests for regression detection and CI wiring.

Tests baseline storage, regression detection logic, and CI environment helpers.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from tests.benchmarks.regression_detection import (
    BaselineStore,
    RegressionDetector,
    RegressionSeverity,
    get_ci_runner_class,
    get_ci_is_nightly,
    should_fail_on_regression,
)


# --- BaselineStore Tests ---


@pytest.mark.unit
def test_baseline_store_init(tmp_path):
    """Test baseline store initialization."""
    store = BaselineStore(tmp_path)
    assert store.baseline_dir.exists()


@pytest.mark.unit
def test_baseline_store_get_file(tmp_path):
    """Test baseline file path generation."""
    store = BaselineStore(tmp_path)
    file_path = store.get_baseline_file("linux-x86_64")

    assert str(file_path).endswith("linux-x86_64_baseline.json")


@pytest.mark.unit
def test_baseline_store_save_and_load(tmp_path):
    """Test save and load baseline."""
    store = BaselineStore(tmp_path)
    data = {"httpx_get": 3.5, "duckdb_query": 150.0}

    store.save_baseline("linux-x86_64", data)
    loaded = store.load_baseline("linux-x86_64")

    assert loaded == data


@pytest.mark.unit
def test_baseline_store_load_nonexistent(tmp_path):
    """Test load nonexistent baseline."""
    store = BaselineStore(tmp_path)
    loaded = store.load_baseline("nonexistent-runner")

    assert loaded == {}


@pytest.mark.unit
def test_baseline_store_update_entry(tmp_path):
    """Test updating single baseline entry."""
    store = BaselineStore(tmp_path)

    store.update_entry("linux-x86_64", "httpx_get", 3.5)
    store.update_entry("linux-x86_64", "duckdb_query", 150.0)

    loaded = store.load_baseline("linux-x86_64")

    assert loaded["httpx_get"]["value_ms"] == 3.5
    assert loaded["duckdb_query"]["value_ms"] == 150.0


@pytest.mark.unit
def test_baseline_store_multiple_runners(tmp_path):
    """Test multiple runner classes."""
    store = BaselineStore(tmp_path)

    store.save_baseline("linux-x86_64", {"httpx": {"value_ms": 3.5}})
    store.save_baseline("macos-arm64", {"httpx": {"value_ms": 4.0}})

    linux_baseline = store.load_baseline("linux-x86_64")
    macos_baseline = store.load_baseline("macos-arm64")

    assert linux_baseline["httpx"]["value_ms"] == 3.5
    assert macos_baseline["httpx"]["value_ms"] == 4.0


# --- RegressionDetector Tests ---


@pytest.mark.unit
def test_regression_detector_init():
    """Test regression detector initialization."""
    detector = RegressionDetector(pr_lane_threshold_pct=15.0, nightly_threshold_pct=20.0)

    assert detector.pr_lane_threshold_pct == 15.0
    assert detector.nightly_threshold_pct == 20.0


@pytest.mark.unit
def test_regression_detector_first_run():
    """Test detection on first run (no baseline)."""
    detector = RegressionDetector()
    severity, message = detector.detect("test_bench", 100.0, None)

    assert severity == RegressionSeverity.NONE
    assert "First run" in message


@pytest.mark.unit
def test_regression_detector_no_regression():
    """Test detection with no regression."""
    detector = RegressionDetector()
    severity, message = detector.detect("test_bench", 100.0, 100.0)

    assert severity == RegressionSeverity.NONE
    assert "tolerance" in message.lower()


@pytest.mark.unit
def test_regression_detector_improvement():
    """Test detection of improvement."""
    detector = RegressionDetector()
    severity, message = detector.detect("test_bench", 80.0, 100.0)

    assert severity == RegressionSeverity.NONE
    assert "faster" in message.lower()
    assert len(detector.improvements) == 1


@pytest.mark.unit
def test_regression_detector_pr_lane_warning():
    """Test PR lane regression warning."""
    detector = RegressionDetector(pr_lane_threshold_pct=15.0)
    severity, message = detector.detect("test_bench", 120.0, 100.0, is_nightly=False)

    assert severity == RegressionSeverity.WARNING
    assert "20" in message  # 20% regression
    assert len(detector.regressions) == 1


@pytest.mark.unit
def test_regression_detector_nightly_failure():
    """Test nightly regression failure."""
    detector = RegressionDetector(nightly_threshold_pct=20.0)
    severity, message = detector.detect("test_bench", 125.0, 100.0, is_nightly=True)

    assert severity == RegressionSeverity.FAILURE
    assert "25" in message  # 25% regression
    assert len(detector.regressions) == 1


@pytest.mark.unit
def test_regression_detector_small_nightly_regression():
    """Test small regression doesn't fail on nightly."""
    detector = RegressionDetector(nightly_threshold_pct=20.0)
    severity, message = detector.detect("test_bench", 105.0, 100.0, is_nightly=True)

    assert severity == RegressionSeverity.NONE
    assert "tolerance" in message.lower()


@pytest.mark.unit
def test_regression_detector_has_failures():
    """Test has_failures detection."""
    detector = RegressionDetector(pr_lane_threshold_pct=15.0)

    # Add warning
    detector.detect("test1", 120.0, 100.0, is_nightly=False)
    assert detector.has_failures()

    # Add failure
    detector.detect("test2", 125.0, 100.0, is_nightly=True)
    assert detector.has_failures()


@pytest.mark.unit
def test_regression_detector_summary():
    """Test regression summary."""
    detector = RegressionDetector(pr_lane_threshold_pct=15.0)

    detector.detect("bench1", 120.0, 100.0, is_nightly=False)  # Regression
    detector.detect("bench2", 80.0, 100.0)  # Improvement
    detector.detect("bench3", 100.0, 100.0)  # No change

    summary = detector.summary()

    assert summary["regressions"] == 1
    assert summary["improvements"] == 1
    assert len(summary["regression_details"]) == 1


# --- CI Helpers Tests ---


@pytest.mark.unit
def test_get_ci_runner_class():
    """Test CI runner class detection."""
    runner_class = get_ci_runner_class()

    # Should have format: os-arch
    assert "-" in runner_class
    parts = runner_class.split("-")
    assert len(parts) == 2


@pytest.mark.unit
def test_get_ci_is_nightly_default():
    """Test nightly detection defaults to False."""
    # Without env vars set, should be False
    import os
    
    # Clear potential env vars
    os.environ.pop("CI_NIGHTLY", None)
    os.environ.pop("GITHUB_EVENT_NAME", None)
    os.environ.pop("CI_JOB_NAME", None)

    is_nightly = get_ci_is_nightly()
    assert is_nightly is False


@pytest.mark.unit
def test_get_ci_is_nightly_with_env():
    """Test nightly detection with CI_NIGHTLY env var."""
    import os

    os.environ["CI_NIGHTLY"] = "1"
    try:
        is_nightly = get_ci_is_nightly()
        assert is_nightly is True
    finally:
        os.environ.pop("CI_NIGHTLY", None)


@pytest.mark.unit
def test_should_fail_on_regression_default():
    """Test fail on regression defaults based on CI context."""
    import os

    os.environ.pop("CI_NIGHTLY", None)
    os.environ.pop("GITHUB_EVENT_NAME", None)

    should_fail = should_fail_on_regression()
    # Default PR context: False
    assert should_fail is False


# --- Integration Tests ---


@pytest.mark.component
def test_regression_detection_full_workflow(tmp_path):
    """Test full regression detection workflow."""
    # Setup baseline store
    store = BaselineStore(tmp_path)
    runner_class = "linux-x86_64"

    # Save initial baseline
    baseline_data = {
        "httpx_get": {"value_ms": 3.5},
        "duckdb_query": {"value_ms": 150.0},
    }
    store.save_baseline(runner_class, baseline_data)

    # Simulate current measurements
    detector = RegressionDetector()

    httpx_baseline = baseline_data["httpx_get"]["value_ms"]
    duckdb_baseline = baseline_data["duckdb_query"]["value_ms"]

    # Good measurement
    severity1, msg1 = detector.detect("httpx_get", 3.6, httpx_baseline)
    assert severity1 == RegressionSeverity.NONE

    # Regression
    severity2, msg2 = detector.detect("duckdb_query", 180.0, duckdb_baseline)
    assert severity2 == RegressionSeverity.WARNING

    # Check summary
    summary = detector.summary()
    assert summary["regressions"] == 1
    assert summary["improvements"] == 0

    # Update baseline with new measurements
    store.update_entry(runner_class, "httpx_get", 3.6)
    store.update_entry(runner_class, "duckdb_query", 180.0)

    # Verify update
    new_baseline = store.load_baseline(runner_class)
    assert new_baseline["httpx_get"]["value_ms"] == 3.6
    assert new_baseline["duckdb_query"]["value_ms"] == 180.0
