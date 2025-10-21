# === NAVMAP v1 ===
# {
#   "module": "tests.test_perf_utils",
#   "purpose": "Unit tests for performance utility functions",
#   "sections": [
#     {"id": "budget-tests", "name": "Budget Tests", "anchor": "budget-tests", "kind": "section"},
#     {"id": "monitor-tests", "name": "Monitor Tests", "anchor": "monitor-tests", "kind": "section"},
#     {"id": "baseline-tests", "name": "Baseline Tests", "anchor": "baseline-tests", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Unit tests for performance benchmarking utilities.

Tests budget checking, resource monitoring, and baseline comparison logic.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from tests.benchmarks.perf_utils import (
    PerformanceBudget,
    ResourceMonitor,
    load_baseline,
    save_baseline,
    compare_to_baseline,
    HTTPX_BUDGET,
)


# --- Budget Tests ---


@pytest.mark.unit
def test_performance_budget_within():
    """Test budget check when value is within tolerance."""
    budget = PerformanceBudget(name="test", threshold_ms=100.0, tolerance_pct=5.0)

    passes, message = budget.check(95.0)
    assert passes
    assert "within budget" in message.lower()


@pytest.mark.unit
def test_performance_budget_exceeds():
    """Test budget check when value exceeds threshold."""
    budget = PerformanceBudget(name="test", threshold_ms=100.0, tolerance_pct=5.0)

    passes, message = budget.check(110.0)
    assert not passes
    assert "exceeded" in message.lower()


@pytest.mark.unit
def test_performance_budget_improves():
    """Test budget check when value improves significantly."""
    budget = PerformanceBudget(name="test", threshold_ms=100.0, tolerance_pct=5.0)

    passes, message = budget.check(80.0)
    assert passes
    assert "improved" in message.lower()


@pytest.mark.unit
def test_httpx_budget_spec():
    """Test HTTPX budget matches specification."""
    assert HTTPX_BUDGET.threshold_ms == 5.0
    assert HTTPX_BUDGET.tolerance_pct == 5.0
    assert "GET 200" in HTTPX_BUDGET.description


# --- Resource Monitor Tests ---


@pytest.mark.unit
def test_resource_monitor_start_stop():
    """Test resource monitor start/stop flow."""
    monitor = ResourceMonitor()

    monitor.start()
    metrics = monitor.stop()

    assert "elapsed_sec" in metrics
    assert "rss_delta_mb" in metrics
    assert "fd_delta" in metrics
    assert metrics["elapsed_sec"] > 0


@pytest.mark.unit
def test_resource_monitor_not_started():
    """Test resource monitor raises when stopped before start."""
    monitor = ResourceMonitor()

    with pytest.raises(RuntimeError, match="not started"):
        monitor.stop()


@pytest.mark.unit
def test_resource_monitor_no_leak():
    """Test resource monitor asserts no leak for small delta."""
    monitor = ResourceMonitor()
    monitor.start()
    monitor.stop()

    # Should not raise (mock data shows no leak)
    monitor.assert_no_leak(threshold_mb=100.0, threshold_fds=10)


@pytest.mark.unit
def test_resource_monitor_memory_leak():
    """Test resource monitor detects memory leak."""
    monitor = ResourceMonitor()

    # Manually set metrics to simulate leak
    monitor.metrics = {
        "rss_delta_mb": 50.0,  # Exceeds default threshold
        "fd_delta": 0,
    }

    with pytest.raises(AssertionError, match="Memory leak"):
        monitor.assert_no_leak(threshold_mb=10.0)


@pytest.mark.unit
def test_resource_monitor_fd_leak():
    """Test resource monitor detects FD leak."""
    monitor = ResourceMonitor()

    # Manually set metrics to simulate leak
    monitor.metrics = {
        "rss_delta_mb": 0,
        "fd_delta": 20,  # Exceeds default threshold
    }

    with pytest.raises(AssertionError, match="FD leak"):
        monitor.assert_no_leak(threshold_fds=5)


# --- Baseline Tests ---


@pytest.mark.unit
def test_load_baseline_nonexistent():
    """Test load_baseline returns empty dict for nonexistent file."""
    result = load_baseline(Path("/nonexistent/baseline.json"))
    assert result == {}


@pytest.mark.unit
def test_save_and_load_baseline(tmp_path):
    """Test save and load baseline round-trip."""
    baseline_file = tmp_path / "baseline.json"
    data = {
        "httpx_get": 3.5,
        "duckdb_query": 150.0,
    }

    save_baseline(baseline_file, data)
    loaded = load_baseline(baseline_file)

    assert loaded == data


@pytest.mark.unit
def test_load_baseline_invalid_json(tmp_path):
    """Test load_baseline handles invalid JSON."""
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("invalid json {")

    result = load_baseline(baseline_file)
    assert result == {}


@pytest.mark.unit
def test_compare_to_baseline_first_run():
    """Test baseline comparison for first run (no baseline)."""
    passes, message = compare_to_baseline(100.0, None)

    assert passes
    assert "first run" in message.lower()


@pytest.mark.unit
def test_compare_to_baseline_no_regression():
    """Test baseline comparison with no regression."""
    passes, message = compare_to_baseline(100.0, 100.0)

    assert passes
    assert "tolerance" in message.lower()


@pytest.mark.unit
def test_compare_to_baseline_improvement():
    """Test baseline comparison detects improvement."""
    passes, message = compare_to_baseline(80.0, 100.0)

    assert passes
    assert "improvement" in message.lower()
    assert "20" in message  # 20% improvement


@pytest.mark.unit
def test_compare_to_baseline_regression():
    """Test baseline comparison detects regression."""
    passes, message = compare_to_baseline(120.0, 100.0)

    assert not passes
    assert "regression" in message.lower()
    assert "20" in message  # 20% regression


@pytest.mark.unit
def test_compare_to_baseline_small_regression():
    """Test baseline comparison ignores small regressions."""
    passes, message = compare_to_baseline(103.0, 100.0)

    assert passes
    assert "tolerance" in message.lower()


# --- Integration Tests ---


@pytest.mark.component
def test_perf_utils_full_workflow(tmp_path):
    """Test full performance utility workflow."""
    # Save initial baseline
    baseline_file = tmp_path / "baseline.json"
    initial_data = {"httpx_get": 3.5, "duckdb_query": 150.0}
    save_baseline(baseline_file, initial_data)

    # Load and verify
    loaded = load_baseline(baseline_file)
    assert loaded == initial_data

    # Check performance against baseline
    passes, msg = compare_to_baseline(3.6, loaded["httpx_get"])
    assert passes

    # Budget check
    passes, msg = HTTPX_BUDGET.check(3.6)
    assert passes

    # Resource monitoring
    monitor = ResourceMonitor()
    monitor.start()
    _ = [i for i in range(1000)]  # Some work
    metrics = monitor.stop()

    # Should not leak
    monitor.assert_no_leak()
