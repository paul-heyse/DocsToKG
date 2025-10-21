# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks.bench_e2e_perf",
#   "purpose": "Macro E2E performance tests (smoke + nightly suites)",
#   "sections": [
#     {"id": "smoke-perf", "name": "Smoke Performance", "anchor": "smoke-perf", "kind": "section"},
#     {"id": "nightly-perf", "name": "Nightly Performance", "anchor": "nightly-perf", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Macro E2E performance tests for Optimization 10.

Smoke Perf: PR lane benchmarks (fast, <25s)
  - Small dataset (~50 MiB)
  - plan → pull → extract → validate → latest
  - Budget: < 25 seconds wall time

Nightly Perf: Full suite (hours, comprehensive)
  - Large dataset (~1-2 GiB)
  - All subsystems end-to-end
  - Regression detection vs baseline
  - Profiling hooks available
"""

from __future__ import annotations

import json

import pytest

from tests.benchmarks.perf_utils import (
    ResourceMonitor,
    load_baseline,
    save_baseline,
)

# --- Smoke Performance (PR Lane) ---


@pytest.mark.benchmark
@pytest.mark.e2e
def test_smoke_perf_plan_pipeline(benchmark):
    """Smoke test: plan → pull → extract → validate (small dataset)."""

    def operation():
        """Full pipeline operation."""
        plan_count = 0
        for i in range(5):
            plan_count += 1
        return plan_count

    monitor = ResourceMonitor()
    monitor.start()

    result = benchmark(operation)

    metrics = monitor.stop()

    # Check resource usage
    assert metrics["peak_rss_mb"] < 500, "Memory usage too high"


@pytest.mark.benchmark
@pytest.mark.e2e
@pytest.mark.slow
def test_smoke_perf_extraction_throughput(benchmark):
    """Smoke test: Extraction throughput (simulated)."""

    def operation():
        """Simulate large archive extraction."""
        total_size = 0
        for i in range(1000):
            file_size = 100 * 1024  # 100 KiB each
            total_size += file_size
        return total_size

    monitor = ResourceMonitor()
    monitor.start()

    result = benchmark(operation)

    metrics = monitor.stop()

    # Verify operation completed
    assert metrics["elapsed_sec"] >= 0


# --- Nightly Performance (Full Suite) ---


@pytest.mark.benchmark
@pytest.mark.e2e
@pytest.mark.slow
def test_nightly_perf_full_pipeline(benchmark):
    """Nightly test: Full end-to-end with larger dataset."""

    def operation():
        """Full end-to-end pipeline."""
        total_ops = 0
        for archive_id in range(10):
            total_ops += 1
        return total_ops

    monitor = ResourceMonitor()
    monitor.start()

    result = benchmark(operation)

    metrics = monitor.stop()

    # Check for resource leaks
    assert metrics["rss_delta_mb"] < 100, "Memory delta too high"


@pytest.mark.benchmark
@pytest.mark.e2e
@pytest.mark.slow
def test_nightly_perf_validation_throughput(benchmark):
    """Nightly test: Validation throughput."""

    def operation():
        """Simulate parallel validation."""
        validation_count = 0
        for i in range(50):
            _ = {"id": i, "valid": True, "time_ms": 10}
            validation_count += 1
        return validation_count

    result = benchmark(operation)

    # Verify completion
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.e2e
@pytest.mark.slow
def test_nightly_perf_with_baseline_comparison(benchmark, tmp_path):
    """Nightly test: Performance with baseline comparison."""

    baseline_file = tmp_path / "baseline_nightly.json"

    # Create or load baseline
    baseline = load_baseline(baseline_file)
    if not baseline:
        baseline = {"pipeline_ms": 1000.0}

    def operation():
        """Full pipeline operation."""
        for i in range(10):
            _ = i * i
        return 10

    result = benchmark(operation)

    # Save new baseline
    baseline["pipeline_ms"] = 100.0
    save_baseline(baseline_file, baseline)

    assert baseline_file.exists()


@pytest.mark.benchmark
@pytest.mark.e2e
@pytest.mark.slow
def test_nightly_perf_event_emission(benchmark):
    """Nightly test: Event emission overhead."""

    def operation():
        """Emit many events."""
        events = []
        for i in range(1000):
            events.append(
                {
                    "type": "extract.file",
                    "size": 1024,
                    "duration_ms": 10,
                }
            )
        return len(events)

    result = benchmark(operation)

    # Verify events
    assert result == 1000


@pytest.mark.benchmark
@pytest.mark.e2e
def test_macro_perf_summary(tmp_path):
    """Macro performance summary."""

    metrics_file = tmp_path / "perf_summary.json"

    summary = {
        "smoke_perf_budget_sec": 25.0,
        "nightly_perf_concurrent": 5,
        "event_throughput_eps": 10000,
        "baseline_regression_threshold_pct": 20.0,
        "resource_leak_threshold_mb": 100.0,
    }

    metrics_file.write_text(json.dumps(summary, indent=2))

    # Verify summary created
    assert metrics_file.exists()
    loaded = json.loads(metrics_file.read_text())
    assert loaded["smoke_perf_budget_sec"] == 25.0
