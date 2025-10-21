# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks.perf_utils",
#   "purpose": "Performance utility functions and budget management",
#   "sections": [
#     {"id": "perf-budget", "name": "PerformanceBudget", "anchor": "class-perf-budget", "kind": "class"},
#     {"id": "resource-monitor", "name": "ResourceMonitor", "anchor": "class-resource-monitor", "kind": "class"},
#     {"id": "baseline-helpers", "name": "Baseline Helpers", "anchor": "baseline-helpers", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Performance benchmarking utilities for Optimization 10.

Provides budget tracking, resource monitoring, and baseline management
for deterministic performance regression detection.
"""

from __future__ import annotations

import json
import os
import psutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class PerformanceBudget:
    """Performance budget with thresholds and tolerance."""

    name: str
    threshold_ms: float  # p95 threshold in milliseconds
    tolerance_pct: float = 5.0  # ±5% tolerance
    description: str = ""

    def check(self, actual_ms: float) -> tuple[bool, str]:
        """
        Check if actual time is within budget.

        Args:
            actual_ms: Actual time in milliseconds

        Returns:
            (passes, message): Pass status and explanation
        """
        upper_bound = self.threshold_ms * (1 + self.tolerance_pct / 100)
        lower_bound = self.threshold_ms * (1 - self.tolerance_pct / 100)

        if actual_ms > upper_bound:
            pct_over = ((actual_ms - self.threshold_ms) / self.threshold_ms) * 100
            return (
                False,
                f"{self.name} exceeded budget: {actual_ms:.2f}ms (budget: {self.threshold_ms:.2f}ms, +{pct_over:.1f}%)",
            )
        elif actual_ms < lower_bound:
            pct_under = ((self.threshold_ms - actual_ms) / self.threshold_ms) * 100
            return (
                True,
                f"{self.name} improved: {actual_ms:.2f}ms (budget: {self.threshold_ms:.2f}ms, -{pct_under:.1f}%)",
            )
        else:
            return (True, f"{self.name} within budget: {actual_ms:.2f}ms")


class ResourceMonitor:
    """Monitor resource usage (CPU, memory, file descriptors)."""

    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process(os.getpid())
        self.start_time: float | None = None
        self.start_rss: int | None = None
        self.start_vms: int | None = None
        self.start_fds: int | None = None
        self.metrics: dict[str, Any] = {}

    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.perf_counter()
        self.start_rss = self.process.memory_info().rss
        self.start_vms = self.process.memory_info().vms
        try:
            self.start_fds = len(self.process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            self.start_fds = 0

    def stop(self) -> dict[str, Any]:
        """Stop monitoring and return metrics."""
        if self.start_time is None:
            raise RuntimeError("Monitor not started")

        elapsed = time.perf_counter() - self.start_time
        current_rss = self.process.memory_info().rss
        current_vms = self.process.memory_info().vms

        try:
            current_fds = len(self.process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            current_fds = self.start_fds or 0

        self.metrics = {
            "elapsed_sec": elapsed,
            "rss_delta_mb": (current_rss - (self.start_rss or 0)) / (1024 * 1024),
            "vms_delta_mb": (current_vms - (self.start_vms or 0)) / (1024 * 1024),
            "fd_delta": (current_fds or 0) - (self.start_fds or 0),
            "peak_rss_mb": current_rss / (1024 * 1024),
            "peak_vms_mb": current_vms / (1024 * 1024),
        }
        return self.metrics

    def assert_no_leak(self, threshold_mb: float = 10.0, threshold_fds: int = 5) -> None:
        """Assert no resource leak (memory growth or FD leaks)."""
        if not self.metrics:
            raise RuntimeError("No metrics collected")

        rss_delta = self.metrics["rss_delta_mb"]
        fd_delta = self.metrics["fd_delta"]

        assert rss_delta < threshold_mb, (
            f"Memory leak: {rss_delta:.2f}MB > {threshold_mb:.2f}MB threshold"
        )
        assert fd_delta < threshold_fds, f"FD leak: {fd_delta} > {threshold_fds} threshold"


# === Standard Budgets (from Optimization 10 spec) ===

HTTPX_BUDGET = PerformanceBudget(
    name="HTTPX",
    threshold_ms=5.0,
    description="GET 200 (128 KiB body) p95 elapsed time",
)

HTTPX_REDIRECT_BUDGET = PerformanceBudget(
    name="HTTPX Redirect",
    threshold_ms=8.0,
    description="302→200 redirect audit (1 hop) p95",
)

RATELIMITER_BUDGET = PerformanceBudget(
    name="Ratelimiter",
    threshold_ms=0.1,
    description="fail-fast path p95 (no wait)",
)

EXTRACTION_PRESCAN_BUDGET = PerformanceBudget(
    name="Extraction Pre-scan",
    threshold_ms=250.0,
    description="Pre-scan 10k headers (no writes) p95",
)

EXTRACTION_THROUGHPUT_BUDGET = PerformanceBudget(
    name="Extraction Throughput",
    threshold_ms=6667.0,  # 1 GiB ÷ 150 MiB/s ≈ 6.67s
    description="Stream 1 GiB at ≥150 MiB/s p95",
)

DUCKDB_INSERT_BUDGET = PerformanceBudget(
    name="DuckDB Insert",
    threshold_ms=1500.0,
    description="Bulk insert 50k rows via Arrow appender p95",
)

DUCKDB_QUERY_BUDGET = PerformanceBudget(
    name="DuckDB Query",
    threshold_ms=200.0,
    description="v_version_stats on 200k rows p95",
)

POLARS_BUDGET = PerformanceBudget(
    name="Polars Pipeline",
    threshold_ms=2000.0,
    description="scan_ndjson 1M entries + collect(streaming=True) p95",
)


def load_baseline(baseline_file: Path) -> dict[str, Any]:
    """
    Load baseline benchmark results.

    Args:
        baseline_file: Path to baseline JSON

    Returns:
        Baseline data
    """
    if not baseline_file.exists():
        return {}

    try:
        return json.loads(baseline_file.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}


def save_baseline(baseline_file: Path, data: dict[str, Any]) -> None:
    """
    Save baseline benchmark results.

    Args:
        baseline_file: Path to baseline JSON
        data: Baseline data to save
    """
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(json.dumps(data, indent=2, sort_keys=True))


def compare_to_baseline(
    current_value: float, baseline_value: float | None, regression_threshold_pct: float = 15.0
) -> tuple[bool, str]:
    """
    Compare current value to baseline.

    Args:
        current_value: Current measurement
        baseline_value: Previous baseline (None = first run)
        regression_threshold_pct: Threshold for regression warning

    Returns:
        (passes, message): Pass status and message
    """
    if baseline_value is None:
        return (True, "First run (no baseline for comparison)")

    pct_change = ((current_value - baseline_value) / baseline_value) * 100

    if pct_change > regression_threshold_pct:
        return (
            False,
            f"Performance regression: {pct_change:.1f}% slower than baseline",
        )
    elif pct_change < -5.0:  # Improvement
        return (True, f"Performance improvement: {abs(pct_change):.1f}% faster")
    else:
        return (True, f"Within tolerance: {pct_change:+.1f}% vs baseline")
