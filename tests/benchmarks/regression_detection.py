# === NAVMAP v1 ===
# {
#   "module": "tests.benchmarks.regression_detection",
#   "purpose": "Regression detection and CI wiring for performance tracking",
#   "sections": [
#     {"id": "baseline-store", "name": "BaselineStore", "anchor": "class-baseline-store", "kind": "class"},
#     {"id": "regression-detector", "name": "RegressionDetector", "anchor": "class-regression-detector", "kind": "class"},
#     {"id": "ci-helpers", "name": "CI Helpers", "anchor": "ci-helpers", "kind": "section"}
#   ]
# }
# === /NAVMAP ===

"""
Regression detection and CI wiring for Optimization 10 Phase 3.

Provides persistent baseline storage, regression analysis, and CI integration
for detecting performance regressions across PR lanes and nightly suites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from enum import Enum


class RegressionSeverity(Enum):
    """Regression severity levels."""

    NONE = "none"
    WARNING = "warning"  # >15% slower
    FAILURE = "failure"  # >20% slower (nightly)


@dataclass
class BaselineEntry:
    """Single baseline entry with metadata."""

    name: str
    value_ms: float
    timestamp: str = ""
    platform: str = "linux"  # linux, macos, windows
    python_version: str = "3.13"
    commit_hash: str = ""


class BaselineStore:
    """Persistent storage for performance baselines."""

    def __init__(self, baseline_dir: Path):
        """
        Initialize baseline store.

        Args:
            baseline_dir: Directory to store baselines (e.g., .ci/perf/baselines)
        """
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def get_baseline_file(self, runner_class: str) -> Path:
        """
        Get baseline file for runner class.

        Args:
            runner_class: Runner class identifier (e.g., 'linux-x86_64', 'macos-arm64')

        Returns:
            Path to baseline JSON file
        """
        return self.baseline_dir / f"{runner_class}_baseline.json"

    def load_baseline(self, runner_class: str) -> dict[str, Any]:
        """
        Load baseline for runner class.

        Args:
            runner_class: Runner class identifier

        Returns:
            Baseline data (empty dict if not found)
        """
        baseline_file = self.get_baseline_file(runner_class)
        if not baseline_file.exists():
            return {}

        try:
            return json.loads(baseline_file.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}

    def save_baseline(self, runner_class: str, data: dict[str, Any]) -> None:
        """
        Save baseline for runner class.

        Args:
            runner_class: Runner class identifier
            data: Baseline data to save
        """
        baseline_file = self.get_baseline_file(runner_class)
        baseline_file.write_text(json.dumps(data, indent=2, sort_keys=True))

    def update_entry(
        self, runner_class: str, entry_name: str, value_ms: float, metadata: dict[str, str] | None = None
    ) -> None:
        """
        Update single baseline entry.

        Args:
            runner_class: Runner class identifier
            entry_name: Benchmark name
            value_ms: Time in milliseconds
            metadata: Optional metadata (timestamp, commit, etc.)
        """
        baseline = self.load_baseline(runner_class)
        baseline[entry_name] = {
            "value_ms": value_ms,
            **(metadata or {}),
        }
        self.save_baseline(runner_class, baseline)


class RegressionDetector:
    """Detect performance regressions against baselines."""

    def __init__(
        self,
        pr_lane_threshold_pct: float = 15.0,
        nightly_threshold_pct: float = 20.0,
    ):
        """
        Initialize regression detector.

        Args:
            pr_lane_threshold_pct: PR lane regression threshold (15% default)
            nightly_threshold_pct: Nightly regression threshold (20% default)
        """
        self.pr_lane_threshold_pct = pr_lane_threshold_pct
        self.nightly_threshold_pct = nightly_threshold_pct
        self.regressions: list[dict[str, Any]] = []
        self.improvements: list[dict[str, Any]] = []

    def detect(
        self,
        name: str,
        current_ms: float,
        baseline_ms: float | None,
        is_nightly: bool = False,
    ) -> tuple[RegressionSeverity, str]:
        """
        Detect regression for single benchmark.

        Args:
            name: Benchmark name
            current_ms: Current measured time
            baseline_ms: Previous baseline time (None for first run)
            is_nightly: Whether this is nightly run (stricter threshold)

        Returns:
            (severity, message): Regression severity and explanation
        """
        if baseline_ms is None:
            return (RegressionSeverity.NONE, f"{name}: First run (no baseline)")

        pct_change = ((current_ms - baseline_ms) / baseline_ms) * 100

        threshold = self.nightly_threshold_pct if is_nightly else self.pr_lane_threshold_pct

        if pct_change > threshold:
            severity = RegressionSeverity.FAILURE if is_nightly else RegressionSeverity.WARNING

            regression = {
                "name": name,
                "baseline_ms": baseline_ms,
                "current_ms": current_ms,
                "pct_change": pct_change,
                "severity": severity.value,
            }
            self.regressions.append(regression)

            return (
                severity,
                f"{name}: {pct_change:.1f}% slower (+{current_ms - baseline_ms:.2f}ms)",
            )
        elif pct_change < -5.0:  # Improvement
            improvement = {
                "name": name,
                "baseline_ms": baseline_ms,
                "current_ms": current_ms,
                "pct_change": pct_change,
            }
            self.improvements.append(improvement)

            return (
                RegressionSeverity.NONE,
                f"{name}: {abs(pct_change):.1f}% faster ({pct_change:.2f}ms improvement)",
            )
        else:
            return (
                RegressionSeverity.NONE,
                f"{name}: Within tolerance ({pct_change:+.1f}%)",
            )

    def has_failures(self) -> bool:
        """Check if any regressions detected."""
        return len(self.regressions) > 0

    def summary(self) -> dict[str, Any]:
        """
        Get regression summary for reporting.

        Returns:
            Summary dict with regressions and improvements
        """
        return {
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "regression_details": self.regressions,
            "improvement_details": self.improvements,
        }


# --- CI Integration Helpers ---


def get_ci_runner_class() -> str:
    """
    Detect CI runner class for baseline selection.

    Returns:
        Runner class string (e.g., 'linux-x86_64', 'macos-arm64')
    """
    import platform
    import os

    os_name = platform.system().lower()
    arch = platform.machine()

    # Format: os-arch
    runner_class = f"{os_name}-{arch}"
    return runner_class


def get_ci_is_nightly() -> bool:
    """
    Detect if running in nightly CI context.

    Returns:
        True if nightly run, False if PR lane
    """
    # Check for CI environment variables
    if os.environ.get("CI_NIGHTLY"):
        return True
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        return True
    if "nightly" in os.environ.get("CI_JOB_NAME", "").lower():
        return True

    return False


def should_fail_on_regression() -> bool:
    """
    Determine if regressions should fail CI.

    Returns:
        True if should fail (nightly), False if should warn (PR)
    """
    return get_ci_is_nightly()


import os
