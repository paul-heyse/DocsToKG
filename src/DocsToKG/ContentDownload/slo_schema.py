# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.slo_schema",
#   "purpose": "SLO definitions and thresholds for idempotency system",
#   "sections": [
#     {
#       "id": "slothreshold",
#       "name": "SLOThreshold",
#       "anchor": "class-slothreshold",
#       "kind": "class"
#     },
#     {
#       "id": "slometric",
#       "name": "SLOMetric",
#       "anchor": "class-slometric",
#       "kind": "class"
#     },
#     {
#       "id": "get-slo-definitions",
#       "name": "get_slo_definitions",
#       "anchor": "function-get-slo-definitions",
#       "kind": "function"
#     },
#     {
#       "id": "evaluate-slo-status",
#       "name": "evaluate_slo_status",
#       "anchor": "function-evaluate-slo-status",
#       "kind": "function"
#     },
#     {
#       "id": "calculate-error-budget-remaining",
#       "name": "calculate_error_budget_remaining",
#       "anchor": "function-calculate-error-budget-remaining",
#       "kind": "function"
#     },
#     {
#       "id": "should-alert",
#       "name": "should_alert",
#       "anchor": "function-should-alert",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""SLO definitions and thresholds for idempotency system.

Defines Service Level Objectives for monitoring job completion, timing,
crash recovery, and operational health.

SLOs:
  1. Job Completion Rate: 99.5% of jobs reach FINALIZED state
  2. Mean Time to Complete: p50 < 30s, p95 < 120s
  3. Crash Recovery Success: 99.9% of crashes recovered
  4. Lease Acquisition Latency: p99 < 100ms
  5. Operation Replay Rate: < 5% of operations replayed (indicates stability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# === SLO THRESHOLD CONSTANTS ===

# 1. Job Completion Rate SLO
JOB_COMPLETION_RATE_TARGET = 0.995  # 99.5%
JOB_COMPLETION_ERROR_BUDGET = 0.005  # 0.5% per day
JOB_COMPLETION_FAILURE_STATES = {"FAILED", "SKIPPED_DUPLICATE"}

# 2. Mean Time to Complete SLO
TIME_TO_COMPLETE_P50_TARGET_MS = 30000  # 30 seconds
TIME_TO_COMPLETE_P95_TARGET_MS = 120000  # 2 minutes
TIME_TO_COMPLETE_P99_TARGET_MS = 300000  # 5 minutes

# 3. Crash Recovery Success SLO
CRASH_RECOVERY_SUCCESS_TARGET = 0.999  # 99.9%
CRASH_RECOVERY_ERROR_BUDGET = 0.001  # 0.1%

# 4. Lease Acquisition Latency SLO
LEASE_ACQUISITION_P50_TARGET_MS = 10  # 10ms
LEASE_ACQUISITION_P99_TARGET_MS = 100  # 100ms

# 5. Operation Replay Rate SLO
OPERATION_REPLAY_RATE_TARGET = 0.05  # 5% max (indicates system is stable)
OPERATION_REPLAY_ERROR_BUDGET = 0.10  # 10% threshold before alert

# === SLO WINDOW CONSTANTS ===

SLO_WINDOW_1DAY_SECONDS = 86400
SLO_WINDOW_7DAY_SECONDS = 604800
SLO_WINDOW_30DAY_SECONDS = 2592000


# === SLO DATACLASSES ===


@dataclass
class SLOThreshold:
    """Single SLO threshold definition."""

    name: str
    metric: str
    target: float
    error_budget: float
    window_seconds: int
    unit: str  # e.g., "count", "ms", "percent"


@dataclass
class SLOMetric:
    """Computed SLO metric result."""

    name: str
    actual_value: float
    target_value: float
    error_budget: float
    status: str  # "pass", "warning", "fail"
    details: Dict[str, str]


# === SLO DEFINITIONS ===


def get_slo_definitions() -> List[SLOThreshold]:
    """Get all active SLO definitions."""
    return [
        SLOThreshold(
            name="Job Completion Rate",
            metric="job_completion_rate",
            target=JOB_COMPLETION_RATE_TARGET,
            error_budget=JOB_COMPLETION_ERROR_BUDGET,
            window_seconds=SLO_WINDOW_1DAY_SECONDS,
            unit="percent",
        ),
        SLOThreshold(
            name="Mean Time to Complete (p50)",
            metric="time_to_complete_p50",
            target=TIME_TO_COMPLETE_P50_TARGET_MS,
            error_budget=5000,  # 5 second allowance
            window_seconds=SLO_WINDOW_1DAY_SECONDS,
            unit="ms",
        ),
        SLOThreshold(
            name="Mean Time to Complete (p95)",
            metric="time_to_complete_p95",
            target=TIME_TO_COMPLETE_P95_TARGET_MS,
            error_budget=20000,  # 20 second allowance
            window_seconds=SLO_WINDOW_1DAY_SECONDS,
            unit="ms",
        ),
        SLOThreshold(
            name="Crash Recovery Success Rate",
            metric="crash_recovery_success_rate",
            target=CRASH_RECOVERY_SUCCESS_TARGET,
            error_budget=CRASH_RECOVERY_ERROR_BUDGET,
            window_seconds=SLO_WINDOW_7DAY_SECONDS,
            unit="percent",
        ),
        SLOThreshold(
            name="Lease Acquisition Latency (p99)",
            metric="lease_acquisition_p99",
            target=LEASE_ACQUISITION_P99_TARGET_MS,
            error_budget=50,  # 50ms allowance
            window_seconds=SLO_WINDOW_1DAY_SECONDS,
            unit="ms",
        ),
        SLOThreshold(
            name="Operation Replay Rate",
            metric="operation_replay_rate",
            target=OPERATION_REPLAY_RATE_TARGET,
            error_budget=OPERATION_REPLAY_ERROR_BUDGET,
            window_seconds=SLO_WINDOW_7DAY_SECONDS,
            unit="percent",
        ),
    ]


# === SLO EVALUATION HELPERS ===


def evaluate_slo_status(actual: float, target: float, error_budget: float) -> str:
    """Determine SLO status: pass, warning, or fail.

    Parameters
    ----------
    actual : float
        Actual metric value
    target : float
        Target threshold
    error_budget : float
        Allowable deviation from target

    Returns
    -------
    str
        Status: "pass", "warning", or "fail"
    """
    deviation = abs(actual - target)

    if deviation <= error_budget:
        return "pass"
    elif deviation <= (error_budget * 2):
        return "warning"
    else:
        return "fail"


def calculate_error_budget_remaining(
    actual: float,
    target: float,
    error_budget: float,
) -> float:
    """Calculate remaining error budget in percentage.

    Parameters
    ----------
    actual : float
        Actual metric value
    target : float
        Target threshold
    error_budget : float
        Total error budget

    Returns
    -------
    float
        Percentage of error budget remaining (0-100)
    """
    deviation = abs(actual - target)
    if deviation >= error_budget:
        return 0.0
    return 100.0 * (1.0 - (deviation / error_budget))


# === SLO MONITORING ALERTS ===


def should_alert(status: str, error_budget_remaining: float) -> bool:
    """Determine if alert should be triggered.

    Parameters
    ----------
    status : str
        SLO status ("pass", "warning", "fail")
    error_budget_remaining : float
        Percentage of error budget remaining

    Returns
    -------
    bool
        True if alert should be sent
    """
    # Alert on failure or if less than 25% error budget remains
    return status == "fail" or error_budget_remaining < 25.0
