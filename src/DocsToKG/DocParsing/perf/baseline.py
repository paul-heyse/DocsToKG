# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.perf.baseline",
#   "purpose": "Baseline comparison helpers for DocParsing performance runs.",
#   "sections": [
#     {
#       "id": "baseline-metrics",
#       "name": "BaselineMetrics",
#       "anchor": "class-baseline-metrics",
#       "kind": "class"
#     },
#     {
#       "id": "comparison-result",
#       "name": "ComparisonResult",
#       "anchor": "class-comparison-result",
#       "kind": "class"
#     },
#     {
#       "id": "load-baseline",
#       "name": "load_baseline",
#       "anchor": "function-load-baseline",
#       "kind": "function"
#     },
#     {
#       "id": "compare-metrics",
#       "name": "compare_metrics",
#       "anchor": "function-compare-metrics",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Baseline comparison helpers for DocParsing performance runs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from DocsToKG.DocParsing.perf.runner import StageMetrics


@dataclass(slots=True)
class BaselineMetrics:
    """Persisted metrics from a previous run used for comparison."""

    stage: str
    wall_time_s: float
    cpu_time_s: float
    max_rss_bytes: int | None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> BaselineMetrics:
        """Initialise from a JSON payload."""

        return cls(
            stage=payload["stage"],
            wall_time_s=float(payload["wall_time_s"]),
            cpu_time_s=float(payload["cpu_time_s"]),
            max_rss_bytes=(
                int(payload["max_rss_bytes"])
                if payload.get("max_rss_bytes") is not None
                else None
            ),
        )


@dataclass(slots=True)
class ComparisonResult:
    """Comparison output that surfaces regressions and improvements."""

    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "unchanged": list(self.unchanged),
        }


def load_baseline(path: Path) -> dict[str, BaselineMetrics]:
    """Load baseline metrics from ``path``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages", [])
    baseline: dict[str, BaselineMetrics] = {}
    for stage_payload in stages:
        metrics = BaselineMetrics.from_json(stage_payload)
        baseline[metrics.stage] = metrics
    return baseline


def _evaluate_ratio(current: float, previous: float) -> float:
    """Return fractional delta between ``current`` and ``previous``."""

    if previous == 0:
        return float("inf") if current > 0 else 0.0
    return (current - previous) / previous


def compare_metrics(
    *,
    current: Iterable[StageMetrics],
    baseline: dict[str, BaselineMetrics],
    wall_threshold: float,
    cpu_threshold: float,
    rss_threshold: float,
) -> ComparisonResult:
    """Compare ``current`` metrics against ``baseline`` with thresholds."""

    result = ComparisonResult()

    for metrics in current:
        base = baseline.get(metrics.stage)
        if not base:
            continue

        wall_delta = _evaluate_ratio(metrics.wall_time_s, base.wall_time_s)
        cpu_delta = _evaluate_ratio(metrics.cpu_time_s, base.cpu_time_s)
        rss_delta = 0.0
        if metrics.max_rss_bytes is not None and base.max_rss_bytes is not None:
            rss_delta = _evaluate_ratio(metrics.max_rss_bytes, base.max_rss_bytes)

        regressions = []
        if wall_delta > wall_threshold:
            regressions.append(
                f"wall_time_s +{wall_delta*100:.1f}% (threshold {wall_threshold*100:.1f}%)"
            )
        if cpu_delta > cpu_threshold:
            regressions.append(
                f"cpu_time_s +{cpu_delta*100:.1f}% (threshold {cpu_threshold*100:.1f}%)"
            )
        if (
            metrics.max_rss_bytes is not None
            and base.max_rss_bytes is not None
            and rss_delta > rss_threshold
        ):
            regressions.append(
                f"max_rss_bytes +{rss_delta*100:.1f}% (threshold {rss_threshold*100:.1f}%)"
            )

        if regressions:
            result.regressions.append(f"{metrics.stage}: {'; '.join(regressions)}")
            continue

        improvements_detected = False
        if wall_delta < -wall_threshold:
            result.improvements.append(
                f"{metrics.stage}: wall_time_s {wall_delta*100:.1f}%"
            )
            improvements_detected = True
        if cpu_delta < -cpu_threshold:
            result.improvements.append(
                f"{metrics.stage}: cpu_time_s {cpu_delta*100:.1f}%"
            )
            improvements_detected = True
        if (
            metrics.max_rss_bytes is not None
            and base.max_rss_bytes is not None
            and rss_delta < -rss_threshold
        ):
            result.improvements.append(
                f"{metrics.stage}: max_rss_bytes {rss_delta*100:.1f}%"
            )
            improvements_detected = True

        if not improvements_detected:
            result.unchanged.append(metrics.stage)

    return result
