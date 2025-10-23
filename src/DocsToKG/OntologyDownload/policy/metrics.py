# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.policy.metrics",
#   "purpose": "Per-gate telemetry and metrics collection for operational dashboards.",
#   "sections": [
#     {
#       "id": "gatemetric",
#       "name": "GateMetric",
#       "anchor": "class-gatemetric",
#       "kind": "class"
#     },
#     {
#       "id": "gatemetricssnapshot",
#       "name": "GateMetricsSnapshot",
#       "anchor": "class-gatemetricssnapshot",
#       "kind": "class"
#     },
#     {
#       "id": "metricscollector",
#       "name": "MetricsCollector",
#       "anchor": "class-metricscollector",
#       "kind": "class"
#     },
#     {
#       "id": "get-metrics-collector",
#       "name": "get_metrics_collector",
#       "anchor": "function-get-metrics-collector",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Per-gate telemetry and metrics collection for operational dashboards.

Provides counters, histograms, and summaries for policy gate operations.
Designed for integration with monitoring systems (Prometheus, CloudWatch, etc.).
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from DocsToKG.OntologyDownload.policy.registry import get_registry

# ============================================================================
# Metrics Data Models
# ============================================================================


@dataclass(frozen=True)
class GateMetric:
    """A single metric observation from a gate."""

    gate_name: str
    passed: bool  # True = PolicyOK, False = PolicyReject
    elapsed_ms: float
    error_code: Optional[str] = None  # Set if failed


@dataclass
class GateMetricsSnapshot:
    """Snapshot of metrics for a single gate."""

    gate_name: str
    domain: str
    invocations: int = 0
    passes: int = 0
    rejects: int = 0
    pass_rate: float = 0.0
    reject_rate: float = 0.0
    total_ms: float = 0.0
    avg_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    p50_ms: float = 0.0  # Median
    p95_ms: float = 0.0  # 95th percentile
    p99_ms: float = 0.0  # 99th percentile


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """Collects and aggregates gate metrics for dashboarding."""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        """Singleton constructor (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize metrics collector (only once)."""
        if not hasattr(self, "_metrics"):
            self._metrics: Dict[str, List[GateMetric]] = {}
            self._collector_lock = threading.Lock()

    def record_metric(self, metric: GateMetric) -> None:
        """Record a gate metric observation.

        Args:
            metric: GateMetric to record
        """
        with self._collector_lock:
            if metric.gate_name not in self._metrics:
                self._metrics[metric.gate_name] = []
            self._metrics[metric.gate_name].append(metric)

    def get_snapshot(self, gate_name: str) -> Optional[GateMetricsSnapshot]:
        """Get a snapshot of metrics for a gate.

        Args:
            gate_name: Name of gate

        Returns:
            GateMetricsSnapshot with aggregated metrics, or None if no data
        """
        with self._collector_lock:
            if gate_name not in self._metrics or not self._metrics[gate_name]:
                return None

            metrics = self._metrics[gate_name]
            registry = get_registry()

            try:
                gate_metadata = registry.list_gates()[gate_name]
                domain = gate_metadata.domain
            except KeyError:
                domain = "unknown"

            # Calculate stats
            timings = [m.elapsed_ms for m in metrics]
            timings_sorted = sorted(timings)
            n = len(timings_sorted)

            invocations = len(metrics)
            passes = sum(1 for m in metrics if m.passed)
            rejects = invocations - passes

            snapshot = GateMetricsSnapshot(
                gate_name=gate_name,
                domain=domain,
                invocations=invocations,
                passes=passes,
                rejects=rejects,
                pass_rate=passes / invocations if invocations > 0 else 0.0,
                reject_rate=rejects / invocations if invocations > 0 else 0.0,
                total_ms=sum(timings),
                avg_ms=sum(timings) / invocations if invocations > 0 else 0.0,
                min_ms=min(timings),
                max_ms=max(timings),
                p50_ms=timings_sorted[n // 2] if n > 0 else 0.0,
                p95_ms=timings_sorted[int(n * 0.95)] if n > 0 else 0.0,
                p99_ms=timings_sorted[int(n * 0.99)] if n > 0 else 0.0,
            )

            return snapshot

    def get_all_snapshots(self) -> Dict[str, GateMetricsSnapshot]:
        """Get snapshots for all gates with metrics.

        Returns:
            Dict mapping gate names to their snapshots
        """
        with self._collector_lock:
            gate_names = list(self._metrics.keys())

        snapshots = {}
        for gate_name in gate_names:
            snapshot = self.get_snapshot(gate_name)
            if snapshot:
                snapshots[gate_name] = snapshot

        return snapshots

    def get_snapshots_by_domain(self, domain: str) -> Dict[str, GateMetricsSnapshot]:
        """Get snapshots for all gates in a domain.

        Args:
            domain: Gate domain (network, filesystem, extraction, storage, db, config)

        Returns:
            Dict of snapshots for gates in that domain
        """
        all_snapshots = self.get_all_snapshots()
        return {
            name: snapshot for name, snapshot in all_snapshots.items() if snapshot.domain == domain
        }

    def clear_metrics(self, gate_name: Optional[str] = None) -> None:
        """Clear metrics for a gate or all gates.

        Args:
            gate_name: Gate to clear, or None for all
        """
        with self._collector_lock:
            if gate_name:
                if gate_name in self._metrics:
                    self._metrics[gate_name] = []
            else:
                self._metrics.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Summary dict with aggregate stats
        """
        snapshots = self.get_all_snapshots()

        if not snapshots:
            return {
                "total_gates": 0,
                "total_invocations": 0,
                "total_passes": 0,
                "total_rejects": 0,
                "average_pass_rate": 0.0,
                "by_domain": {},
            }

        total_invocations = sum(s.invocations for s in snapshots.values())
        total_passes = sum(s.passes for s in snapshots.values())
        total_rejects = sum(s.rejects for s in snapshots.values())

        # Group by domain
        by_domain: Dict[str, Dict[str, Any]] = {}
        for snapshot in snapshots.values():
            domain_stats = by_domain.setdefault(
                snapshot.domain,
                {
                    "gates": 0,
                    "invocations": 0,
                    "_passes": 0,
                },
            )
            domain_stats["gates"] += 1
            domain_stats["invocations"] += snapshot.invocations
            domain_stats["_passes"] += snapshot.passes

        for stats in by_domain.values():
            invocations = stats["invocations"]
            passes = stats.pop("_passes")
            stats["pass_rate"] = passes / invocations if invocations > 0 else 0.0

        return {
            "total_gates": len(snapshots),
            "total_invocations": total_invocations,
            "total_passes": total_passes,
            "total_rejects": total_rejects,
            "average_pass_rate": (
                total_passes / total_invocations if total_invocations > 0 else 0.0
            ),
            "by_domain": by_domain,
        }


# ============================================================================
# Singleton API
# ============================================================================


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector (singleton).

    Returns:
        MetricsCollector instance
    """
    return MetricsCollector()


__all__ = [
    "GateMetric",
    "GateMetricsSnapshot",
    "MetricsCollector",
    "get_metrics_collector",
]
