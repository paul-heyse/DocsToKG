# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.fallback.dashboard_integration",
#   "purpose": "Dashboard Integration Utilities.",
#   "sections": [
#     {
#       "id": "metricssnapshot",
#       "name": "MetricsSnapshot",
#       "anchor": "class-metricssnapshot",
#       "kind": "class"
#     },
#     {
#       "id": "dashboardexporter",
#       "name": "DashboardExporter",
#       "anchor": "class-dashboardexporter",
#       "kind": "class"
#     },
#     {
#       "id": "realtimemonitor",
#       "name": "RealTimeMonitor",
#       "anchor": "class-realtimemonitor",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Dashboard Integration Utilities

Provides utilities for integrating telemetry with:
  - Grafana datasources
  - Prometheus metrics
  - Time-series visualization
  - Real-time monitoring dashboards
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from DocsToKG.ContentDownload.fallback.cli_commands import TelemetryAnalyzer
from DocsToKG.ContentDownload.fallback.telemetry_storage import get_telemetry_storage


@dataclass
class MetricsSnapshot:
    """Represents a snapshot of metrics at a point in time."""

    timestamp: str
    total_attempts: int
    success_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    top_tier: str
    top_tier_success_rate: float


class DashboardExporter:
    """Exports telemetry data for dashboard consumption."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize exporter with storage path."""
        self.storage = get_telemetry_storage(storage_path)

    def export_for_grafana(self, period: str = "24h") -> Dict[str, Any]:
        """Export metrics in Grafana-compatible JSON format.

        Returns:
            Grafana-compatible dashboard data
        """
        records = self.storage.load_records(period)
        analyzer = TelemetryAnalyzer(records)

        overall = analyzer.get_overall_stats()
        tier_stats = analyzer.get_tier_stats()

        return {
            "dashboard": {
                "title": "Fallback Strategy Metrics",
                "panels": [
                    {
                        "title": "Success Rate",
                        "type": "gauge",
                        "targets": [
                            {
                                "target": "success_rate",
                                "value": overall.get("success_rate", 0) * 100,
                            }
                        ],
                    },
                    {
                        "title": "Average Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "target": "avg_latency_ms",
                                "value": overall.get("avg_elapsed_ms", 0),
                            }
                        ],
                    },
                    {
                        "title": "Tier Performance",
                        "type": "table",
                        "targets": [
                            {
                                "tier": tier,
                                "success_rate": stats["success_rate"] * 100,
                                "attempts": stats["attempts"],
                            }
                            for tier, stats in tier_stats.items()
                        ],
                    },
                ],
            }
        }

    def export_for_prometheus(self, period: str = "24h") -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics text
        """
        records = self.storage.load_records(period)
        analyzer = TelemetryAnalyzer(records)

        overall = analyzer.get_overall_stats()
        tier_stats = analyzer.get_tier_stats()
        source_stats = analyzer.get_source_stats()

        lines = [
            "# HELP fallback_total_attempts Total resolution attempts",
            f"fallback_total_attempts {overall.get('total_attempts', 0)}",
            "",
            "# HELP fallback_success_rate Resolution success rate",
            f"fallback_success_rate {overall.get('success_rate', 0)}",
            "",
            "# HELP fallback_latency_avg Average resolution latency in milliseconds",
            f"fallback_latency_avg {overall.get('avg_elapsed_ms', 0)}",
            "",
            "# HELP fallback_latency_p95 P95 resolution latency in milliseconds",
            f"fallback_latency_p95 {overall.get('p95_elapsed_ms', 0)}",
            "",
            "# HELP fallback_latency_p99 P99 resolution latency in milliseconds",
            f"fallback_latency_p99 {overall.get('p99_elapsed_ms', 0)}",
            "",
        ]

        # Per-tier metrics
        for tier, stats in tier_stats.items():
            lines.append(f'fallback_tier_success_rate{{tier="{tier}"}} {stats["success_rate"]}')
            lines.append(f'fallback_tier_attempts{{tier="{tier}"}} {stats["attempts"]}')

        lines.append("")

        # Per-source metrics
        for source, stats in source_stats.items():
            lines.append(
                f'fallback_source_success_rate{{source="{source}"}} {stats["success_rate"]}'
            )
            lines.append(f'fallback_source_error_rate{{source="{source}"}} {stats["error_rate"]}')
            lines.append(
                f'fallback_source_timeout_rate{{source="{source}"}} {stats["timeout_rate"]}'
            )

        return "\n".join(lines)

    def export_timeseries(self, period: str = "24h") -> List[MetricsSnapshot]:
        """Export metrics as time-series snapshots.

        Returns:
            List of metrics snapshots over time
        """
        records = self.storage.load_records(period)
        analyzer = TelemetryAnalyzer(records)

        overall = analyzer.get_overall_stats()
        tier_stats = analyzer.get_tier_stats()

        top_tier = max(tier_stats.items(), key=lambda x: x[1]["success_rate"])[0]
        top_tier_success = tier_stats[top_tier]["success_rate"]

        return [
            MetricsSnapshot(
                timestamp=datetime.utcnow().isoformat(),
                total_attempts=overall.get("total_attempts", 0),
                success_rate=overall.get("success_rate", 0),
                avg_latency_ms=overall.get("avg_elapsed_ms", 0),
                p50_latency_ms=overall.get("p50_elapsed_ms", 0),
                p95_latency_ms=overall.get("p95_elapsed_ms", 0),
                p99_latency_ms=overall.get("p99_elapsed_ms", 0),
                top_tier=top_tier,
                top_tier_success_rate=top_tier_success,
            )
        ]

    def export_dashboard_json(self, period: str = "24h") -> str:
        """Export complete dashboard JSON for visualization tools.

        Returns:
            JSON string for dashboard consumption
        """
        records = self.storage.load_records(period)
        analyzer = TelemetryAnalyzer(records)

        dashboard_data = {
            "title": "Fallback Strategy Dashboard",
            "period": period,
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": {
                "overall": analyzer.get_overall_stats(),
                "by_tier": analyzer.get_tier_stats(),
                "by_source": analyzer.get_source_stats(),
                "top_failures": analyzer.get_failure_reasons(top_n=10),
            },
        }

        return json.dumps(dashboard_data, indent=2)


class RealTimeMonitor:
    """Provides real-time monitoring capabilities."""

    def __init__(self, storage_path: Optional[str] = None, poll_interval_s: int = 5):
        """Initialize monitor.

        Args:
            storage_path: Path to telemetry storage
            poll_interval_s: Polling interval in seconds
        """
        self.storage = get_telemetry_storage(storage_path)
        self.poll_interval_s = poll_interval_s
        self._last_record_count = 0

    def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live metrics.

        Returns:
            Current metrics snapshot
        """
        records = self.storage.load_records(period="1h")
        analyzer = TelemetryAnalyzer(records)

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "record_count": len(records),
            "new_records": len(records) - self._last_record_count,
            "metrics": analyzer.get_overall_stats(),
        }

        self._last_record_count = len(records)
        return metrics

    def get_trend(self, period: str = "24h") -> Dict[str, Any]:
        """Get trend analysis over period.

        Returns:
            Trend data and changes
        """
        records = self.storage.load_records(period)
        analyzer = TelemetryAnalyzer(records)

        # Split into two halves for comparison
        mid_point = len(records) // 2
        first_half = TelemetryAnalyzer(records[:mid_point])
        second_half = TelemetryAnalyzer(records[mid_point:])

        first_stats = first_half.get_overall_stats()
        second_stats = second_half.get_overall_stats()

        return {
            "period": period,
            "first_half": first_stats,
            "second_half": second_stats,
            "success_rate_change": second_stats.get("success_rate", 0)
            - first_stats.get("success_rate", 0),
            "latency_change_percent": (
                (second_stats.get("avg_elapsed_ms", 0) - first_stats.get("avg_elapsed_ms", 0))
                / max(first_stats.get("avg_elapsed_ms", 1), 1)
                * 100
            ),
        }


__all__ = [
    "MetricsSnapshot",
    "DashboardExporter",
    "RealTimeMonitor",
]
