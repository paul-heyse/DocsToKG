# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.deployment.monitoring",
#   "purpose": "Production monitoring and alerting system with comprehensive observability.",
#   "sections": [
#     {
#       "id": "alertseverity",
#       "name": "AlertSeverity",
#       "anchor": "class-alertseverity",
#       "kind": "class"
#     },
#     {
#       "id": "metrictype",
#       "name": "MetricType",
#       "anchor": "class-metrictype",
#       "kind": "class"
#     },
#     {
#       "id": "threshold",
#       "name": "Threshold",
#       "anchor": "class-threshold",
#       "kind": "class"
#     },
#     {
#       "id": "alert",
#       "name": "Alert",
#       "anchor": "class-alert",
#       "kind": "class"
#     },
#     {
#       "id": "metric",
#       "name": "Metric",
#       "anchor": "class-metric",
#       "kind": "class"
#     },
#     {
#       "id": "metricscollector",
#       "name": "MetricsCollector",
#       "anchor": "class-metricscollector",
#       "kind": "class"
#     },
#     {
#       "id": "alertmanager",
#       "name": "AlertManager",
#       "anchor": "class-alertmanager",
#       "kind": "class"
#     },
#     {
#       "id": "productionmonitor",
#       "name": "ProductionMonitor",
#       "anchor": "class-productionmonitor",
#       "kind": "class"
#     },
#     {
#       "id": "get-production-monitor",
#       "name": "get_production_monitor",
#       "anchor": "function-get-production-monitor",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Production monitoring and alerting system with comprehensive observability.

Provides real-time metrics collection, anomaly detection, and alert management
for production operational visibility.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric types for different measurement kinds."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass(frozen=True)
class Threshold:
    """Alert threshold configuration."""

    name: str
    metric_name: str
    operator: str  # "gt", "lt", "eq", "neq"
    value: float
    severity: AlertSeverity = AlertSeverity.WARNING
    message_template: str = "{metric_name} {operator} {value}"

    def check(self, metric_value: float) -> bool:
        """Check if threshold is breached.

        Args:
            metric_value: Current metric value

        Returns:
            True if threshold is breached
        """
        if self.operator == "gt":
            return metric_value > self.value
        elif self.operator == "lt":
            return metric_value < self.value
        elif self.operator == "eq":
            return metric_value == self.value
        elif self.operator == "neq":
            return metric_value != self.value
        return False

    def format_message(self, metric_value: float) -> str:
        """Format alert message.

        Args:
            metric_value: Current metric value

        Returns:
            Formatted message
        """
        return self.message_template.format(
            metric_name=self.metric_name, operator=self.operator, value=self.value
        )


@dataclass(frozen=True)
class Alert:
    """Alert instance."""

    threshold: Threshold
    metric_value: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> AlertSeverity:
        """Get alert severity."""
        return self.threshold.severity

    @property
    def message(self) -> str:
        """Get alert message."""
        return self.threshold.format_message(self.metric_value)


@dataclass
class Metric:
    """Single metric measurement."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._metrics: Dict[str, List[Metric]] = {}
        self._lock = threading.Lock()

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric measurement.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for metric
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []

            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                tags=tags or {},
            )
            self._metrics[name].append(metric)

            # Keep only last 1000 measurements per metric
            if len(self._metrics[name]) > 1000:
                self._metrics[name] = self._metrics[name][-1000:]

    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with statistics or None if metric not found
        """
        with self._lock:
            metrics = self._metrics.get(name)
            if not metrics:
                return None

            values = [m.value for m in metrics]
            values.sort()

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "p50": values[len(values) // 2],
                "p95": values[int(len(values) * 0.95)],
                "p99": values[int(len(values) * 0.99)],
            }

    def get_recent_metrics(self, name: str, seconds: float = 300) -> List[Metric]:
        """Get recent metrics within time window.

        Args:
            name: Metric name
            seconds: Time window in seconds

        Returns:
            List of recent metrics
        """
        now = time.time()
        cutoff = now - seconds

        with self._lock:
            metrics = self._metrics.get(name, [])
            return [m for m in metrics if m.timestamp >= cutoff]


class AlertManager:
    """Manages alerts and thresholds."""

    def __init__(self) -> None:
        """Initialize alert manager."""
        self._thresholds: Dict[str, List[Threshold]] = {}
        self._alerts: List[Alert] = []
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def register_threshold(self, threshold: Threshold) -> None:
        """Register alert threshold.

        Args:
            threshold: Threshold to register
        """
        with self._lock:
            if threshold.metric_name not in self._thresholds:
                self._thresholds[threshold.metric_name] = []
            self._thresholds[threshold.metric_name].append(threshold)

    def register_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register alert handler.

        Args:
            handler: Function to call when alert is triggered
        """
        with self._lock:
            self._alert_handlers.append(handler)

    def check_thresholds(self, metric: Metric) -> List[Alert]:
        """Check if metric violates any thresholds.

        Args:
            metric: Metric to check

        Returns:
            List of triggered alerts
        """
        alerts = []

        with self._lock:
            thresholds = self._thresholds.get(metric.name, [])

        for threshold in thresholds:
            if threshold.check(metric.value):
                alert = Alert(
                    threshold=threshold,
                    metric_value=metric.value,
                    context={"tags": metric.tags},
                )
                alerts.append(alert)

                # Call alert handlers
                for handler in self._alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        LOGGER.error(f"Alert handler failed: {e}")

        # Store alerts
        with self._lock:
            self._alerts.extend(alerts)
            # Keep last 10000 alerts
            if len(self._alerts) > 10000:
                self._alerts = self._alerts[-10000:]

        return alerts

    def get_recent_alerts(
        self, severity: Optional[AlertSeverity] = None, seconds: float = 300
    ) -> List[Alert]:
        """Get recent alerts.

        Args:
            severity: Optional filter by severity
            seconds: Time window in seconds

        Returns:
            List of recent alerts
        """
        now = time.time()
        cutoff = now - seconds

        with self._lock:
            alerts = [a for a in self._alerts if a.timestamp >= cutoff]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts


class ProductionMonitor:
    """Main production monitoring system."""

    def __init__(self) -> None:
        """Initialize production monitor."""
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start monitoring system."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        LOGGER.info("Production monitor started")

    def stop_monitoring(self) -> None:
        """Stop monitoring system."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        LOGGER.info("Production monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Perform health checks
                self._check_system_health()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                LOGGER.error(f"Monitor loop error: {e}")

    def _check_system_health(self) -> None:
        """Check system health."""
        # This would be extended with actual health checks
        pass

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags
        """
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
        )
        self.metrics.record_metric(name, value, metric_type, tags)

        # Check thresholds
        self.alerts.check_thresholds(metric)

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status.

        Returns:
            Dictionary with health information
        """
        critical_alerts = self.alerts.get_recent_alerts(severity=AlertSeverity.CRITICAL)
        error_alerts = self.alerts.get_recent_alerts(severity=AlertSeverity.ERROR)

        return {
            "status": "critical" if critical_alerts else "healthy",
            "critical_alerts": len(critical_alerts),
            "error_alerts": len(error_alerts),
            "timestamp": datetime.now().isoformat(),
        }


# Global monitoring instance
_GLOBAL_MONITOR: Optional[ProductionMonitor] = None


def get_production_monitor() -> ProductionMonitor:
    """Get global production monitor instance.

    Returns:
        Global ProductionMonitor instance
    """
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        _GLOBAL_MONITOR = ProductionMonitor()
    return _GLOBAL_MONITOR
