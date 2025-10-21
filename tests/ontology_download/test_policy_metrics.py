"""Tests for policy metrics collection.

Covers:
- Gate metric recording
- Snapshot generation
- Percentile calculations
- Domain filtering
- Summary aggregation
"""

import pytest

from DocsToKG.OntologyDownload.policy.metrics import (
    GateMetric,
    MetricsCollector,
    get_metrics_collector,
)

# ============================================================================
# GateMetric Tests
# ============================================================================


class TestGateMetric:
    """Test GateMetric dataclass."""

    def test_gate_metric_creation_pass(self):
        """GateMetric for pass can be created."""
        metric = GateMetric(
            gate_name="url_gate",
            passed=True,
            elapsed_ms=1.5,
        )
        assert metric.gate_name == "url_gate"
        assert metric.passed is True
        assert metric.elapsed_ms == 1.5
        assert metric.error_code is None

    def test_gate_metric_creation_reject(self):
        """GateMetric for reject can be created."""
        metric = GateMetric(
            gate_name="url_gate",
            passed=False,
            elapsed_ms=0.8,
            error_code="E_HOST_DENY",
        )
        assert metric.gate_name == "url_gate"
        assert metric.passed is False
        assert metric.error_code == "E_HOST_DENY"

    def test_gate_metric_frozen(self):
        """GateMetric is immutable."""
        metric = GateMetric(
            gate_name="url_gate",
            passed=True,
            elapsed_ms=1.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            metric.gate_name = "path_gate"


# ============================================================================
# MetricsCollector Tests
# ============================================================================


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_metrics_collector_singleton(self):
        """MetricsCollector is a singleton."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        assert collector1 is collector2

    def test_record_metric(self):
        """Metrics can be recorded."""
        collector = MetricsCollector()
        collector.clear_metrics("test_gate")

        metric = GateMetric(
            gate_name="test_gate",
            passed=True,
            elapsed_ms=1.0,
        )
        collector.record_metric(metric)

        snapshot = collector.get_snapshot("test_gate")
        assert snapshot is not None
        assert snapshot.invocations == 1
        assert snapshot.passes == 1

    def test_get_snapshot_multiple_metrics(self):
        """Snapshot aggregates multiple metrics."""
        collector = MetricsCollector()
        collector.clear_metrics("agg_gate")

        collector.record_metric(GateMetric("agg_gate", True, 1.0))
        collector.record_metric(GateMetric("agg_gate", True, 2.0))
        collector.record_metric(GateMetric("agg_gate", False, 0.5))

        snapshot = collector.get_snapshot("agg_gate")
        assert snapshot.invocations == 3
        assert snapshot.passes == 2
        assert snapshot.rejects == 1
        assert snapshot.pass_rate == pytest.approx(2 / 3)
        assert snapshot.avg_ms == pytest.approx(1.167, rel=0.01)

    def test_percentiles_calculation(self):
        """Percentiles are calculated correctly."""
        collector = MetricsCollector()
        collector.clear_metrics("perc_gate")

        # Record 100 metrics with timings 0.1 to 10.0
        for i in range(1, 101):
            collector.record_metric(GateMetric("perc_gate", True, float(i) * 0.1))

        snapshot = collector.get_snapshot("perc_gate")
        assert snapshot.p50_ms > 5.0  # Median around 5
        assert snapshot.p95_ms > 9.0  # 95th percentile near end
        assert snapshot.p99_ms > 9.8  # 99th percentile very near end

    def test_get_snapshot_no_data(self):
        """get_snapshot returns None for unknown gate."""
        collector = MetricsCollector()
        collector.clear_metrics()

        snapshot = collector.get_snapshot("unknown_gate")
        assert snapshot is None

    def test_clear_metrics_single(self):
        """clear_metrics can clear a single gate."""
        collector = MetricsCollector()
        collector.clear_metrics("clear_gate")
        collector.record_metric(GateMetric("clear_gate", True, 1.0))

        collector.clear_metrics("clear_gate")
        snapshot = collector.get_snapshot("clear_gate")
        assert snapshot is None

    def test_clear_metrics_all(self):
        """clear_metrics can clear all gates."""
        collector = MetricsCollector()
        collector.record_metric(GateMetric("gate1", True, 1.0))
        collector.record_metric(GateMetric("gate2", True, 2.0))

        collector.clear_metrics()
        assert collector.get_snapshot("gate1") is None
        assert collector.get_snapshot("gate2") is None

    def test_get_all_snapshots(self):
        """get_all_snapshots returns snapshots for all gates."""
        collector = MetricsCollector()
        collector.clear_metrics()
        collector.record_metric(GateMetric("gate1", True, 1.0))
        collector.record_metric(GateMetric("gate2", True, 2.0))

        snapshots = collector.get_all_snapshots()
        assert len(snapshots) >= 2
        assert "gate1" in snapshots
        assert "gate2" in snapshots

    def test_get_summary(self):
        """get_summary returns aggregate stats."""
        collector = MetricsCollector()
        collector.clear_metrics()
        collector.record_metric(GateMetric("gate1", True, 1.0))
        collector.record_metric(GateMetric("gate1", False, 2.0))
        collector.record_metric(GateMetric("gate2", True, 1.5))

        summary = collector.get_summary()
        assert summary["total_gates"] >= 2
        assert summary["total_invocations"] >= 3
        assert summary["total_passes"] >= 2
        assert summary["total_rejects"] >= 1

    def test_get_summary_empty(self):
        """get_summary returns defaults for empty collector."""
        collector = MetricsCollector()
        collector.clear_metrics()

        summary = collector.get_summary()
        assert summary["total_gates"] == 0
        assert summary["total_invocations"] == 0
        assert summary["average_pass_rate"] == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMetricsIntegration:
    """Test metrics integration."""

    def test_get_metrics_collector_singleton(self):
        """get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2

    def test_metrics_workflow(self):
        """End-to-end metrics workflow."""
        collector = get_metrics_collector()
        collector.clear_metrics("workflow_gate")

        # Simulate gate invocations
        collector.record_metric(GateMetric("workflow_gate", True, 1.0))
        collector.record_metric(GateMetric("workflow_gate", True, 1.1))
        collector.record_metric(GateMetric("workflow_gate", False, 0.9, "E_HOST_DENY"))
        collector.record_metric(GateMetric("workflow_gate", True, 1.2))

        snapshot = collector.get_snapshot("workflow_gate")
        assert snapshot.invocations == 4
        assert snapshot.passes == 3
        assert snapshot.rejects == 1
        assert snapshot.pass_rate == pytest.approx(0.75)
        assert snapshot.avg_ms == pytest.approx(1.05, rel=0.01)
