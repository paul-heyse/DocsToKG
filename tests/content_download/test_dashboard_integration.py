"""Tests for Dashboard Integration"""

from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest

from DocsToKG.ContentDownload.fallback.dashboard_integration import (
    DashboardExporter,
    RealTimeMonitor,
    MetricsSnapshot,
)


class TestMetricsSnapshot:
    """Test metrics snapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a metrics snapshot."""
        snapshot = MetricsSnapshot(
            timestamp="2025-10-21T12:00:00",
            total_attempts=100,
            success_rate=0.85,
            avg_latency_ms=1200,
            p50_latency_ms=1000,
            p95_latency_ms=2000,
            p99_latency_ms=2500,
            top_tier="tier_1",
            top_tier_success_rate=0.95,
        )
        
        assert snapshot.total_attempts == 100
        assert snapshot.success_rate == 0.85


class TestDashboardExporter:
    """Test dashboard export functionality."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        with patch("DocsToKG.ContentDownload.fallback.dashboard_integration.get_telemetry_storage") as mock_get:
            mock_storage = Mock()
            mock_storage.load_records.return_value = [
                {"tier": "tier_1", "host": "unpaywall.org", "outcome": "success", "elapsed_ms": 850},
                {"tier": "tier_1", "host": "arxiv.org", "outcome": "error", "reason": "timeout", "elapsed_ms": 2000},
                {"tier": "tier_2", "host": "doi.org", "outcome": "success", "elapsed_ms": 1200},
            ]
            mock_get.return_value = mock_storage
            yield mock_storage

    def test_export_for_grafana(self, mock_storage):
        """Test Grafana export."""
        exporter = DashboardExporter()
        result = exporter.export_for_grafana()
        
        assert "dashboard" in result
        assert "title" in result["dashboard"]
        assert "panels" in result["dashboard"]

    def test_export_for_prometheus(self, mock_storage):
        """Test Prometheus export."""
        exporter = DashboardExporter()
        result = exporter.export_for_prometheus()
        
        assert "fallback_total_attempts" in result
        assert "fallback_success_rate" in result
        assert "HELP" in result

    def test_export_timeseries(self, mock_storage):
        """Test timeseries export."""
        exporter = DashboardExporter()
        result = exporter.export_timeseries()
        
        assert len(result) > 0
        assert isinstance(result[0], MetricsSnapshot)

    def test_export_dashboard_json(self, mock_storage):
        """Test dashboard JSON export."""
        exporter = DashboardExporter()
        result = exporter.export_dashboard_json()
        
        assert "title" in result
        assert "metrics" in result


class TestRealTimeMonitor:
    """Test real-time monitoring."""

    @pytest.fixture
    def mock_monitor(self):
        """Create mock monitor."""
        with patch("DocsToKG.ContentDownload.fallback.dashboard_integration.get_telemetry_storage") as mock_get:
            mock_storage = Mock()
            mock_storage.load_records.return_value = [
                {"tier": "tier_1", "host": "unpaywall.org", "outcome": "success", "elapsed_ms": 850},
                {"tier": "tier_1", "host": "arxiv.org", "outcome": "success", "elapsed_ms": 900},
            ]
            mock_get.return_value = mock_storage
            yield RealTimeMonitor()

    def test_get_live_metrics(self, mock_monitor):
        """Test getting live metrics."""
        metrics = mock_monitor.get_live_metrics()
        
        assert "timestamp" in metrics
        assert "record_count" in metrics
        assert "metrics" in metrics

    def test_get_trend(self, mock_monitor):
        """Test getting trend analysis."""
        trend = mock_monitor.get_trend()
        
        assert "period" in trend
        assert "first_half" in trend
        assert "second_half" in trend
        assert "success_rate_change" in trend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
