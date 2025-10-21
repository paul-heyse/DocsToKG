"""Tests for Pydantic telemetry models."""

from datetime import datetime
import pytest

from DocsToKG.ContentDownload.fallback.models import (
    TelemetryAttemptRecord,
    TelemetryBatchRecord,
    TelemetryConfig,
    StorageConfig,
    PerformanceMetrics,
    AttemptStatus,
    TierName,
)


class TestTelemetryAttemptRecord:
    """Test telemetry record validation."""

    def test_valid_record_creation(self):
        """Test creating a valid record."""
        record = TelemetryAttemptRecord(
            run_id="run1",
            attempt_id="attempt1",
            tier=TierName.TIER_1,
            host="example.com",
            url="https://example.com/pdf",
            status=AttemptStatus.SUCCESS,
            elapsed_ms=500,
            http_status=200,
        )
        
        assert record.run_id == "run1"
        assert record.status == AttemptStatus.SUCCESS

    def test_invalid_url(self):
        """Test URL validation."""
        with pytest.raises(ValueError, match="URL must start with"):
            TelemetryAttemptRecord(
                run_id="run1",
                attempt_id="attempt1",
                tier=TierName.TIER_1,
                host="example.com",
                url="not-a-url",  # Invalid
                status=AttemptStatus.SUCCESS,
                elapsed_ms=500,
            )

    def test_success_requires_http_200(self):
        """Test that success requires HTTP 200."""
        with pytest.raises(ValueError, match="Success status requires HTTP 200"):
            TelemetryAttemptRecord(
                run_id="run1",
                attempt_id="attempt1",
                tier=TierName.TIER_1,
                host="example.com",
                url="https://example.com",
                status=AttemptStatus.SUCCESS,
                elapsed_ms=500,
                http_status=404,  # Invalid for success
            )


class TestTelemetryBatchRecord:
    """Test batch record validation."""

    def test_valid_batch(self):
        """Test creating a valid batch."""
        records = [
            TelemetryAttemptRecord(
                run_id="run1",
                attempt_id=f"attempt{i}",
                tier=TierName.TIER_1,
                host="example.com",
                url="https://example.com/pdf",
                status=AttemptStatus.SUCCESS,
                elapsed_ms=500,
                http_status=200,
            )
            for i in range(3)
        ]
        
        batch = TelemetryBatchRecord(
            batch_id="batch1",
            records=records,
            count=3,
        )
        
        assert len(batch.records) == 3

    def test_count_validation(self):
        """Test that count must match records."""
        records = [
            TelemetryAttemptRecord(
                run_id="run1",
                attempt_id="attempt1",
                tier=TierName.TIER_1,
                host="example.com",
                url="https://example.com/pdf",
                status=AttemptStatus.SUCCESS,
                elapsed_ms=500,
                http_status=200,
            )
        ]
        
        with pytest.raises(ValueError, match="Count must match"):
            TelemetryBatchRecord(
                batch_id="batch1",
                records=records,
                count=2,  # Wrong count
            )


class TestStorageConfig:
    """Test storage configuration validation."""

    def test_valid_config(self):
        """Test valid storage config."""
        config = StorageConfig(
            storage_type="sqlite",
            path="data/manifest.db",
            batch_size=100,
        )
        
        assert config.storage_type == "sqlite"

    def test_batch_size_bounds(self):
        """Test batch size validation."""
        with pytest.raises(ValueError):
            StorageConfig(batch_size=0)  # Too small
        
        with pytest.raises(ValueError):
            StorageConfig(batch_size=20000)  # Too large


class TestPerformanceMetrics:
    """Test performance metrics validation."""

    def test_valid_metrics(self):
        """Test valid metrics."""
        metrics = PerformanceMetrics(
            total_attempts=100,
            success_count=85,
            success_rate=0.85,
            avg_latency_ms=1000,
            p50_latency_ms=800,
            p95_latency_ms=2000,
            p99_latency_ms=3000,
            error_count=10,
            timeout_count=5,
        )
        
        assert metrics.success_rate == 0.85

    def test_percentile_ordering(self):
        """Test that percentiles must be ordered."""
        with pytest.raises(ValueError, match="Percentiles must be in ascending order"):
            PerformanceMetrics(
                total_attempts=100,
                success_count=85,
                success_rate=0.85,
                avg_latency_ms=1000,
                p50_latency_ms=3000,  # Out of order
                p95_latency_ms=2000,
                p99_latency_ms=1000,
                error_count=10,
                timeout_count=5,
            )


class TestTelemetryConfig:
    """Test complete telemetry configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = TelemetryConfig()
        
        assert config.enabled is True
        assert config.log_level == "INFO"

    def test_custom_config(self):
        """Test custom configuration."""
        storage_config = StorageConfig(path="custom/path.db")
        config = TelemetryConfig(storage=storage_config)
        
        assert config.storage.path == "custom/path.db"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
