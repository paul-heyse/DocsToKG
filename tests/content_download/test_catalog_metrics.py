"""Tests for catalog metrics and OpenTelemetry integration.

Tests OTel counter initialization and metric recording.
"""

from __future__ import annotations

import pytest

from DocsToKG.ContentDownload.catalog.metrics import (
    CatalogMetrics,
    get_catalog_metrics,
    reset_metrics_for_tests,
)


class TestCatalogMetrics:
    """Test catalog metrics functionality."""

    def teardown_method(self):
        """Reset metrics after each test."""
        reset_metrics_for_tests()

    def test_metrics_singleton(self):
        """Test that get_catalog_metrics returns singleton."""
        metrics1 = get_catalog_metrics()
        metrics2 = get_catalog_metrics()

        assert metrics1 is metrics2

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CatalogMetrics()

        # Should not be initialized yet
        assert not metrics._initialized

        # Trigger initialization
        metrics._init_meters()

        # Should be marked as initialized
        assert metrics._initialized

    def test_record_dedup_hit_no_otel(self):
        """Test recording dedup hit without OTel installed."""
        metrics = CatalogMetrics()

        # Should not raise even if OTel unavailable
        try:
            metrics.record_dedup_hit(count=5)
            metrics.record_dedup_hit(count=1, resolver="unpaywall")
            metrics.record_dedup_hit()  # Default count=1
        except Exception as e:
            pytest.fail(f"record_dedup_hit raised: {e}")

    def test_record_gc_removed_no_otel(self):
        """Test recording GC removed without OTel."""
        metrics = CatalogMetrics()

        try:
            metrics.record_gc_removed(count=10)
            metrics.record_gc_removed(count=5, retention_days=30)
            metrics.record_gc_removed()  # Default count=1
        except Exception as e:
            pytest.fail(f"record_gc_removed raised: {e}")

    def test_record_verify_failure_no_otel(self):
        """Test recording verify failures without OTel."""
        metrics = CatalogMetrics()

        try:
            metrics.record_verify_failure(count=2)
            metrics.record_verify_failure(count=1, record_id=42)
            metrics.record_verify_failure()  # Default count=1
        except Exception as e:
            pytest.fail(f"record_verify_failure raised: {e}")

    def test_metrics_with_attributes(self):
        """Test metrics recording with attributes."""
        metrics = CatalogMetrics()

        # Should accept arbitrary attributes
        try:
            metrics.record_dedup_hit(
                count=3,
                resolver="unpaywall",
                layout="cas",
                status="success",
            )
            metrics.record_gc_removed(
                count=5,
                orphan_ttl_days=7,
                root_dir="/data/docs",
            )
            metrics.record_verify_failure(
                count=1,
                record_id=123,
                reason="hash_mismatch",
            )
        except Exception as e:
            pytest.fail(f"Metrics with attributes raised: {e}")

    def test_global_metrics_instance(self):
        """Test global metrics instance."""
        reset_metrics_for_tests()

        metrics1 = get_catalog_metrics()
        assert metrics1 is not None

        metrics1.record_dedup_hit(count=1)

        metrics2 = get_catalog_metrics()
        assert metrics1 is metrics2

        reset_metrics_for_tests()

        metrics3 = get_catalog_metrics()
        assert metrics1 is not metrics3


class TestMetricsIntegration:
    """Integration tests for metrics with catalog operations."""

    def teardown_method(self):
        """Reset metrics after each test."""
        reset_metrics_for_tests()

    def test_dedup_hit_recording_pattern(self):
        """Test typical dedup hit recording pattern."""
        metrics = get_catalog_metrics()

        # Simulate finding dedup hits in a batch
        dedup_count = 5
        metrics.record_dedup_hit(count=dedup_count, resolver="unpaywall")

        # Should complete without error
        assert True

    def test_gc_workflow_metrics(self):
        """Test typical GC workflow metrics."""
        metrics = get_catalog_metrics()

        # Simulate GC workflow
        orphan_files_found = 12
        files_deleted = 10

        # Record findings
        metrics.record_gc_removed(count=files_deleted, reason="orphan")

        # Should complete without error
        assert True

    def test_verify_workflow_metrics(self):
        """Test typical verification workflow metrics."""
        metrics = get_catalog_metrics()

        # Simulate verification workflow
        total_verified = 100
        failures = 3

        # Record failures
        metrics.record_verify_failure(count=failures, reason="hash_mismatch")

        # Should complete without error
        assert True

    def test_batch_operations_metrics(self):
        """Test metrics during batch operations."""
        metrics = get_catalog_metrics()

        # Simulate batch import
        for i in range(10):
            if i % 3 == 0:
                metrics.record_dedup_hit(count=1)

        # Simulate batch GC
        metrics.record_gc_removed(count=5)

        # Simulate batch verification
        metrics.record_verify_failure(count=0)  # No failures

        # Should complete without error
        assert True


class TestMetricsResilience:
    """Test metrics resilience and error handling."""

    def teardown_method(self):
        """Reset metrics after each test."""
        reset_metrics_for_tests()

    def test_multiple_metric_instances(self):
        """Test multiple CatalogMetrics instances."""
        metrics1 = CatalogMetrics()
        metrics2 = CatalogMetrics()

        # Each should track initialization independently
        metrics1._init_meters()
        assert metrics1._initialized
        assert not metrics2._initialized

        metrics2._init_meters()
        assert metrics2._initialized

    def test_repeated_initialization(self):
        """Test that repeated initialization is safe."""
        metrics = CatalogMetrics()

        # Initialize multiple times
        metrics._init_meters()
        metrics._init_meters()
        metrics._init_meters()

        # Should still work
        metrics.record_dedup_hit(count=1)

        assert True

    def test_zero_count_recording(self):
        """Test recording zero counts."""
        metrics = get_catalog_metrics()

        # Recording zero should be safe
        metrics.record_dedup_hit(count=0)
        metrics.record_gc_removed(count=0)
        metrics.record_verify_failure(count=0)

        assert True

    def test_large_count_recording(self):
        """Test recording large counts."""
        metrics = get_catalog_metrics()

        # Record large numbers
        metrics.record_dedup_hit(count=1_000_000)
        metrics.record_gc_removed(count=10_000)
        metrics.record_verify_failure(count=999)

        assert True

    def test_special_characters_in_attributes(self):
        """Test attributes with special characters."""
        metrics = get_catalog_metrics()

        # Record with special characters in attributes
        metrics.record_dedup_hit(
            count=1,
            resolver="test/resolver",
            path="/data/docs/file-with-dash.pdf",
            error="sha256 mismatch: expected=abc, got=def",
        )

        assert True
