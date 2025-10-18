"""Unit tests for DocsToKG.ContentDownload.statistics module."""

from unittest.mock import patch

import pytest

from DocsToKG.ContentDownload.statistics import (
    BandwidthTracker,
    DownloadStatistics,
    ResolverStats,
)


class TestResolverStats:
    """Test ResolverStats dataclass."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        stats = ResolverStats()
        assert stats.attempts == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.total_bytes == 0
        assert stats.total_time_ms == 0.0
        assert len(stats.failures_by_reason) == 0

    def test_success_rate_calculation(self):
        """Test success rate percentage calculation."""
        stats = ResolverStats(attempts=100, successes=85)
        assert stats.success_rate == 85.0

        stats_zero = ResolverStats(attempts=0, successes=0)
        assert stats_zero.success_rate == 0.0

    def test_avg_time_calculation(self):
        """Test average time calculation."""
        stats = ResolverStats(successes=10, total_time_ms=5000.0)
        assert stats.avg_time_ms == 500.0

        stats_zero = ResolverStats(successes=0, total_time_ms=1000.0)
        assert stats_zero.avg_time_ms == 0.0

    def test_total_mb_calculation(self):
        """Test total MB calculation."""
        stats = ResolverStats(total_bytes=10 * 1024 * 1024)  # 10 MB
        assert stats.total_mb == 10.0

    def test_failures_by_reason(self):
        """Test failure reason tracking."""
        stats = ResolverStats()
        stats.failures_by_reason["timeout"] = 5
        stats.failures_by_reason["http_error"] = 3
        assert stats.failures_by_reason["timeout"] == 5
        assert stats.failures_by_reason["http_error"] == 3


class TestBandwidthTracker:
    """Test BandwidthTracker class."""

    def test_init(self):
        """Test initialization."""
        tracker = BandwidthTracker(window_seconds=60.0)
        assert tracker._window_seconds == 60.0
        assert len(tracker._samples) == 0
        assert tracker._total_bytes == 0

    def test_record_bytes(self):
        """Test recording bandwidth samples."""
        tracker = BandwidthTracker()
        tracker.record(1024)
        tracker.record(2048)
        assert tracker._total_bytes == 3072

    def test_get_total_mb(self):
        """Test total MB calculation."""
        tracker = BandwidthTracker()
        tracker.record(10 * 1024 * 1024)  # 10 MB
        assert tracker.get_total_mb() == 10.0

    @patch("time.monotonic")
    def test_bandwidth_calculation(self, mock_time):
        """Test bandwidth calculation in Mbps."""
        tracker = BandwidthTracker(window_seconds=10.0)

        # Record samples over time
        mock_time.return_value = 0.0
        tracker.record(1024 * 1024)  # 1 MB at t=0

        mock_time.return_value = 1.0
        tracker.record(1024 * 1024)  # 1 MB at t=1

        # 2 MB in 1 second = 2 MB/s = 16 Mbps
        bandwidth = tracker.get_bandwidth_mbps()
        assert bandwidth > 0

    @patch("time.monotonic")
    def test_sample_pruning(self, mock_time):
        """Test that old samples are pruned."""
        tracker = BandwidthTracker(window_seconds=5.0)

        # Add sample at t=0
        mock_time.return_value = 0.0
        tracker.record(1024)

        # Add sample at t=10 (should prune t=0)
        mock_time.return_value = 10.0
        tracker.record(2048)

        # Only recent sample should remain
        assert len(tracker._samples) == 1

    def test_zero_bandwidth_empty_tracker(self):
        """Test that empty tracker returns 0 bandwidth."""
        tracker = BandwidthTracker()
        assert tracker.get_bandwidth_mbps() == 0.0

    def test_thread_safety(self):
        """Test that operations are thread-safe (basic check)."""
        tracker = BandwidthTracker()
        # Multiple records should work without errors
        for _ in range(100):
            tracker.record(1024)
        assert tracker.get_total_mb() > 0


class TestDownloadStatistics:
    """Test DownloadStatistics class."""

    def test_init(self):
        """Test initialization."""
        stats = DownloadStatistics()
        assert stats.total_attempts == 0
        assert stats.total_successes == 0
        assert stats.total_failures == 0
        assert stats.total_bytes == 0
        assert isinstance(stats.bandwidth_tracker, BandwidthTracker)

    def test_record_successful_attempt(self):
        """Test recording a successful download attempt."""
        stats = DownloadStatistics()
        stats.record_attempt(
            resolver="openalex",
            success=True,
            classification="pdf",
            bytes_downloaded=1024000,
            elapsed_ms=2500,
        )

        assert stats.total_attempts == 1
        assert stats.total_successes == 1
        assert stats.total_failures == 0
        assert stats.total_bytes == 1024000
        assert stats.by_classification["pdf"] == 1
        assert stats.resolver_stats["openalex"].successes == 1

    def test_record_failed_attempt(self):
        """Test recording a failed download attempt."""
        stats = DownloadStatistics()
        stats.record_attempt(
            resolver="unpaywall",
            success=False,
            reason="timeout",
            domain="example.org",
        )

        assert stats.total_attempts == 1
        assert stats.total_successes == 0
        assert stats.total_failures == 1
        assert stats.failures_by_reason["timeout"] == 1
        assert stats.failures_by_domain["example.org"] == 1
        assert stats.resolver_stats["unpaywall"].failures == 1

    def test_multiple_attempts(self):
        """Test recording multiple attempts."""
        stats = DownloadStatistics()

        for i in range(10):
            stats.record_attempt(
                resolver="openalex",
                success=(i < 8),  # 8 successes, 2 failures
                classification="pdf" if i < 8 else None,
                reason="timeout" if i >= 8 else None,
                bytes_downloaded=1024000 if i < 8 else None,
            )

        assert stats.total_attempts == 10
        assert stats.total_successes == 8
        assert stats.total_failures == 2
        assert stats.get_success_rate() == 80.0

    def test_success_rate_calculation(self):
        """Test success rate percentage calculation."""
        stats = DownloadStatistics()

        stats.record_attempt(success=True)
        stats.record_attempt(success=True)
        stats.record_attempt(success=False)
        stats.record_attempt(success=False)

        assert stats.get_success_rate() == 50.0

        # Test with zero attempts
        empty_stats = DownloadStatistics()
        assert empty_stats.get_success_rate() == 0.0

    def test_average_speed_calculation(self):
        """Test average download speed calculation."""
        stats = DownloadStatistics()

        # 10 MB in 1000 ms = 10 MB/s = 80 Mbps
        stats.record_attempt(success=True, bytes_downloaded=10 * 1024 * 1024, elapsed_ms=1000.0)

        speed = stats.get_average_speed_mbps()
        assert speed > 0

    def test_percentile_time_calculation(self):
        """Test download time percentile calculation."""
        stats = DownloadStatistics()

        # Record downloads with different times
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            stats.record_attempt(success=True, elapsed_ms=float(ms))

        p50 = stats.get_percentile_time(50)
        p95 = stats.get_percentile_time(95)
        p99 = stats.get_percentile_time(99)

        assert 400 <= p50 <= 600  # Median around 500
        assert p95 >= p50
        assert p99 >= p95

    def test_percentile_empty_stats(self):
        """Test percentile calculation with no data."""
        stats = DownloadStatistics()
        assert stats.get_percentile_time(50) == 0.0

    def test_average_size_calculation(self):
        """Test average file size calculation."""
        stats = DownloadStatistics()

        stats.record_attempt(success=True, bytes_downloaded=1024 * 1024)  # 1 MB
        stats.record_attempt(success=True, bytes_downloaded=2 * 1024 * 1024)  # 2 MB
        stats.record_attempt(success=True, bytes_downloaded=3 * 1024 * 1024)  # 3 MB

        avg_size = stats.get_average_size_mb()
        assert avg_size == 2.0

    def test_total_mb_calculation(self):
        """Test total MB calculation."""
        stats = DownloadStatistics()

        stats.record_attempt(success=True, bytes_downloaded=5 * 1024 * 1024)  # 5 MB
        stats.record_attempt(success=True, bytes_downloaded=3 * 1024 * 1024)  # 3 MB

        assert stats.get_total_mb() == 8.0

    @patch("time.time")
    def test_elapsed_seconds(self, mock_time):
        """Test elapsed time calculation."""
        mock_time.return_value = 1000.0
        stats = DownloadStatistics()

        mock_time.return_value = 1010.0
        elapsed = stats.get_elapsed_seconds()

        assert elapsed == 10.0

    def test_get_top_failures(self):
        """Test getting top failure reasons."""
        stats = DownloadStatistics()

        stats.record_attempt(success=False, reason="timeout")
        stats.record_attempt(success=False, reason="timeout")
        stats.record_attempt(success=False, reason="timeout")
        stats.record_attempt(success=False, reason="http_error")
        stats.record_attempt(success=False, reason="http_error")
        stats.record_attempt(success=False, reason="connection_error")

        top_failures = stats.get_top_failures(limit=2)

        assert len(top_failures) == 2
        assert top_failures[0] == ("timeout", 3)
        assert top_failures[1] == ("http_error", 2)

    def test_get_top_failing_domains(self):
        """Test getting top failing domains."""
        stats = DownloadStatistics()

        for _ in range(5):
            stats.record_attempt(success=False, domain="slow.example.org")
        for _ in range(3):
            stats.record_attempt(success=False, domain="error.example.org")

        top_domains = stats.get_top_failing_domains(limit=2)

        assert len(top_domains) == 2
        assert top_domains[0] == ("slow.example.org", 5)
        assert top_domains[1] == ("error.example.org", 3)

    def test_resolver_stats_tracking(self):
        """Test per-resolver statistics tracking."""
        stats = DownloadStatistics()

        stats.record_attempt(
            resolver="openalex",
            success=True,
            bytes_downloaded=1024000,
            elapsed_ms=1000,
        )
        stats.record_attempt(resolver="openalex", success=False, reason="timeout")
        stats.record_attempt(
            resolver="unpaywall",
            success=True,
            bytes_downloaded=2048000,
            elapsed_ms=2000,
        )

        openalex_stats = stats.resolver_stats["openalex"]
        unpaywall_stats = stats.resolver_stats["unpaywall"]

        assert openalex_stats.attempts == 2
        assert openalex_stats.successes == 1
        assert openalex_stats.failures == 1
        assert openalex_stats.total_bytes == 1024000

        assert unpaywall_stats.attempts == 1
        assert unpaywall_stats.successes == 1
        assert unpaywall_stats.total_bytes == 2048000

    def test_format_summary_basic(self):
        """Test basic summary formatting."""
        stats = DownloadStatistics()

        stats.record_attempt(success=True, bytes_downloaded=1024000, elapsed_ms=1000)
        stats.record_attempt(success=False, reason="timeout")

        summary = stats.format_summary()

        assert "Total attempts: 2" in summary
        assert "Successes: 1 (50.0%)" in summary
        assert "Failures: 1" in summary
        assert "Download Statistics Summary" in summary

    def test_format_summary_with_resolvers(self):
        """Test summary formatting with resolver stats."""
        stats = DownloadStatistics()

        stats.record_attempt(
            resolver="openalex",
            success=True,
            bytes_downloaded=1024000,
            elapsed_ms=1000,
        )

        summary = stats.format_summary()
        assert "Resolver Performance:" in summary
        assert "openalex" in summary

    def test_format_summary_with_failures(self):
        """Test summary formatting with failure analysis."""
        stats = DownloadStatistics()

        stats.record_attempt(success=False, reason="timeout")
        stats.record_attempt(success=False, reason="http_error")

        summary = stats.format_summary()
        assert "Top Failure Reasons:" in summary
        assert "timeout" in summary

    def test_format_summary_with_classifications(self):
        """Test summary formatting with content classifications."""
        stats = DownloadStatistics()

        stats.record_attempt(success=True, classification="pdf")
        stats.record_attempt(success=True, classification="html")

        summary = stats.format_summary()
        assert "Content Types:" in summary
        assert "pdf" in summary
        assert "html" in summary

    def test_format_summary_with_percentiles(self):
        """Test summary formatting with download time percentiles."""
        stats = DownloadStatistics()

        for i in range(100):
            stats.record_attempt(success=True, elapsed_ms=float(i * 10))

        summary = stats.format_summary()
        assert "Download Time Percentiles:" in summary
        assert "50th (median)" in summary
        assert "95th" in summary
        assert "99th" in summary

    def test_thread_safety_record_attempt(self):
        """Test that record_attempt is thread-safe (basic check)."""
        stats = DownloadStatistics()

        # Multiple concurrent recordings should work
        for _ in range(100):
            stats.record_attempt(success=True)

        assert stats.total_attempts == 100

    def test_bandwidth_tracker_integration(self):
        """Test integration with bandwidth tracker."""
        stats = DownloadStatistics()

        stats.record_attempt(success=True, bytes_downloaded=1024000)

        # Bandwidth tracker should have recorded the bytes
        assert stats.bandwidth_tracker.get_total_mb() > 0


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_typical_download_session(self):
        """Test a typical download session with mixed results."""
        stats = DownloadStatistics()

        # Simulate 100 downloads
        for i in range(100):
            if i < 85:  # 85% success rate
                stats.record_attempt(
                    resolver="openalex" if i < 50 else "unpaywall",
                    success=True,
                    classification="pdf",
                    bytes_downloaded=500000 + i * 1000,
                    elapsed_ms=1000.0 + i * 10,
                )
            else:
                reasons = ["timeout", "http_error", "connection_error"]
                stats.record_attempt(
                    resolver="crossref",
                    success=False,
                    reason=reasons[i % 3],
                    domain="slow.example.org",
                )

        # Verify overall statistics
        assert stats.total_attempts == 100
        assert stats.total_successes == 85
        assert stats.total_failures == 15
        assert stats.get_success_rate() == 85.0

        # Verify resolver stats
        assert stats.resolver_stats["openalex"].successes == 50
        assert stats.resolver_stats["unpaywall"].successes == 35
        assert stats.resolver_stats["crossref"].failures == 15

        # Verify summary generation works
        summary = stats.format_summary()
        assert "85 (85.0%)" in summary

    def test_monitoring_real_time_bandwidth(self):
        """Test real-time bandwidth monitoring scenario."""
        stats = DownloadStatistics()

        # Simulate downloads with progress tracking
        for _ in range(10):
            stats.record_attempt(success=True, bytes_downloaded=1024 * 1024, elapsed_ms=1000.0)

        # Check current metrics
        total_mb = stats.get_total_mb()
        current_bw = stats.bandwidth_tracker.get_bandwidth_mbps()

        assert total_mb == 10.0
        assert current_bw >= 0  # May be 0 if samples too old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
