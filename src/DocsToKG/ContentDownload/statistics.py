"""Download statistics tracking and reporting for performance analysis.

This module provides real-time statistics collection for download operations,
including success rates, bandwidth usage, performance metrics, and failure analysis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = (
    "DownloadStatistics",
    "ResolverStats",
    "BandwidthTracker",
)


@dataclass
class ResolverStats:
    """Statistics for a single resolver."""

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_bytes: int = 0
    total_time_ms: float = 0.0
    failures_by_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.attempts == 0:
            return 0.0
        return (self.successes / self.attempts) * 100.0

    @property
    def avg_time_ms(self) -> float:
        """Calculate average download time in milliseconds."""
        if self.successes == 0:
            return 0.0
        return self.total_time_ms / self.successes

    @property
    def total_mb(self) -> float:
        """Calculate total megabytes downloaded."""
        return self.total_bytes / (1024 * 1024)


class BandwidthTracker:
    """Track bandwidth usage over time windows."""

    def __init__(self, window_seconds: float = 60.0):
        """Initialize bandwidth tracker.

        Args:
            window_seconds: Time window for bandwidth calculation
        """
        self._window_seconds = window_seconds
        self._samples: List[tuple[float, int]] = []  # (timestamp, bytes)
        self._lock = threading.Lock()
        self._total_bytes = 0

    def record(self, bytes_downloaded: int) -> None:
        """Record a bandwidth sample.

        Args:
            bytes_downloaded: Number of bytes in this sample
        """
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, bytes_downloaded))
            self._total_bytes += bytes_downloaded
            # Prune old samples outside the window
            cutoff = now - self._window_seconds
            self._samples = [(ts, b) for ts, b in self._samples if ts > cutoff]

    def get_bandwidth_mbps(self) -> float:
        """Get current bandwidth in megabits per second.

        Returns:
            Current bandwidth usage in Mbps
        """
        now = time.monotonic()
        cutoff = now - self._window_seconds

        with self._lock:
            recent_samples = [(ts, b) for ts, b in self._samples if ts > cutoff]
            if not recent_samples:
                return 0.0

            total_bytes = sum(b for _, b in recent_samples)
            oldest_ts = recent_samples[0][0] if recent_samples else now
            elapsed = now - oldest_ts

            if elapsed <= 0:
                return 0.0

            bytes_per_second = total_bytes / elapsed
            bits_per_second = bytes_per_second * 8
            return bits_per_second / (1024 * 1024)  # Convert to Mbps

    def get_total_mb(self) -> float:
        """Get total megabytes downloaded.

        Returns:
            Total MB downloaded since tracker creation
        """
        with self._lock:
            return self._total_bytes / (1024 * 1024)


class DownloadStatistics:
    """Comprehensive statistics tracker for download operations."""

    def __init__(self):
        """Initialize statistics tracker."""
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Overall statistics
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_bytes = 0
        self.total_time_ms = 0.0

        # Per-resolver statistics
        self.resolver_stats: Dict[str, ResolverStats] = defaultdict(ResolverStats)

        # Classification statistics
        self.by_classification: Dict[str, int] = defaultdict(int)

        # Failure analysis
        self.failures_by_reason: Dict[str, int] = defaultdict(int)
        self.failures_by_domain: Dict[str, int] = defaultdict(int)

        # Performance metrics
        self.bandwidth_tracker = BandwidthTracker()
        self.download_times: List[float] = []  # For percentile calculation

        # Size statistics
        self.sizes_mb: List[float] = []

    def record_attempt(
        self,
        resolver: Optional[str] = None,
        success: bool = False,
        classification: Optional[str] = None,
        reason: Optional[str] = None,
        bytes_downloaded: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Record a download attempt with all relevant metrics.

        Args:
            resolver: Name of resolver that attempted download
            success: Whether download succeeded
            classification: Content classification (pdf, html, etc.)
            reason: Failure reason code if applicable
            bytes_downloaded: Number of bytes downloaded
            elapsed_ms: Time taken in milliseconds
            domain: Domain of the download URL
        """
        with self._lock:
            self.total_attempts += 1

            if success:
                self.total_successes += 1
            else:
                self.total_failures += 1
                if reason:
                    self.failures_by_reason[reason] += 1
                if domain:
                    self.failures_by_domain[domain] += 1

            if classification:
                self.by_classification[classification] += 1

            if bytes_downloaded and bytes_downloaded > 0:
                self.total_bytes += bytes_downloaded
                self.bandwidth_tracker.record(bytes_downloaded)
                size_mb = bytes_downloaded / (1024 * 1024)
                self.sizes_mb.append(size_mb)

            if elapsed_ms and elapsed_ms > 0:
                self.total_time_ms += elapsed_ms
                if success:
                    self.download_times.append(elapsed_ms)

            # Update resolver-specific stats
            if resolver:
                stats = self.resolver_stats[resolver]
                stats.attempts += 1
                if success:
                    stats.successes += 1
                    if elapsed_ms:
                        stats.total_time_ms += elapsed_ms
                else:
                    stats.failures += 1
                    if reason:
                        stats.failures_by_reason[reason] += 1

                if bytes_downloaded:
                    stats.total_bytes += bytes_downloaded

    def get_success_rate(self) -> float:
        """Calculate overall success rate as percentage."""
        with self._lock:
            total_attempts = self.total_attempts
            total_successes = self.total_successes
        if total_attempts == 0:
            return 0.0
        return (total_successes / total_attempts) * 100.0

    def get_average_speed_mbps(self) -> float:
        """Calculate average download speed in megabits per second."""
        with self._lock:
            total_time_ms = self.total_time_ms
            total_bytes = self.total_bytes
        if total_time_ms == 0:
            return 0.0

        seconds = total_time_ms / 1000.0
        bytes_per_second = total_bytes / seconds
        bits_per_second = bytes_per_second * 8
        return bits_per_second / (1024 * 1024)

    def get_percentile_time(self, percentile: float) -> float:
        """Get download time at specified percentile.

        Args:
            percentile: Percentile value (0-100)

        Returns:
            Download time in milliseconds at the percentile
        """
        with self._lock:
            times_snapshot = tuple(self.download_times)
        if not times_snapshot:
            return 0.0

        sorted_times = sorted(times_snapshot)
        index = int((percentile / 100.0) * len(sorted_times))
        index = min(index, len(sorted_times) - 1)
        return sorted_times[index]

    def get_average_size_mb(self) -> float:
        """Get average file size in megabytes."""
        with self._lock:
            sizes_snapshot = tuple(self.sizes_mb)
        if not sizes_snapshot:
            return 0.0
        return sum(sizes_snapshot) / len(sizes_snapshot)

    def get_total_mb(self) -> float:
        """Get total megabytes downloaded."""
        with self._lock:
            total_bytes = self.total_bytes
        return total_bytes / (1024 * 1024)

    def get_elapsed_seconds(self) -> float:
        """Get total elapsed time since tracker creation."""
        return time.time() - self._start_time

    def get_top_failures(self, limit: int = 5) -> List[tuple[str, int]]:
        """Get top failure reasons by count.

        Args:
            limit: Maximum number of reasons to return

        Returns:
            List of (reason, count) tuples sorted by count descending
        """
        with self._lock:
            sorted_failures = sorted(
                self.failures_by_reason.items(), key=lambda x: x[1], reverse=True
            )
        return sorted_failures[:limit]

    def get_top_failing_domains(self, limit: int = 5) -> List[tuple[str, int]]:
        """Get domains with most failures.

        Args:
            limit: Maximum number of domains to return

        Returns:
            List of (domain, count) tuples sorted by count descending
        """
        with self._lock:
            sorted_domains = sorted(
                self.failures_by_domain.items(), key=lambda x: x[1], reverse=True
            )
        return sorted_domains[:limit]

    def format_summary(self) -> str:
        """Format comprehensive statistics summary.

        Returns:
            Human-readable statistics summary
        """
        with self._lock:
            total_attempts = self.total_attempts
            total_successes = self.total_successes
            total_failures = self.total_failures
            total_bytes = self.total_bytes
            total_time_ms = self.total_time_ms
            sizes_snapshot = tuple(self.sizes_mb)
            download_times_snapshot = tuple(self.download_times)
            by_classification_snapshot = dict(self.by_classification)
            failures_by_reason_snapshot = dict(self.failures_by_reason)
            resolver_stats_snapshot = {
                name: ResolverStats(
                    attempts=stats.attempts,
                    successes=stats.successes,
                    failures=stats.failures,
                    total_bytes=stats.total_bytes,
                    total_time_ms=stats.total_time_ms,
                    failures_by_reason=dict(stats.failures_by_reason),
                )
                for name, stats in self.resolver_stats.items()
            }

        elapsed = self.get_elapsed_seconds()
        success_rate = (
            (total_successes / total_attempts) * 100.0 if total_attempts else 0.0
        )
        avg_speed = 0.0
        if total_time_ms:
            seconds = total_time_ms / 1000.0
            if seconds > 0:
                bytes_per_second = total_bytes / seconds
                bits_per_second = bytes_per_second * 8
                avg_speed = bits_per_second / (1024 * 1024)
        total_mb = total_bytes / (1024 * 1024)
        current_bandwidth = self.bandwidth_tracker.get_bandwidth_mbps()

        lines = [
            "=" * 70,
            "Download Statistics Summary",
            "=" * 70,
            "",
            "Overall Performance:",
            f"  Total attempts: {total_attempts}",
            f"  Successes: {total_successes} ({success_rate:.1f}%)",
            f"  Failures: {total_failures}",
            f"  Elapsed time: {elapsed:.1f}s",
            "",
            "Data Transfer:",
            f"  Total downloaded: {total_mb:.2f} MB",
            f"  Average speed: {avg_speed:.2f} Mbps",
            f"  Current bandwidth: {current_bandwidth:.2f} Mbps",
        ]

        if sizes_snapshot:
            avg_size = sum(sizes_snapshot) / len(sizes_snapshot)
            lines.extend(
                [
                    f"  Average file size: {avg_size:.2f} MB",
                ]
            )

        if download_times_snapshot:
            sorted_times = sorted(download_times_snapshot)
            last_index = len(sorted_times) - 1

            def percentile_value(p: float) -> float:
                index = int((p / 100.0) * len(sorted_times))
                index = min(max(index, 0), last_index)
                return sorted_times[index]

            p50 = percentile_value(50)
            p95 = percentile_value(95)
            p99 = percentile_value(99)
            lines.extend(
                [
                    "",
                    "Download Time Percentiles:",
                    f"  50th (median): {p50:.0f} ms",
                    f"  95th: {p95:.0f} ms",
                    f"  99th: {p99:.0f} ms",
                ]
            )

        if by_classification_snapshot:
            lines.extend(["", "Content Types:"])
            for classification, count in sorted(
                by_classification_snapshot.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (count / total_attempts * 100) if total_attempts > 0 else 0
                lines.append(f"  {classification}: {count} ({pct:.1f}%)")

        if failures_by_reason_snapshot:
            lines.extend(["", "Top Failure Reasons:"])
            sorted_failures = sorted(
                failures_by_reason_snapshot.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for reason, count in sorted_failures:
                pct = (count / total_failures * 100) if total_failures > 0 else 0
                lines.append(f"  {reason}: {count} ({pct:.1f}% of failures)")

        if resolver_stats_snapshot:
            lines.extend(["", "Resolver Performance:"])
            for name, stats in sorted(
                resolver_stats_snapshot.items(),
                key=lambda x: x[1].successes,
                reverse=True,
            ):
                lines.append(
                    f"  {name}: {stats.successes}/{stats.attempts} "
                    f"({stats.success_rate:.1f}%), "
                    f"{stats.total_mb:.1f} MB"
                )

        lines.extend(["", "=" * 70])
        return "\n".join(lines)
