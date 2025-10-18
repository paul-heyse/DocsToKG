# 1. Module: statistics

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.statistics``.

## 1. Overview

Download statistics tracking and reporting for performance analysis.

This module provides real-time statistics collection for download operations,
including success rates, bandwidth usage, performance metrics, and failure analysis.

## 2. Functions

### `success_rate(self)`

Calculate success rate as percentage.

### `avg_time_ms(self)`

Calculate average download time in milliseconds.

### `total_mb(self)`

Calculate total megabytes downloaded.

### `record(self, bytes_downloaded)`

Record a bandwidth sample.

Args:
bytes_downloaded: Number of bytes in this sample

### `get_bandwidth_mbps(self)`

Get current bandwidth in megabits per second.

Returns:
Current bandwidth usage in Mbps

### `get_total_mb(self)`

Get total megabytes downloaded.

Returns:
Total MB downloaded since tracker creation

### `record_attempt(self, resolver, success, classification, reason, bytes_downloaded, elapsed_ms, domain)`

Record a download attempt with all relevant metrics.

Args:
resolver: Name of resolver that attempted download
success: Whether download succeeded
classification: Content classification (pdf, html, etc.)
reason: Failure reason code if applicable
bytes_downloaded: Number of bytes downloaded
elapsed_ms: Time taken in milliseconds
domain: Domain of the download URL

### `get_success_rate(self)`

Calculate overall success rate as percentage.

### `get_average_speed_mbps(self)`

Calculate average download speed in megabits per second.

### `get_percentile_time(self, percentile)`

Get download time at specified percentile.

Args:
percentile: Percentile value (0-100)

Returns:
Download time in milliseconds at the percentile

### `get_average_size_mb(self)`

Get average file size in megabytes.

### `get_total_mb(self)`

Get total megabytes downloaded.

### `get_elapsed_seconds(self)`

Get total elapsed time since tracker creation.

### `get_top_failures(self, limit)`

Get top failure reasons by count.

Args:
limit: Maximum number of reasons to return

Returns:
List of (reason, count) tuples sorted by count descending

### `get_top_failing_domains(self, limit)`

Get domains with most failures.

Args:
limit: Maximum number of domains to return

Returns:
List of (domain, count) tuples sorted by count descending

### `format_summary(self)`

Format comprehensive statistics summary.

Returns:
Human-readable statistics summary

### `percentile_value(p)`

Return the download time at percentile ``p`` from cached samples.

## 3. Classes

### `ResolverStats`

Statistics for a single resolver.

### `BandwidthTracker`

Track bandwidth usage over time windows.

### `DownloadStatistics`

Comprehensive statistics tracker for download operations.
