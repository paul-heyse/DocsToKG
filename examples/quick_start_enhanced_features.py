#!/usr/bin/env python3
"""
Quick Start: Enhanced Content Download Features

A minimal example demonstrating the key enhancements:
- Progress tracking
- Error handling
- Statistics collection
"""

import logging
from pathlib import Path
from typing import Optional

from DocsToKG.ContentDownload.statistics import DownloadStatistics
from DocsToKG.ContentDownload.errors import log_download_failure

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def progress_callback(bytes_downloaded: int, total_bytes: Optional[int], url: str):
    """Simple progress callback that logs every 25% completion."""
    if total_bytes:
        percent = int((bytes_downloaded / total_bytes) * 100)
        if percent % 25 == 0:
            mb_dl = bytes_downloaded / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            LOGGER.info(f"Progress: {percent}% ({mb_dl:.1f}/{mb_total:.1f} MB)")


def main():
    """Demonstrate enhanced features with a simple workflow."""

    # Initialize statistics tracker
    stats = DownloadStatistics()

    LOGGER.info("Starting enhanced download example...")

    # Simulate some download attempts
    test_cases = [
        ("example1.pdf", True, 1024 * 1024, 1500, "openalex"),
        ("example2.pdf", True, 2048 * 1024, 2300, "unpaywall"),
        ("example3.pdf", False, None, 800, "crossref"),
    ]

    for filename, success, bytes_dl, elapsed_ms, resolver in test_cases:
        url = f"https://example.org/{filename}"

        # Record attempt in statistics
        stats.record_attempt(
            resolver=resolver,
            success=success,
            classification="pdf" if success else None,
            bytes_downloaded=bytes_dl,
            elapsed_ms=elapsed_ms,
            reason="timeout" if not success else None,
            domain="example.org" if not success else None,
        )

        if success:
            LOGGER.info(f"✓ Downloaded {filename}: {bytes_dl / 1024:.1f} KB in {elapsed_ms:.0f} ms")
        else:
            # Demonstrate enhanced error logging
            log_download_failure(
                LOGGER,
                url,
                work_id="W12345",
                http_status=None,
                reason_code="timeout",
                error_details="Connection timed out after 30 seconds",
            )

    # Display statistics
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Download Session Summary")
    LOGGER.info("=" * 60)

    print(stats.format_summary())

    LOGGER.info(f"\nQuick Stats:")
    LOGGER.info(f"  Success Rate: {stats.get_success_rate():.1f}%")
    LOGGER.info(f"  Total Downloaded: {stats.get_total_mb():.2f} MB")
    LOGGER.info(f"  Average Speed: {stats.get_average_speed_mbps():.2f} Mbps")

    # Demonstrate progress callback usage
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Progress Callback Demo")
    LOGGER.info("=" * 60)

    # Simulate progress updates
    total_bytes = 10 * 1024 * 1024  # 10 MB
    for percent in [0, 25, 50, 75, 100]:
        bytes_downloaded = int((percent / 100) * total_bytes)
        progress_callback(bytes_downloaded, total_bytes, "https://example.org/demo.pdf")


if __name__ == "__main__":
    main()
