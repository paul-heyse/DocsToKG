#!/usr/bin/env python3
"""
Example script demonstrating the enhanced content download features.

This script showcases:
- Progress tracking callbacks
- Size warnings and download limits
- Improved error handling with actionable messages
- Download statistics tracking
- HTTP Range resume support (infrastructure in place)
- Enhanced retry strategies
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from DocsToKG.ContentDownload.cli import WorkArtifact, download_candidate
from DocsToKG.ContentDownload.core import ensure_dir
from DocsToKG.ContentDownload.errors import format_download_summary, log_download_failure
from DocsToKG.ContentDownload.networking import create_session
from DocsToKG.ContentDownload.statistics import DownloadStatistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


class ProgressTracker:
    """Simple progress tracker for demonstrating progress callbacks."""

    def __init__(self, url: str):
        self.url = url
        self.last_percent = -1

    def __call__(self, bytes_downloaded: int, total_bytes: Optional[int], url: str) -> None:
        """Progress callback invoked during downloads."""
        if total_bytes:
            percent = int((bytes_downloaded / total_bytes) * 100)
            if percent != self.last_percent and percent % 10 == 0:
                mb_downloaded = bytes_downloaded / (1024 * 1024)
                mb_total = total_bytes / (1024 * 1024)
                LOGGER.info(
                    f"Download progress: {percent}% ({mb_downloaded:.2f}/{mb_total:.2f} MB) - {url}"
                )
                self.last_percent = percent
        else:
            # Unknown total size, just report bytes
            mb_downloaded = bytes_downloaded / (1024 * 1024)
            if bytes_downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                LOGGER.info(f"Downloaded {mb_downloaded:.2f} MB - {url}")


def example_basic_download():
    """Example 1: Basic download with progress tracking."""
    LOGGER.info("=" * 60)
    LOGGER.info("Example 1: Basic download with progress tracking")
    LOGGER.info("=" * 60)

    # Create work artifact
    work_id = "W2741809807"  # Example OpenAlex work ID
    base_dir = Path("./example_downloads")
    ensure_dir(base_dir / "pdfs")
    ensure_dir(base_dir / "html")
    ensure_dir(base_dir / "xml")

    artifact = WorkArtifact(
        work_id=work_id,
        base_stem=work_id,
        pdf_dir=base_dir / "pdfs",
        html_dir=base_dir / "html",
        xml_dir=base_dir / "xml",
    )

    # Create session
    session = create_session(
        headers={"User-Agent": "DocsToKG/1.0 (mailto:example@example.org)"},
        enable_compression=True,  # Enable gzip compression
    )

    # Example URL (replace with actual PDF URL)
    url = "https://arxiv.org/pdf/2301.00000.pdf"

    # Setup progress callback
    progress_tracker = ProgressTracker(url)

    # Configure download context with new features
    context: Dict[str, Any] = {
        "chunk_size": 32 * 1024,  # 32KB chunks
        "progress_callback": progress_tracker,
        "size_warning_threshold": 50 * 1024 * 1024,  # Warn if file > 50MB
        "skip_large_downloads": False,  # Don't skip, just warn
    }

    try:
        outcome = download_candidate(
            session=session,
            artifact=artifact,
            url=url,
            referer=None,
            timeout=30.0,
            context=context,
        )

        LOGGER.info(f"Download completed: {outcome.classification}")
        if outcome.path:
            LOGGER.info(f"File saved to: {outcome.path}")
            LOGGER.info(f"SHA256: {outcome.sha256}")
            LOGGER.info(f"Size: {outcome.content_length / (1024 * 1024):.2f} MB")
            LOGGER.info(f"Time: {outcome.elapsed_ms:.0f} ms")

    except Exception as exc:
        LOGGER.error(f"Download failed: {exc}", exc_info=True)
    finally:
        session.close()


def example_statistics_tracking():
    """Example 2: Download statistics tracking across multiple files."""
    LOGGER.info("=" * 60)
    LOGGER.info("Example 2: Statistics tracking for batch downloads")
    LOGGER.info("=" * 60)

    # Initialize statistics tracker
    stats = DownloadStatistics()

    # Create session
    session = create_session(headers={"User-Agent": "DocsToKG/1.0 (mailto:example@example.org)"})

    # Setup base directory
    base_dir = Path("./example_downloads_batch")
    ensure_dir(base_dir / "pdfs")
    ensure_dir(base_dir / "html")
    ensure_dir(base_dir / "xml")

    # Example URLs to download (mix of PDF and HTML)
    test_urls = [
        ("https://arxiv.org/pdf/2301.00000.pdf", "openalex"),
        ("https://www.example.org/paper.pdf", "crossref"),
        ("https://www.example.org/article.html", "unpaywall"),
    ]

    try:
        for idx, (url, resolver) in enumerate(test_urls):
            work_id = f"W{1000000 + idx}"
            artifact = WorkArtifact(
                work_id=work_id,
                base_stem=work_id,
                pdf_dir=base_dir / "pdfs",
                html_dir=base_dir / "html",
                xml_dir=base_dir / "xml",
            )

            LOGGER.info(f"\nDownloading {idx + 1}/{len(test_urls)}: {url}")

            try:
                outcome = download_candidate(
                    session=session,
                    artifact=artifact,
                    url=url,
                    referer=None,
                    timeout=30.0,
                )

                # Record statistics
                success = outcome.classification.value in ["pdf", "html", "xml"]
                stats.record_attempt(
                    resolver=resolver,
                    success=success,
                    classification=outcome.classification.value if success else None,
                    bytes_downloaded=outcome.content_length,
                    elapsed_ms=outcome.elapsed_ms,
                    reason=outcome.reason if not success else None,
                    domain=url.split("/")[2] if not success else None,
                )

                if success:
                    LOGGER.info(
                        f"✓ Success: {outcome.classification} ({outcome.content_length / 1024:.1f} KB)"
                    )
                else:
                    LOGGER.warning(f"✗ Failed: {outcome.reason} - {outcome.reason_detail}")

            except Exception as exc:
                # Record failure
                stats.record_attempt(resolver=resolver, success=False, reason="exception")
                LOGGER.error(f"✗ Exception: {exc}")

        # Print comprehensive statistics
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Download Statistics Summary")
        LOGGER.info("=" * 60)
        print(stats.format_summary())

        # Additional statistics
        LOGGER.info("\n" + "-" * 60)
        LOGGER.info(f"Success Rate: {stats.get_success_rate():.1f}%")
        LOGGER.info(f"Total Downloaded: {stats.get_total_mb():.2f} MB")
        LOGGER.info(f"Average Speed: {stats.get_average_speed_mbps():.2f} Mbps")
        LOGGER.info(f"Average File Size: {stats.get_average_size_mb():.2f} MB")

        # Download time percentiles
        p50 = stats.get_percentile_time(50)
        p95 = stats.get_percentile_time(95)
        p99 = stats.get_percentile_time(99)
        LOGGER.info("\nDownload Time Percentiles:")
        LOGGER.info(f"  50th (median): {p50:.0f} ms")
        LOGGER.info(f"  95th: {p95:.0f} ms")
        LOGGER.info(f"  99th: {p99:.0f} ms")

        # Top failures
        top_failures = stats.get_top_failures(limit=3)
        if top_failures:
            LOGGER.info("\nTop Failure Reasons:")
            for reason, count in top_failures:
                LOGGER.info(f"  {reason}: {count} occurrences")

    finally:
        session.close()


def example_error_handling():
    """Example 3: Enhanced error handling with actionable messages."""
    LOGGER.info("=" * 60)
    LOGGER.info("Example 3: Enhanced error handling")
    LOGGER.info("=" * 60)

    # Create session
    session = create_session()

    # Setup base directory
    base_dir = Path("./example_downloads_errors")
    ensure_dir(base_dir / "pdfs")
    ensure_dir(base_dir / "html")
    ensure_dir(base_dir / "xml")

    # Test various error scenarios
    error_scenarios = [
        ("https://httpbin.org/status/404", "not_found"),
        ("https://httpbin.org/status/403", "forbidden"),
        ("https://httpbin.org/status/429", "rate_limited"),
        ("https://httpbin.org/status/500", "server_error"),
        ("https://invalid-domain-that-does-not-exist-12345.com/test.pdf", "dns_error"),
    ]

    try:
        for idx, (url, scenario) in enumerate(error_scenarios):
            work_id = f"W{2000000 + idx}"
            artifact = WorkArtifact(
                work_id=work_id,
                base_stem=work_id,
                pdf_dir=base_dir / "pdfs",
                html_dir=base_dir / "html",
                xml_dir=base_dir / "xml",
            )

            LOGGER.info(f"\nTesting {scenario}: {url}")

            try:
                outcome = download_candidate(
                    session=session,
                    artifact=artifact,
                    url=url,
                    referer=None,
                    timeout=10.0,
                )

                if outcome.http_status and outcome.http_status >= 400:
                    # Demonstrate actionable error logging
                    log_download_failure(
                        LOGGER,
                        url,
                        work_id,
                        http_status=outcome.http_status,
                        reason_code=outcome.reason or "http_error",
                        error_details=outcome.reason_detail,
                    )

            except Exception as exc:
                log_download_failure(
                    LOGGER,
                    url,
                    work_id,
                    reason_code="request_exception",
                    error_details=str(exc),
                    exception=exc,
                )

    finally:
        session.close()


def example_size_limits_and_warnings():
    """Example 4: Size warnings and download limits."""
    LOGGER.info("=" * 60)
    LOGGER.info("Example 4: Size limits and warnings")
    LOGGER.info("=" * 60)

    session = create_session()
    base_dir = Path("./example_downloads_limits")
    ensure_dir(base_dir / "pdfs")
    ensure_dir(base_dir / "html")
    ensure_dir(base_dir / "xml")

    artifact = WorkArtifact(
        work_id="W3000000",
        base_stem="W3000000",
        pdf_dir=base_dir / "pdfs",
        html_dir=base_dir / "html",
        xml_dir=base_dir / "xml",
    )

    # URL that returns a large file (adjust as needed)
    url = "https://example.org/large-paper.pdf"

    # Configure strict size limits
    context: Dict[str, Any] = {
        "size_warning_threshold": 10 * 1024 * 1024,  # Warn at 10MB
        "skip_large_downloads": True,  # Skip downloads > threshold
    }

    try:
        outcome = download_candidate(
            session=session,
            artifact=artifact,
            url=url,
            referer=None,
            timeout=30.0,
            context=context,
        )

        LOGGER.info(f"Download result: {outcome.classification}")
        if outcome.reason:
            LOGGER.info(f"Reason: {outcome.reason} - {outcome.reason_detail}")

    except Exception as exc:
        LOGGER.error(f"Download failed: {exc}", exc_info=True)
    finally:
        session.close()


def example_formatted_summary():
    """Example 5: Formatted download summary for reporting."""
    LOGGER.info("=" * 60)
    LOGGER.info("Example 5: Formatted download summary")
    LOGGER.info("=" * 60)

    # Create sample summary from theoretical batch download
    summary = format_download_summary(
        total_attempts=100,
        successes=85,
        failures_by_reason={
            "timeout": 8,
            "http_error": 4,
            "connection_error": 2,
            "robots_disallowed": 1,
        },
    )

    print("\n" + summary)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DocsToKG Enhanced Content Download Features - Examples")
    print("=" * 60 + "\n")

    examples = [
        ("Basic Download with Progress", example_basic_download),
        ("Statistics Tracking", example_statistics_tracking),
        ("Error Handling", example_error_handling),
        ("Size Limits", example_size_limits_and_warnings),
        ("Formatted Summary", example_formatted_summary),
    ]

    if len(sys.argv) > 1:
        # Run specific example by number
        try:
            example_num = int(sys.argv[1]) - 1
            if 0 <= example_num < len(examples):
                name, func = examples[example_num]
                LOGGER.info(f"Running example {example_num + 1}: {name}")
                func()
            else:
                print(f"Invalid example number. Choose 1-{len(examples)}")
        except ValueError:
            print("Usage: python content_download_with_enhancements.py [example_number]")
    else:
        # Run all examples
        for idx, (name, func) in enumerate(examples, 1):
            try:
                func()
                print("\n")
            except Exception as exc:
                LOGGER.error(f"Example {idx} ({name}) failed: {exc}", exc_info=True)
                print("\n")


if __name__ == "__main__":
    main()
