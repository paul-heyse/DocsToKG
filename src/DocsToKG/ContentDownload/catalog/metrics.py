# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.metrics",
#   "purpose": "OpenTelemetry metrics for catalog operations.",
#   "sections": [
#     {
#       "id": "catalogmetrics",
#       "name": "CatalogMetrics",
#       "anchor": "class-catalogmetrics",
#       "kind": "class"
#     },
#     {
#       "id": "get-catalog-metrics",
#       "name": "get_catalog_metrics",
#       "anchor": "function-get-catalog-metrics",
#       "kind": "function"
#     },
#     {
#       "id": "reset-metrics-for-tests",
#       "name": "reset_metrics_for_tests",
#       "anchor": "function-reset-metrics-for-tests",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""OpenTelemetry metrics for catalog operations.

Provides observability into catalog usage with 3 key metrics:
  - dedup_hits_total: Count of deduplication hits
  - gc_removed_total: Count of files removed by GC
  - verify_failures_total: Count of verification failures
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CatalogMetrics:
    """Container for catalog operation metrics."""

    def __init__(self):
        """Initialize catalog metrics.

        Lazy-loads OpenTelemetry on first use to avoid hard dependency.
        """
        self._dedup_hits_counter = None
        self._gc_removed_counter = None
        self._verify_failures_counter = None
        self._initialized = False

    def _init_meters(self) -> None:
        """Initialize OpenTelemetry meters."""
        if self._initialized:
            return

        try:
            from opentelemetry import metrics  # type: ignore[import-untyped]

            meter = metrics.get_meter(__name__)

            # Deduplication hits counter
            self._dedup_hits_counter = meter.create_counter(
                name="contentdownload.dedup_hits_total",
                description="Total count of deduplication hits (same content found)",
                unit="1",
            )

            # GC removed files counter
            self._gc_removed_counter = meter.create_counter(
                name="contentdownload.gc_removed_total",
                description="Total count of files removed by garbage collection",
                unit="1",
            )

            # Verification failures counter
            self._verify_failures_counter = meter.create_counter(
                name="contentdownload.verify_failures_total",
                description="Total count of verification failures (hash mismatches)",
                unit="1",
            )

            self._initialized = True
            logger.debug("OpenTelemetry meters initialized for catalog metrics")

        except ImportError:
            logger.warning("OpenTelemetry not available; metrics will be no-ops")
            self._initialized = True

    def record_dedup_hit(self, count: int = 1, **attributes) -> None:
        """Record a deduplication hit.

        Args:
            count: Number of hits (default 1)
            **attributes: Optional OTel attributes (e.g., resolver, layout)
        """
        self._init_meters()

        if self._dedup_hits_counter:
            try:
                self._dedup_hits_counter.add(count, attributes=attributes or {})
            except Exception as e:
                logger.warning(f"Failed to record dedup hit metric: {e}")

    def record_gc_removed(self, count: int = 1, **attributes) -> None:
        """Record files removed by garbage collection.

        Args:
            count: Number of files removed (default 1)
            **attributes: Optional OTel attributes (e.g., reason, retention_days)
        """
        self._init_meters()

        if self._gc_removed_counter:
            try:
                self._gc_removed_counter.add(count, attributes=attributes or {})
            except Exception as e:
                logger.warning(f"Failed to record GC removed metric: {e}")

    def record_verify_failure(self, count: int = 1, **attributes) -> None:
        """Record verification failures.

        Args:
            count: Number of failures (default 1)
            **attributes: Optional OTel attributes (e.g., record_id, reason)
        """
        self._init_meters()

        if self._verify_failures_counter:
            try:
                self._verify_failures_counter.add(count, attributes=attributes or {})
            except Exception as e:
                logger.warning(f"Failed to record verify failure metric: {e}")


# Global metrics instance
_metrics_instance: CatalogMetrics | None = None


def get_catalog_metrics() -> CatalogMetrics:
    """Get or create the global catalog metrics instance.

    Returns:
        CatalogMetrics singleton instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = CatalogMetrics()
    return _metrics_instance


def reset_metrics_for_tests() -> None:
    """Reset the global metrics instance (for testing).

    Call this in test teardown to isolate metric state.
    """
    global _metrics_instance
    _metrics_instance = None
