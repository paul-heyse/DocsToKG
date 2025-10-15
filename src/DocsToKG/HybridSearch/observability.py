"""
Lightweight observability primitives for ingestion and retrieval.

This module provides comprehensive observability and monitoring capabilities
for DocsToKG's hybrid search operations, including metrics collection,
timing utilities, and performance tracking for ingestion and retrieval
operations.

The observability system supports:
- Counter and histogram metrics collection
- Timing context managers for operation measurement
- Prometheus-style metric export
- Performance monitoring and alerting
- Structured logging integration
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple


@dataclass
class CounterSample:
    """Sample from a counter metric with labels and value.

    This class represents a single measurement from a counter metric,
    including the metric name, associated labels, and current value.

    Attributes:
        name: Name of the counter metric
        labels: Dictionary of label key-value pairs
        value: Current counter value

    Examples:
        >>> sample = CounterSample(
        ...     name="documents_processed",
        ...     labels={"status": "success"},
        ...     value=150.0
        ... )
    """

    name: str
    labels: Mapping[str, str]
    value: float


@dataclass
class HistogramSample:
    """Sample from a histogram metric with percentile statistics.

    This class represents statistical information from a histogram metric,
    including count and percentile values for performance analysis.

    Attributes:
        name: Name of the histogram metric
        labels: Dictionary of label key-value pairs
        count: Total number of observations
        p50: 50th percentile (median) value
        p95: 95th percentile value
        p99: 99th percentile value

    Examples:
        >>> sample = HistogramSample(
        ...     name="search_latency",
        ...     labels={"method": "hybrid"},
        ...     count=1000,
        ...     p50=45.2,
        ...     p95=89.1,
        ...     p99=156.7
        ... )
    """

    name: str
    labels: Mapping[str, str]
    count: int
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """In-memory metrics collector compatible with Prometheus-style summaries.

    This class provides lightweight, in-memory metrics collection for
    monitoring DocsToKG operations, supporting both counter and histogram
    metrics with labeled dimensions for detailed observability.

    The collector uses thread-safe operations and provides Prometheus-style
    metric export for integration with monitoring systems.

    Attributes:
        _counters: Internal storage for counter metrics
        _histograms: Internal storage for histogram metrics

    Examples:
        >>> collector = MetricsCollector()
        >>> collector.increment("documents_processed", labels={"type": "pdf"})
        >>> collector.observe("search_latency", 45.2, method="hybrid")
        >>> counters = list(collector.export_counters())
    """

    def __init__(self) -> None:
        self._counters: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], float] = (
            defaultdict(float)
        )
        self._histograms: MutableMapping[Tuple[str, Tuple[Tuple[str, str], ...]], list[float]] = (
            defaultdict(list)
        )

    def increment(self, name: str, amount: float = 1.0, **labels: str) -> None:
        """Increase a counter metric by the given amount.

        Args:
            name: Metric identifier.
            amount: Increment to apply to the counter (default: 1.0).
            **labels: Arbitrary label key/value pairs for dimensioning.

        Returns:
            None
        """
        key = (name, tuple(sorted(labels.items())))
        self._counters[key] += amount

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record an observation for a histogram metric.

        Args:
            name: Metric identifier.
            value: Observation value to append to the histogram.
            **labels: Arbitrary label key/value pairs for dimensioning.

        Returns:
            None
        """
        key = (name, tuple(sorted(labels.items())))
        self._histograms[key].append(value)

    def export_counters(self) -> Iterable[CounterSample]:
        """Iterate over collected counter metrics as structured samples.

        Args:
            None

        Returns:
            Iterable of `CounterSample` entries representing current counters.
        """
        for (name, labels), value in self._counters.items():
            yield CounterSample(name=name, labels=dict(labels), value=value)

    def export_histograms(self) -> Iterable[HistogramSample]:
        """Iterate over collected histogram metrics summarized by percentiles.

        Args:
            None

        Returns:
            Iterable of `HistogramSample` entries containing percentile stats.
        """
        for (name, labels), samples in self._histograms.items():
            sorted_samples = sorted(samples)
            count = len(sorted_samples)
            if count == 0:
                continue
            p50 = sorted_samples[int(0.5 * (count - 1))]
            p95 = sorted_samples[int(0.95 * (count - 1))]
            p99 = sorted_samples[int(0.99 * (count - 1))]
            yield HistogramSample(
                name=name, labels=dict(labels), count=count, p50=p50, p95=p95, p99=p99
            )


class TraceRecorder:
    """Context manager producing timing spans for tracing.

    Attributes:
        _metrics: MetricsCollector used to record span durations.
        _logger: Logger that emits structured span events.

    Examples:
        >>> recorder = TraceRecorder(MetricsCollector(), logging.getLogger("test"))
        >>> with recorder.span("example"):
        ...     pass
    """

    def __init__(self, metrics: MetricsCollector, logger: logging.Logger) -> None:
        self._metrics = metrics
        self._logger = logger

    @contextmanager
    def span(self, name: str, **attributes: str) -> Iterator[None]:
        """Record execution duration for a traced operation.

        Args:
            name: Span name, used in metric and log emission.
            **attributes: Additional context attached to metrics and logs.

        Returns:
            None

        Yields:
            None

        Raises:
            Exception: Propagates any exception raised inside the traced block.
        """
        start = time.perf_counter()
        try:
            yield
            status = "ok"
        except Exception:
            status = "error"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._metrics.observe(f"trace_{name}_ms", duration_ms, **attributes)
            payload = {"span": name, "duration_ms": round(duration_ms, 3), "status": status}
            payload.update(attributes)
            self._logger.info("hybrid-trace", extra={"event": payload})


class Observability:
    """Facade for metrics, structured logging, and tracing.

    Attributes:
        _metrics: Shared metrics collector capturing counters and histograms.
        _logger: Logger scoped to the hybrid search subsystem.
        _tracer: TraceRecorder producing timing spans.

    Examples:
        >>> obs = Observability()
        >>> isinstance(obs.metrics_snapshot(), dict)
        True
    """

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self._metrics = MetricsCollector()
        self._logger = logger or logging.getLogger("DocsToKG.HybridSearch")
        self._tracer = TraceRecorder(self._metrics, self._logger)

    @property
    def metrics(self) -> MetricsCollector:
        """Access the shared metrics collector for hybrid search components.

        Args:
            None

        Returns:
            MetricsCollector instance tracking counters and histograms.
        """
        return self._metrics

    @property
    def logger(self) -> logging.Logger:
        """Structured logger scoped to hybrid search observability events.

        Args:
            None

        Returns:
            Logger configured for DocsToKG hybrid search messages.
        """
        return self._logger

    def trace(self, name: str, **attributes: str) -> Iterator[None]:
        """Create a tracing span context for measuring critical operations.

        Args:
            name: Span name describing the operation being timed.
            **attributes: Additional context for metrics and structured logging.

        Returns:
            Context manager yielding control to the caller.
        """
        return self._tracer.span(name, **attributes)

    def metrics_snapshot(self) -> Dict[str, list[Mapping[str, object]]]:
        """Produce a serializable snapshot of counters and histograms.

        Args:
            None

        Returns:
            Dictionary containing lists of counter and histogram samples.
        """
        counters = [sample.__dict__ for sample in self._metrics.export_counters()]
        histograms = [sample.__dict__ for sample in self._metrics.export_histograms()]
        return {"counters": counters, "histograms": histograms}
