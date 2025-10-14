# Module: observability

Lightweight observability primitives for ingestion and retrieval.

## Functions

### `increment(self, name, amount)`

*No documentation available.*

### `observe(self, name, value)`

*No documentation available.*

### `export_counters(self)`

*No documentation available.*

### `export_histograms(self)`

*No documentation available.*

### `span(self, name)`

*No documentation available.*

### `metrics(self)`

*No documentation available.*

### `logger(self)`

*No documentation available.*

### `trace(self, name)`

*No documentation available.*

### `metrics_snapshot(self)`

*No documentation available.*

## Classes

### `CounterSample`

*No documentation available.*

### `HistogramSample`

*No documentation available.*

### `MetricsCollector`

In-memory metrics collector compatible with Prometheus-style summaries.

### `TraceRecorder`

Context manager producing timing spans for tracing.

### `Observability`

Facade for metrics, structured logging, and tracing.
