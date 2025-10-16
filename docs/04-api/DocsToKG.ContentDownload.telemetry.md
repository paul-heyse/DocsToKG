# 1. Module: telemetry

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.telemetry``.

## 1. Overview

Telemetry contracts shared by DocsToKG content download tooling.

Centralises manifest and attempt payload structures so the CLI, resolver
pipeline, and downstream analytics speak a consistent schema. Keeping these
definitions in one place prevents subtle drift between components while allowing
test doubles to implement the same interfaces.

Args:
    None

Returns:
    None

Raises:
    None

## 2. Functions

### `_utc_timestamp()`

Return the current time as an ISO 8601 UTC timestamp.

### `_ensure_parent_exists(path)`

Ensure the parent directory for ``path`` exists.

### `load_previous_manifest(path)`

Load JSONL manifest entries indexed by work ID and normalised URL.

### `load_manifest_url_index(path)`

Return a mapping of normalised URLs to manifest metadata from SQLite.

### `build_manifest_entry(artifact, resolver, url, outcome, html_paths)`

Create a manifest entry summarising a download attempt.

### `log_attempt(self, record)`

Record a resolver attempt.

Args:
record: Resolver attempt metadata.
timestamp: Optional timestamp override for the event.

Returns:
None

Raises:
Exception: Implementations may propagate write failures.

### `log_manifest(self, entry)`

Persist a manifest entry.

Args:
entry: Manifest payload describing a successfully processed artifact.

Returns:
None

Raises:
Exception: Implementations may propagate write failures.

### `log_summary(self, summary)`

Store aggregated run metrics.

Args:
summary: Aggregated counters or diagnostic information.

Returns:
None

Raises:
Exception: Implementations may propagate write failures.

### `close(self)`

Release any resources held by the sink.

Args:
None

Returns:
None

Raises:
Exception: Implementations may propagate shutdown failures.

### `_write(self, payload)`

*No documentation available.*

### `log_attempt(self, record)`

Append an attempt record to the JSONL log.

Args:
record: Attempt metadata captured during resolver execution.
timestamp: Optional ISO-8601 timestamp overriding the current time.

### `log_manifest(self, entry)`

Append a manifest record representing a persisted document.

Args:
entry: Manifest entry produced once a resolver finalises output.

### `log_summary(self, summary)`

Append a run-level summary describing aggregate metrics.

Args:
summary: Mapping containing summary counters and metadata.

### `close(self)`

Flush buffered data and close the underlying JSONL file handle.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

Append an attempt record to the CSV log.

Args:
record: Attempt metadata captured during resolver execution.
timestamp: Optional override for the timestamp column.

### `log_manifest(self, entry)`

Ignore manifest writes for CSV sinks (interface compatibility).

### `log_summary(self, summary)`

Ignore summary writes for CSV sinks (interface compatibility).

### `close(self)`

Flush buffered data and close the CSV file handle.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

Fan out an attempt record to each sink in the composite.

### `log_manifest(self, entry)`

Fan out manifest records to every sink in the collection.

### `log_summary(self, summary)`

Fan out summary payloads to every sink in the collection.

### `close(self)`

Close sinks while capturing the first raised exception.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

No-op to satisfy the telemetry sink interface.

### `log_summary(self, summary)`

No-op because the manifest index only reacts to manifests.

### `log_manifest(self, entry)`

Index the latest manifest metadata for ``entry.work_id``.

### `close(self)`

Write the collected manifest index to disk once.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

No-op to satisfy the telemetry sink interface.

### `log_summary(self, summary)`

No-op because the CSV sink only records manifest events.

### `_normalise(self, value)`

*No documentation available.*

### `log_manifest(self, entry)`

Store the most recent manifest attributes for ``entry.work_id``.

### `close(self)`

Flush collected manifest rows to the CSV file.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

Accept attempt telemetry for interface parity without persisting it.

Args:
record: Resolver attempt metadata emitted by the download pipeline.
timestamp: Optional ISO-8601 timestamp used when callers override the capture time.

Returns:
None

### `log_manifest(self, entry)`

Receive manifest updates while keeping in-memory summary-only semantics.

Args:
entry: Manifest entry describing the document artifact that was processed.

Returns:
None

### `log_summary(self, summary)`

Capture the latest run summary prior to shutdown.

### `close(self)`

Write the captured summary to disk exactly once.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

Persist a resolver attempt event for downstream document analytics.

Args:
record: Structured attempt data captured during document resolution.
timestamp: Optional ISO-8601 timestamp overriding the event capture time.

Returns:
None

### `log_manifest(self, entry)`

Record manifest outcomes for the processed document artifact.

Args:
entry: Manifest entry describing the document state after processing.

Returns:
None

### `log_summary(self, summary)`

Store the aggregated run summary in the summaries table.

Args:
summary: Aggregated counters and timing metadata for the run.

Returns:
None

### `close(self)`

Commit outstanding changes and dispose of the SQLite connection.

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

## 3. Classes

### `ManifestEntry`

Structured manifest entry describing a resolved artifact.

Attributes:
schema_version: Integer identifying the manifest schema revision.
timestamp: ISO-8601 timestamp describing when the artifact was stored.
work_id: Primary identifier (e.g., OpenAlex work ID) for the artifact.
title: Human-readable work title.
publication_year: Optional publication year derived from upstream data.
resolver: Resolver name that sourced the artifact, when known.
url: Final URL from which the artifact was fetched.
path: Local filesystem path where the artifact is stored.
classification: Classifier label describing the download outcome.
content_type: MIME type or equivalent classification when available.
reason: Diagnostic reason code (see :class:`ReasonCode`).
reason_detail: Optional human-readable diagnostic detail.
html_paths: Any extracted HTML content paths stored alongside the PDF.
sha256: Optional SHA256 digest of the stored artifact.
content_length: Content size in bytes when reported by the server.
etag: Server-provided entity tag when available.
last_modified: HTTP ``Last-Modified`` header value if supplied.
extracted_text_path: Path to extracted text artefacts when produced.
dry_run: Flag indicating whether the artifact was processed in dry-run mode.

Examples:
>>> entry = ManifestEntry(
...     schema_version=2,
...     timestamp="2025-01-01T00:00:00Z",
...     work_id="W123",
...     title="Example",
...     publication_year=2024,
...     resolver="openalex",
...     url="https://example.org/doc.pdf",
...     path="/tmp/doc.pdf",
...     classification="success",
...     content_type="application/pdf",
...     reason=None,
... )
>>> entry.work_id
'W123'

### `AttemptSink`

Protocol implemented by telemetry sinks used by the pipeline and CLI.

Implementations typically write to JSONL/CSV and may maintain external
storage handles; consumers should always call :meth:`close` when finished.

Attributes:
None

Examples:
>>> class InMemorySink:
...     def __init__(self) -> None:
...         self.attempts = []
...     def log_attempt(self, record, *, timestamp=None):
...         self.attempts.append(record)
...     def log_manifest(self, entry):
...         pass
...     def log_summary(self, summary):
...         pass
...     def close(self):
...         pass
>>> isinstance(InMemorySink(), AttemptSink)
True

### `JsonlSink`

Thread-safe sink that streams attempt, manifest, and summary events to JSONL files.

### `CsvSink`

Lightweight sink that mirrors attempt records into a CSV for spreadsheet review.

### `MultiSink`

Composite sink that fans out logging calls to multiple sinks.

### `ManifestIndexSink`

Maintain a JSON index mapping work IDs to their latest PDF artefacts.

### `LastAttemptCsvSink`

Write a CSV snapshot containing the most recent manifest entry per work.

### `SummarySink`

Persist the final metrics summary for a run as JSON.

### `SqliteSink`

Persist attempts, manifests, and summary records to a SQLite database.
