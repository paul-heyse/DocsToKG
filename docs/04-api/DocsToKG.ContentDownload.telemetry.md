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

*No documentation available.*

### `log_manifest(self, entry)`

*No documentation available.*

### `log_summary(self, summary)`

*No documentation available.*

### `close(self)`

*No documentation available.*

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

*No documentation available.*

### `log_manifest(self, entry)`

*No documentation available.*

### `log_summary(self, summary)`

*No documentation available.*

### `close(self)`

*No documentation available.*

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

### `log_attempt(self, record)`

*No documentation available.*

### `log_manifest(self, entry)`

*No documentation available.*

### `log_summary(self, summary)`

*No documentation available.*

### `close(self)`

*No documentation available.*

### `__enter__(self)`

*No documentation available.*

### `__exit__(self, exc_type, exc, tb)`

*No documentation available.*

## 3. Classes

### `ManifestEntry`

Structured manifest entry describing a resolved artifact.

Attributes:
timestamp: ISO-8601 timestamp describing when the artifact was stored.
work_id: Primary identifier (e.g., OpenAlex work ID) for the artifact.
title: Human-readable work title.
publication_year: Optional publication year derived from upstream data.
resolver: Resolver name that sourced the artifact, when known.
url: Final URL from which the artifact was fetched.
path: Local filesystem path where the artifact is stored.
classification: Classifier label describing the download outcome.
content_type: MIME type or equivalent classification when available.
reason: Diagnostic reason for failure or classification notes.
html_paths: Any extracted HTML content paths stored alongside the PDF.
sha256: Optional SHA256 digest of the stored artifact.
content_length: Content size in bytes when reported by the server.
etag: Server-provided entity tag when available.
last_modified: HTTP ``Last-Modified`` header value if supplied.
extracted_text_path: Path to extracted text artefacts when produced.
dry_run: Flag indicating whether the artifact was processed in dry-run mode.

Examples:
>>> entry = ManifestEntry(
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
