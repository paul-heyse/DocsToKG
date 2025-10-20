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

### `normalize_manifest_path(path)`

Return an absolute filesystem path suitable for manifest storage.

Older manifests stored artifact paths relative to the working directory in
effect when the run executed. To keep those entries usable when resuming a
run from another working directory, callers may provide ``base`` so that
relative paths can be resolved against a known directory such as the
manifest location. For new entries, the helper upgrades any provided path
(``Path`` objects or strings) to an absolute representation.

### `_utc_timestamp()`

Return the current time as an ISO 8601 UTC timestamp.

### `_ensure_parent_exists(path)`

Ensure the parent directory for ``path`` exists.

### `_manifest_entry_from_sqlite_row(run_id, work_id, url, canonical_url, original_url, schema_version, classification, reason, reason_detail, path_value, path_mtime_ns, sha256, content_length, etag, last_modified)`

Convert a SQLite manifest row into resume metadata.

### `_iter_resume_rows_from_sqlite(sqlite_path)`

Yield manifest resume rows from ``sqlite_path`` lazily.

### `_load_resume_from_sqlite(sqlite_path)`

Return resume metadata reconstructed from the SQLite manifest cache.

### `looks_like_csv_resume_target(path)`

Return True when ``path`` likely references a CSV attempts log.

### `looks_like_sqlite_resume_target(path)`

Return True when ``path`` likely references a SQLite manifest cache.

### `iter_previous_manifest_entries(path)`

Yield resume manifest entries lazily, preferring SQLite when available.

### `load_previous_manifest(path)`

Load manifest entries indexed by work ID and canonical URL.

### `load_resume_completed_from_sqlite(sqlite_path)`

Return work identifiers completed according to the SQLite manifest cache.

### `load_manifest_url_index(path)`

Return a mapping of canonical URLs to manifest metadata from SQLite.

### `build_manifest_entry(artifact, resolver, url, outcome, html_paths)`

Create a manifest entry summarising a download attempt.

### `__post_init__(self)`

*No documentation available.*

### `get(self, url, default)`

Return manifest metadata for ``url`` if present in the SQLite cache.

Args:
url: URL whose manifest metadata should be retrieved.
default: Fallback value returned when the URL has no cached record.

Returns:
Manifest metadata dictionary when found; otherwise ``default``.

### `__contains__(self, url)`

*No documentation available.*

### `items(self)`

Iterate over cached manifest entries keyed by canonical URL.

Returns:
Iterable of ``(canonical_url, metadata)`` pairs from the cache.

### `iter_existing_paths(self)`

Stream manifest entries whose artifact paths still exist on disk.

### `iter_existing(self)`

Yield manifest entries whose artifact paths still exist on disk.

### `as_dict(self)`

Return a defensive copy of the manifest cache.

Returns:
Dictionary of canonical URLs to manifest metadata.

### `_ensure_loaded(self)`

*No documentation available.*

### `_fetch_one(self, canonical)`

*No documentation available.*

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

### `__enter__(self)`

Enter the runtime context for the sink.

### `__exit__(self, exc_type, exc, tb)`

Exit the runtime context for the sink.

### `log_attempt(self, record)`

Proxy attempt logging to the underlying sink.

Args:
record: Resolver attempt payload to record.
timestamp: Optional ISO-8601 timestamp overriding ``datetime.utcnow``.

### `log_manifest(self, entry)`

Forward manifest entries to the configured sink.

### `log_summary(self, summary)`

Publish the final run summary to downstream sinks.

### `__enter__(self)`

Enter the context manager, delegating to the underlying sink if present.

### `__exit__(self, exc_type, exc, tb)`

Exit the context manager, closing the sink when appropriate.

### `close(self)`

Release sink resources and flush buffered telemetry.

### `record_manifest(self, artifact)`

Construct and emit a manifest entry for ``artifact``.

Args:
artifact: Work artifact being recorded.
resolver: Name of the resolver that produced the outcome.
url: Final URL that yielded the content.
outcome: Download outcome describing classification and metadata.
html_paths: HTML artefacts captured alongside the document.
dry_run: Whether the pipeline ran in dry-run mode.
run_id: Unique identifier for this pipeline execution.
reason: Optional diagnostic reason code.
reason_detail: Optional human-readable reason detail.

Returns:
ManifestEntry: Structured manifest entry persisted via the sink.

### `record_pipeline_result(self, artifact, result)`

Record pipeline output, normalising reason metadata on the way out.

Args:
artifact: Work artifact that was processed.
result: Pipeline result encapsulating resolver outcome details.
dry_run: Whether side effects were suppressed.
run_id: Unique identifier for the current pipeline execution.

Returns:
ManifestEntry: Manifest record produced for downstream sinks.

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

### `_initial_sequence(self)`

*No documentation available.*

### `_rotate(self)`

*No documentation available.*

### `_should_rotate(self, pending_bytes)`

*No documentation available.*

### `_write(self, payload)`

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

### `_ensure_legacy_alias(self)`

Create a compatibility alias at ``.sqlite`` when using ``.sqlite3`` files.

### `_initialise_schema(self, current_version)`

*No documentation available.*

### `_run_migrations(self, current_version)`

*No documentation available.*

### `_safe_add_column(self, table, column, declaration)`

*No documentation available.*

### `_migrate_summary_table(self)`

*No documentation available.*

### `_populate_normalized_urls(self)`

*No documentation available.*

### `_initialise_schema(self)`

*No documentation available.*

### `_populate_from_jsonl(self)`

*No documentation available.*

### `completed_work_ids(self)`

Return work identifiers that completed successfully in the manifest.

### `close(self)`

Close the temporary SQLite database and clean up resources.

### `__getitem__(self, key)`

*No documentation available.*

### `__iter__(self)`

*No documentation available.*

### `__len__(self)`

*No documentation available.*

### `__contains__(self, key)`

*No documentation available.*

### `get(self, key, default)`

Return cached manifest entries for ``key`` or the provided default.

Args:
key: Work identifier to resolve.
default: Value to return when the work ID is not present.

Returns:
Optional[Dict[str, Any]]: Mapping of canonical URLs to manifest entries, if available.

### `enable_preload_on_close(self)`

Load all entries into memory when :meth:`close` is invoked.

### `preload_all_entries(self)`

Populate the in-memory cache with every manifest entry.

### `_preload_all_entries_unlocked(self)`

*No documentation available.*

### `__del__(self)`

*No documentation available.*

### `close(self)`

Close the underlying SQLite connection.

### `__getitem__(self, key)`

*No documentation available.*

### `__iter__(self)`

*No documentation available.*

### `__len__(self)`

*No documentation available.*

### `__contains__(self, key)`

*No documentation available.*

### `get(self, key, default)`

Return work entries for ``key`` if present, otherwise ``default``.

Args:
key: Work identifier to fetch from the SQLite manifest cache.
default: Value to return when the work ID does not exist.

Returns:
Optional[Dict[str, Any]]: Mapping of canonical URLs to manifest entries, if found.

### `_fetch_work_entries(self, work_id)`

*No documentation available.*

### `_fetch_work_entries_unlocked(self, work_id)`

Fetch work entries without acquiring lock (must be called with lock held).

### `_preload_all_entries_unlocked(self)`

Preload all manifest entries into the cache (must be called with lock held).

### `enable_preload_on_close(self)`

Enable preloading all entries when close() is called.

### `preload_all_entries(self)`

Preload all manifest entries into the cache for offline access after close.

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
run_id: Unique identifier for the downloader run that produced the entry.

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

### `ManifestUrlIndex`

Lazy lookup helper for manifest metadata stored in SQLite.

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

### `RunTelemetry`

Telemetry coordinator that centralises manifest aggregation and delegation.

### `JsonlSink`

Thread-safe sink that streams attempt, manifest, and summary events to JSONL files.

### `RotatingJsonlSink`

JSONL sink that rotates the log file once it exceeds a configured size.

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

### `JsonlResumeLookup`

Lazy resume mapping backed by a temporary SQLite index built from JSONL.

### `SqliteResumeLookup`

Lazy resume mapping that fetches manifest entries directly from SQLite.
