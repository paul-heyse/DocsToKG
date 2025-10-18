# 1. Module: download

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.download``.

## 1. Overview

Download orchestration helpers for the content acquisition pipeline.

This module coordinates the streaming download workflow, tying together
resolver outputs, HTTP policy enforcement, and telemetry reporting. It exposes
utilities that transform resolver candidates into stored artifacts while
respecting retry budgets, robots.txt directives, and classification rules.

## 2. Functions

### `ensure_dir(path)`

Create a directory if it does not already exist.

Args:
path: Directory path to create when absent.

Returns:
None

Raises:
OSError: If the directory cannot be created because of permissions.

### `validate_classification(classification, artifact, options)`

Validate a detected classification against expectations and size rules.

Args:
classification: Reported :class:`Classification` for the candidate payload.
artifact: Work artifact metadata under evaluation.
options: :class:`DownloadOptions` (or mapping) controlling validation toggles.

Returns:
``ValidationResult`` describing success, reason codes, and telemetry payloads.

### `handle_resume_logic(artifact, previous_index, options)`

Check prior manifest entries to determine whether a work should be skipped.

Args:
artifact: Work artifact representing the current OpenAlex record.
previous_index: Mapping of work IDs and resolver results from prior runs.
options: :class:`DownloadOptions` describing resume and force-download toggles.

Returns:
``ResumeDecision`` capturing skip status, reuse metadata, and prior outcomes.

### `cleanup_sidecar_files(artifact, classification, options)`

Remove partial downloads and mismatched sidecar files following classification.

Args:
artifact: Work artifact describing file destinations.
classification: Final :class:`Classification` returned by a strategy.
options: :class:`DownloadOptions` controlling dry-run and HTML extraction flags.

Returns:
None

### `build_download_outcome(...)`

Compose a :class:`DownloadOutcome` with shared validation logic for artifacts.

The helper enforces PDF heuristics (minimum byte threshold, HTML tail detection,
``%%EOF`` marker validation) and clears corrupted files before returning a
normalized outcome structure. It records SHA-256 digests, HTTP metadata, retry
headers, and extracted-text paths when available. Callers supply the
``DownloadContext`` options via the ``options`` keyword to ensure dry-run and
content-addressable storage preferences are honoured.

### `_build_download_outcome(...)`

Compatibility wrapper invoking :func:`build_download_outcome` with legacy
parameters used by historical call sites.

### `_validate_cached_artifact(result)`

Return ``True`` when cached artefact metadata matches on-disk state.

### `_apply_content_addressed_storage(dest_path, sha256)`

Move `dest_path` into a content-addressed location and create a symlink.

### `_collect_location_urls(work)`

Return landing/PDF/source URL collections derived from OpenAlex metadata.

Args:
work: OpenAlex work payload as returned by the Works API.

Returns:
Dictionary containing ``landing``, ``pdf``, and ``sources`` URL lists.

### `_cohort_order_for(artifact)`

Return a resolver order tailored to the artifact's identifiers.

### `create_artifact(work, pdf_dir, html_dir, xml_dir)`

Normalize an OpenAlex work into a WorkArtifact instance.

Args:
work: Raw OpenAlex work payload.
pdf_dir: Directory where PDFs should be stored.
html_dir: Directory where HTML resources should be stored.
xml_dir: Directory where XML resources should be stored.

Returns:
WorkArtifact describing the work and candidate URLs.

Raises:
KeyError: If required identifiers are missing from the work payload.

### `download_candidate(session, artifact, url, referer, timeout, context, head_precheck_passed)`

Download a single candidate URL and classify the payload.

Args:
session: HTTP session capable of issuing retried requests via the
centralised :func:`request_with_retries` helper.
artifact: Work metadata and output directory handles for the current record.
url: Candidate download URL discovered by a resolver.
referer: Optional referer header override provided by the resolver.
timeout: Per-request timeout in seconds.
context: :class:`DownloadContext` (or mapping convertible to it) containing
``dry_run``, ``extract_html_text``, prior manifest lookups, and optional
``progress_callback``.

Progress callback signature: ``callback(bytes_downloaded, total_bytes, url)``
where ``total_bytes`` may be ``None`` if Content-Length is unavailable.

Returns:
DownloadOutcome describing the result of the download attempt including
streaming hash metadata when available.

Strategies implementing :class:`DownloadStrategy` are used to specialise
validation and finalisation for PDF, HTML, and XML artifacts. The strategy is
selected based on the detected classification before the response body is
persisted.

Notes:
A lightweight HEAD preflight is issued when the caller has not already
validated the URL. This mirrors the resolver pipeline behaviour and
keeps dry-run tests deterministic.

Progress callbacks are invoked approximately every 128KB to balance
responsiveness with performance overhead.

Raises:
OSError: If writing the downloaded payload to disk fails.
TypeError: If conditional response parsing returns unexpected objects.

### `process_one_work(work, session, pdf_dir, html_dir, xml_dir, pipeline, logger, metrics, *, options, session_factory, strategy_selector)`

Process a single work artifact through the resolver pipeline.

Args:
work: Either a preconstructed :class:`WorkArtifact` or a raw OpenAlex
work payload. Raw payloads are normalised via :func:`create_artifact`.
session: Requests session configured for resolver usage.
pdf_dir: Directory where PDF artefacts are written.
html_dir: Directory where HTML artefacts are written.
xml_dir: Directory where XML artefacts are written.
pipeline: Resolver pipeline orchestrating downstream resolvers.
logger: Structured attempt logger capturing manifest records.
metrics: Resolver metrics collector.
options: :class:`DownloadOptions` describing download behaviour for the work.
session_factory: Optional callable returning a thread-local session for
    resolver execution when concurrency is enabled.
strategy_selector: Optional callable returning a :class:`DownloadStrategy`
    implementation for a detected :class:`Classification`.

Returns:
Dictionary summarizing the outcome (saved/html_only/skipped flags).

Raises:
requests.RequestException: Propagated if resolver HTTP requests fail
unexpectedly outside guarded sections.
Exception: Bubbling from resolver pipeline internals when not handled.

### `get_strategy_for_classification(classification)`

Return the registered :class:`DownloadStrategy` for a detected classification.

Strategies for PDF-like payloads collapse onto :class:`PdfDownloadStrategy`
while HTML and XML classifiers map to their dedicated implementations. Unknown
or legacy classifications reuse the PDF strategy to preserve historical
behaviour.

### `is_allowed(self, session, url, timeout)`

Return ``False`` when robots.txt forbids fetching ``url``.

### `_lookup_parser(self, session, origin, timeout)`

*No documentation available.*

### `_fetch(self, session, origin, timeout)`

Fetch and parse the robots.txt policy for ``origin``.

### `_append_location(loc)`

Accumulate location URLs from a single OpenAlex location record.

Args:
loc: Location dictionary as returned by OpenAlex (may be None).

### `_stream_chunks()`

Optimized streaming: write prefetched chunks inline to avoid double storage.

## 3. Classes

### `DownloadOptions`

Stable collection of per-run download settings applied to each work item.

### `DownloadState`

State machine for streaming downloads.

### `DownloadStrategy`

Protocol describing the shared interface implemented by artifact-specific
download strategies. Each strategy decides whether work should proceed,
interprets HTTP responses, and constructs the final :class:`DownloadOutcome`.

### `DownloadStrategyContext`

Mutable state container passed between strategy phases. It captures the active
response object, destination path, derived telemetry, and configuration used by
strategy implementations.

### `PdfDownloadStrategy`

Concrete :class:`DownloadStrategy` handling PDF artifacts. It validates the
trailing ``%%EOF`` marker, enforces minimum byte thresholds, and builds outcomes
that include SHA-256 metadata when available.

### `HtmlDownloadStrategy`

Concrete :class:`DownloadStrategy` for HTML artifacts. It honours text extraction
preferences, preserves existing downloads when list-only mode is active, and
ensures outcomes capture HTML classification metadata.

### `XmlDownloadStrategy`

Concrete :class:`DownloadStrategy` used for XML artifacts. It shares the common
finalisation flow while deferring classification-specific behaviour to the
strategy protocol.

### `_MaxBytesExceeded`

Internal signal raised when the stream exceeds the configured byte budget.

### `RobotsCache`

Cache robots.txt policies per host and evaluate allowed URLs with TTL.
