# 1. Module: download

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.download``.

## 1. Overview

Download orchestration helpers for the content acquisition pipeline.

This module coordinates the streaming download workflow, tying together
resolver outputs, HTTP policy enforcement, and telemetry reporting. It exposes
utilities that transform resolver candidates into stored artifacts while
respecting retry policies, robots.txt directives, and classification rules.

## 2. Functions

### `ensure_dir(path)`

Create a directory if it does not already exist.

Args:
path: Directory path to create when absent.

Returns:
None

Raises:
OSError: If the directory cannot be created because of permissions.

### `prepare_candidate_download(session, artifact, url, referer, timeout, ctx)`

Prepare request metadata prior to streaming the download.

### `stream_candidate_payload(plan)`

Execute the streaming phase for a prepared download plan.

### `finalize_candidate_download(plan, stream)`

Combine streaming results into a finalized :class:`DownloadOutcome`.

### `get_strategy_for_classification(classification)`

Return a strategy implementation for the provided classification.

### `_cached_sha256(signature)`

Compute and cache SHA-256 digests keyed by path, size, and mtime.

### `_validate_cached_artifact(result)`

Return validation success flag and mode (``fast_path`` or ``digest``).

### `_apply_content_addressed_storage(dest_path, sha256)`

Move `dest_path` into a content-addressed location and create a symlink.

### `validate_classification(classification, artifact, options)`

Validate resolver classification against configured expectations.

### `handle_resume_logic(artifact, previous_index, options)`

Normalise previous attempts and detect early skip conditions.

### `cleanup_sidecar_files(artifact, classification, options)`

Remove temporary sidecar files for the given artifact classification.

When resume is explicitly enabled *and* the remote endpoint confirmed support
(``resume_supported=True``), the partial artifact is preserved so the caller
can retry with a ``Range`` request. This behaviour is documented to avoid
surprising cleanups in the rare deployments that still experiment with
resumable transfers.

### `build_download_outcome()`

Compose a :class:`DownloadOutcome` with shared validation logic.

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

Phase diagram::

[prepare_candidate_download]
|--(robots/cache skip)--> outcome
v
[stream_candidate_payload]
|--(cached/error)-------> outcome
v
[finalize_candidate_download] ---> outcome

Each helper owns a single responsibility: preflight assembles request
context, streaming performs network I/O, and finalization converts the
structured stream result into a manifest-ready :class:`DownloadOutcome`.

### `process_one_work(work, session, pdf_dir, html_dir, xml_dir, pipeline, logger, metrics)`

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
options: :class:`DownloadConfig` describing download behaviour for the work.
session_factory: Optional callable returning a thread-local session for
resolver execution when concurrency is enabled.

Returns:
Dictionary summarizing the outcome (saved/html_only/skipped flags).

Raises:
requests.RequestException: Propagated if resolver HTTP requests fail
unexpectedly outside guarded sections.
Exception: Bubbling from resolver pipeline internals when not handled.

### `__post_init__(self)`

*No documentation available.*

### `_normalize_mapping(value)`

*No documentation available.*

### `_normalize_resume_lookup(value)`

*No documentation available.*

### `_normalize_resume_completed(value)`

*No documentation available.*

### `_coerce_int(value, default)`

*No documentation available.*

### `_coerce_optional_positive(value)`

*No documentation available.*

### `to_context(self, overrides)`

Convert the configuration into a :class:`DownloadContext`.

### `from_options(cls, options)`

Build a configuration instance from legacy option surfaces.

### `to_context(self, overrides)`

Return a `DownloadContext` derived from this options payload.

Args:
overrides: Optional mapping of field names to values that should
replace the existing attributes when constructing the context.

Returns:
DownloadContext: Fully-populated context object consumed by the
content download pipeline.

### `should_download(self, artifact, context)`

Decide whether the current artifact warrants a fresh download attempt.

Args:
artifact: The work artifact under consideration.
context: Mutable state shared across download strategy phases.

Returns:
True when the strategy should perform a download, False when it should
short-circuit the workflow (for example, due to resume state).

### `process_response(self, response, artifact, context)`

Process the HTTP response to derive classification metadata.

Args:
response: Raw HTTP response returned by the download request.
artifact: Work artifact metadata associated with the response.
context: Shared strategy state that accumulates response details.

Returns:
The classification that should be assigned to the downloaded artifact.

### `finalize_artifact(self, artifact, classification, context)`

Assemble the final download outcome after all processing steps.

Args:
artifact: Work artifact being finalized.
classification: Final classification selected for the artifact.
context: Strategy context containing response metadata and paths.

Returns:
Structured outcome describing the finalized download results.

### `should_download(self, artifact, context)`

Evaluate whether to start a download for the provided artifact.

Args:
artifact: Work artifact describing the requested content.
context: Mutable strategy context shared across download phases.

Returns:
False when the download is skipped due to configuration, True otherwise.

### `process_response(self, response, artifact, context)`

Capture the response and select a classification if not already set.

Args:
response: HTTP response received for the download attempt.
artifact: Work artifact metadata being processed.
context: Strategy context that caches response metadata.

Returns:
Classification derived either from the response or preconfigured hints.

### `finalize_artifact(self, artifact, classification, context)`

Convert the accumulated strategy context into a download outcome.

Args:
artifact: Work artifact associated with the download.
classification: Final classification selected for the artifact.
context: Strategy context populated during prior phases.

Returns:
Structured `DownloadOutcome` describing results and metadata.

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

*No documentation available.*

## 3. Classes

### `DownloadConfig`

Unified download configuration shared by CLI, pipeline, and runner.

### `DownloadOptions`

Compatibility shim preserving the legacy ``DownloadOptions`` surface.

### `ValidationResult`

Outcome of validating an artifact classification.

### `ResumeDecision`

Decision container describing resume handling for a work artifact.

### `DownloadStrategyContext`

Mutable state shared between strategy phases for a single download.

### `DownloadPreflightPlan`

Prepared inputs required to stream a candidate download.

### `DownloadStreamResult`

Structured payload returned by :func:`stream_candidate_payload`.

### `DownloadStrategy`

Protocol implemented by artifact-specific download strategies.

### `_BaseDownloadStrategy`

*No documentation available.*

### `PdfDownloadStrategy`

Download strategy that enforces PDF-specific processing rules.

### `HtmlDownloadStrategy`

Download strategy tailored for HTML artifacts.

### `XmlDownloadStrategy`

Download strategy tailored for XML artifacts.

### `DownloadState`

State machine for streaming downloads.

### `RobotsCache`

Cache robots.txt policies per host and evaluate allowed URLs with TTL.
