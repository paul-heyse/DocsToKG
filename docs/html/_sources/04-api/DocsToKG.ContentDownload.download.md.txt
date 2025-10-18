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

### `_build_download_outcome()`

*No documentation available.*

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

Notes:
A lightweight HEAD preflight is issued when the caller has not already
validated the URL. This mirrors the resolver pipeline behaviour and
keeps dry-run tests deterministic.

Progress callbacks are invoked approximately every 128KB to balance
responsiveness with performance overhead.

Raises:
OSError: If writing the downloaded payload to disk fails.
TypeError: If conditional response parsing returns unexpected objects.

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
options: :class:`DownloadOptions` describing download behaviour for the work.
session_factory: Optional callable returning a thread-local session for
resolver execution when concurrency is enabled.

Returns:
Dictionary summarizing the outcome (saved/html_only/skipped flags).

Raises:
requests.RequestException: Propagated if resolver HTTP requests fail
unexpectedly outside guarded sections.
Exception: Bubbling from resolver pipeline internals when not handled.

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

### `_MaxBytesExceeded`

Internal signal raised when the stream exceeds the configured byte budget.

### `RobotsCache`

Cache robots.txt policies per host and evaluate allowed URLs with TTL.
