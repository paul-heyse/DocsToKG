# 1. Module: cli

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.cli``.

## 1. Overview

OpenAlex PDF Downloader CLI

This module implements the command-line interface responsible for downloading
open-access PDFs referenced by OpenAlex works. It combines resolver discovery,
content classification, manifest logging, and polite crawling behaviours into a
single executable entrypoint. The implementation aligns with the modular content
download architecture documented in the OpenSpec proposal and exposes hooks for
custom resolver configuration, dry-run execution, and manifest resume logic.

Key Features:
- Threaded resolver pipeline with conditional request caching.
- Thread-safe JSONL/CSV logging including manifest entries and attempt metrics.
- Streaming content hashing with corruption detection heuristics for PDFs.
- Centralised retry handling and polite header management for resolver requests.
- Single-request download path (no redundant HEAD probes) with classification via
  streamed sniff buffers.
- CLI flags for controlling topic selection, time ranges, resolver order, and
  polite crawling identifiers.
- Optional global URL deduplication and domain-level throttling controls for
  large-scale crawls.

Dependencies:
- `requests`: HTTP communication and connection pooling adapters.
- `pyalex`: Query construction for OpenAlex works and topics.
- `DocsToKG.ContentDownload` submodules: Resolver pipeline orchestration,
  conditional caching, and shared utilities.

Usage:
    python -m DocsToKG.ContentDownload.cli \
        --topic "knowledge graphs" --year-start 2020 --year-end 2023 \
        --out ./pdfs --resolver-config download_config.yaml

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

### `_parse_size(value)`

Parse human-friendly size strings (e.g., ``10MB``) into bytes.

### `_parse_domain_interval(value)`

Parse ``DOMAIN=SECONDS`` CLI arguments for domain throttling.

Args:
value: Argument provided via ``--domain-min-interval``.

Returns:
Tuple containing the normalized domain name and interval seconds.

Raises:
argparse.ArgumentTypeError: If the argument is malformed or negative.

### `_parse_domain_bytes_budget(value)`

Parse ``DOMAIN=BYTES`` CLI arguments for domain byte budgets.

### `_parse_domain_token_bucket(value)`

Parse ``DOMAIN=RPS[:capacity=X]`` specifications into bucket configs.

### `_parse_budget(value)`

Parse ``requests=N`` or ``bytes=N`` budget specifications.

### `_apply_content_addressed_storage(dest_path, sha256)`

Move `dest_path` into a content-addressed location and create a symlink.

### `_collect_location_urls(work)`

Return landing/PDF/source URL collections derived from OpenAlex metadata.

Args:
work: OpenAlex work payload as returned by the Works API.

Returns:
Dictionary containing ``landing``, ``pdf``, and ``sources`` URL lists.

### `build_query(args)`

Build a pyalex Works query based on CLI arguments.

Args:
args: Parsed command-line arguments.

Returns:
Configured Works query object ready for iteration.

### `_lookup_topic_id(topic_text)`

Cached helper to resolve an OpenAlex topic identifier.

### `resolve_topic_id_if_needed(topic_text)`

Resolve a textual topic label into an OpenAlex topic identifier.

Args:
topic_text: Free-form topic text supplied via CLI.

Returns:
OpenAlex topic identifier string if resolved, else None.

### `create_artifact(work, pdf_dir, html_dir)`

Normalize an OpenAlex work into a WorkArtifact instance.

Args:
work: Raw OpenAlex work payload.
pdf_dir: Directory where PDFs should be stored.
html_dir: Directory where HTML resources should be stored.

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
context: Execution context containing ``dry_run``, ``extract_html_text``,
and ``previous`` manifest lookup data.

Returns:
DownloadOutcome describing the result of the download attempt including
streaming hash metadata when available.

Notes:
A lightweight HEAD preflight is issued when the caller has not already
validated the URL. This mirrors the resolver pipeline behaviour and
keeps dry-run tests deterministic.

Raises:
OSError: If writing the downloaded payload to disk fails.
TypeError: If conditional response parsing returns unexpected objects.

### `iterate_openalex(query, per_page, max_results)`

Iterate over OpenAlex works respecting pagination and limits.

Args:
query: Configured Works query instance.
per_page: Number of results to request per page.
max_results: Optional maximum number of works to yield.

Yields:
Work payload dictionaries returned by the OpenAlex API.

Returns:
Iterable yielding the same work payload dictionaries for convenience.

### `process_one_work(work, session, pdf_dir, html_dir, pipeline, logger, metrics)`

Process a single OpenAlex work through the resolver pipeline.

Args:
work: OpenAlex work payload from :func:`iterate_openalex`.
session: Requests session configured for resolver usage.
pdf_dir: Directory where PDF artefacts are written.
html_dir: Directory where HTML artefacts are written.
pipeline: Resolver pipeline orchestrating downstream resolvers.
logger: Structured attempt logger capturing manifest records.
metrics: Resolver metrics collector.
options: :class:`DownloadOptions` describing download behaviour for the work.

Returns:
Dictionary summarizing the outcome (saved/html_only/skipped flags).

Raises:
requests.RequestException: Propagated if resolver HTTP requests fail
unexpectedly outside guarded sections.
Exception: Bubbling from resolver pipeline internals when not handled.

### `main()`

Parse CLI arguments, configure resolvers, and execute downloads.

The entrypoint wires together argument parsing, resolver configuration,
logging setup, and the resolver pipeline orchestration documented in the
modular content download specification. It is exposed both as the module's
``__main__`` handler and via `python -m`.

Args:
None

Returns:
None

### `is_allowed(self, session, url, timeout)`

Return ``False`` when robots.txt forbids fetching ``url``.

### `_fetch(self, session, origin, timeout)`

Fetch and parse the robots.txt policy for ``origin``.

### `_append_location(loc)`

Accumulate location URLs from a single OpenAlex location record.

Args:
loc: Location dictionary as returned by OpenAlex (may be None).

### `_session_factory()`

Return a new :class:`requests.Session` using the run's polite headers.

The factory is invoked by worker threads to obtain an isolated session
that inherits the resolver configuration's polite identification
headers. Creating sessions through this helper ensures each worker
reuses the shared retry configuration while keeping connection pools
thread-local.

### `_record_result(res)`

Update aggregate counters based on a single work result.

### `_should_stop()`

*No documentation available.*

### `_stream_chunks()`

*No documentation available.*

### `_submit(work_item)`

Submit a work item to the executor for asynchronous processing.

### `_runner()`

Process a single work item within a worker-managed session.

## 3. Classes

### `DownloadOptions`

Stable collection of per-run download settings applied to each work item.

### `DownloadState`

State machine for streaming downloads.

### `_MaxBytesExceeded`

Internal signal raised when the stream exceeds the configured byte budget.

### `RobotsCache`

Cache robots.txt policies per host and evaluate allowed URLs.
