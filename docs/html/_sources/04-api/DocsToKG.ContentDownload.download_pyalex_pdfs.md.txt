# 1. Module: download_pyalex_pdfs

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

### `_utc_timestamp()`

Return the current time as an ISO 8601 UTC timestamp.

Returns:
Timestamp string formatted with a trailing ``'Z'`` suffix.

### `_has_pdf_eof(path)`

Check whether a PDF file terminates with the ``%%EOF`` marker.

Args:
path: Path to the candidate PDF file.
window_bytes: Number of trailing bytes to scan for the EOF marker.

Returns:
``True`` if the file ends with ``%%EOF``; ``False`` otherwise.

### `_update_tail_buffer(buffer, chunk)`

Maintain the trailing ``limit`` bytes of a streamed download.

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

### `_parse_domain_interval(value)`

Parse ``DOMAIN=SECONDS`` CLI arguments for domain throttling.

Args:
value: Argument provided via ``--domain-min-interval``.

Returns:
Tuple containing the normalized domain name and interval seconds.

Raises:
argparse.ArgumentTypeError: If the argument is malformed or negative.

### `_make_session(headers)`

Create a :class:`requests.Session` configured for polite crawling.

Adapter-level retries remain disabled so :func:`request_with_retries` fully
controls backoff, ensuring deterministic retry counts across the pipeline.

Args:
headers (Dict[str, str]): Header dictionary returned by
:func:`load_resolver_config`. The mapping must already include the
project user agent and ``mailto`` contact address. A copy of the
mapping is applied to the outgoing session so callers can reuse
mutable dictionaries without side effects.
pool_connections: Minimum pool size shared across HTTP and HTTPS adapters.
pool_maxsize: Upper bound for per-host connections retained in the pool.

Returns:
requests.Session: Session with connection pooling enabled and retries
disabled at the adapter level so the application layer governs backoff.

Notes:
Each worker should call this helper to obtain an isolated session instance.
Example:

>>> _make_session({"User-Agent": "DocsToKGDownloader/1.0", "mailto": "ops@example.org"})  # doctest: +ELLIPSIS
<requests.sessions.Session object at ...>

The returned session is safe for concurrent HTTP requests because
:class:`requests.adapters.HTTPAdapter` manages a thread-safe connection
pool. Avoid mutating shared session state (for example ``session.headers.update``)
once the session is handed to worker threads.

### `load_previous_manifest(path)`

Load manifest JSONL entries indexed by work identifier.

Args:
path: Path to a previous manifest JSONL log, or None.

Returns:
Tuple containing:
- Mapping of work_id -> url -> manifest payloads
- Set of work IDs already completed

Raises:
json.JSONDecodeError: If the manifest contains invalid JSON.
ValueError: If entries omit required fields or use deprecated schemas.

### `build_manifest_entry(artifact, resolver, url, outcome, html_paths)`

Create a manifest entry summarizing a download attempt.

Args:
artifact: Work artifact providing metadata.
resolver: Resolver name responsible for the download.
url: URL that was attempted.
outcome: Download outcome describing classification and metadata.
html_paths: Any HTML paths captured during the attempt.
dry_run: Whether this was a dry-run execution.
reason: Optional reason string for failures.

Returns:
ManifestEntry populated with download metadata.

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

### `read_resolver_config(path)`

Read resolver configuration from JSON or YAML files.

Args:
path: Path to the configuration file.

Returns:
Parsed configuration mapping.

Raises:
RuntimeError: If YAML parsing is requested but PyYAML is unavailable.

### `_seed_resolver_toggle_defaults(config, resolver_names)`

Ensure resolver toggles include defaults for every known resolver.

### `apply_config_overrides(config, data, resolver_names)`

Apply overrides from configuration data onto a ResolverConfig.

Args:
config: Resolver configuration object to mutate.
data: Mapping loaded from a configuration file.
resolver_names: Known resolver names. Defaults are applied after overrides.

Returns:
None

### `load_resolver_config(args, resolver_names, resolver_order_override)`

Construct resolver configuration combining CLI, config files, and env vars.

Args:
args: Parsed CLI arguments.
resolver_names: Sequence of resolver names supported by the pipeline.
resolver_order_override: Optional override list for resolver order.

Returns:
Populated ResolverConfig instance.

Raises:
FileNotFoundError: If the resolver configuration file does not exist.
RuntimeError: If YAML parsing is requested but PyYAML is unavailable.

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
dry_run: When True, simulate downloads without writing files.
list_only: When True, record candidate URLs without fetching content.
extract_html_text: Whether to extract plaintext from HTML artefacts.
previous_lookup: Mapping of work_id/URL to prior manifest entries.
resume_completed: Set of work IDs already processed in resume mode.
max_bytes: Optional size limit per download in bytes.
sniff_bytes: Number of leading bytes to buffer for payload inference.
min_pdf_bytes: Minimum PDF size accepted when HEAD prechecks fail.
tail_check_bytes: Tail window size used to detect embedded HTML payloads.

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

### `__post_init__(self)`

*No documentation available.*

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

### `_submit(work_item)`

Submit a work item to the executor for asynchronous processing.

### `_runner()`

Process a single work item within a worker-managed session.

## 3. Classes

### `WorkArtifact`

Normalized artifact describing an OpenAlex work to process.

### `DownloadState`

State machine for streaming downloads.
