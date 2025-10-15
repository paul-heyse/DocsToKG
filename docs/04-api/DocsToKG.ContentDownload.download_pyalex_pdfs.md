# 1. Module: download_pyalex_pdfs

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.download_pyalex_pdfs``.

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
    python -m DocsToKG.ContentDownload.download_pyalex_pdfs \
        --topic "knowledge graphs" --year-start 2020 --year-end 2023 \
        --out ./pdfs --resolver-config download_config.yaml

## 1. Functions

### `_utc_timestamp()`

Return the current time as an ISO 8601 UTC timestamp.

Returns:
Timestamp string formatted with a trailing ``'Z'`` suffix.

### `_has_pdf_eof(path)`

Check whether a PDF file terminates with the ``%%EOF`` marker.

Args:
path: Path to the candidate PDF file.

Returns:
``True`` if the file ends with ``%%EOF``; ``False`` otherwise.

### `slugify(text, keep)`

Create a filesystem-friendly slug for a work title.

Args:
text: Input string to normalize into a slug.
keep: Maximum number of characters to retain.

Returns:
Sanitized slug string suitable for filenames.

### `ensure_dir(path)`

Create a directory if it does not already exist.

Args:
path: Directory path to create when absent.

Returns:
None

Raises:
OSError: If the directory cannot be created because of permissions.

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
headers: Header dictionary returned by :func:`load_resolver_config`. The mapping
must already include the project user agent and ``mailto`` contact address.
A copy of the mapping is applied to the outgoing session so callers can
reuse mutable dictionaries without side effects.

Returns:
requests.Session: Session with connection pooling enabled and retries
disabled at the adapter level so the application layer governs
backoff behaviour.

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

### `classify_payload(head_bytes, content_type, url)`

Classify a payload as PDF, HTML, or unknown based on heuristics.

Args:
head_bytes: Leading bytes from the HTTP payload.
content_type: Content-Type header reported by the server.
url: Source URL of the payload.

Returns:
Classification string ``"pdf"`` or ``"html"`` when detection succeeds,
otherwise ``None``.

Raises:
UnicodeDecodeError: If heuristics attempt to decode malformed byte
sequences while inspecting the payload prefix.

### `_extract_filename_from_disposition(disposition)`

Return the filename component from a Content-Disposition header.

### `_infer_suffix(url, content_type, disposition, classification, default_suffix)`

Infer a destination suffix from HTTP hints and classification heuristics.

Args:
url: Candidate download URL emitted by a resolver.
content_type: Content-Type header returned by the response (if any).
disposition: Raw Content-Disposition header for RFC 6266 parsing.
classification: Downloader classification such as ``"pdf"`` or ``"html"``.
default_suffix: Fallback extension to use when no signals are present.

Returns:
Lowercase file suffix (including leading dot) chosen from the strongest
available signal. Preference order is:

1. ``filename*`` / ``filename`` parameters in Content-Disposition.
2. Content-Type heuristics (PDF/HTML).
3. URL path suffix derived from :func:`urllib.parse.urlsplit`.
4. Provided ``default_suffix``.

### `_update_tail_buffer(buffer, chunk)`

Maintain the trailing ``limit`` bytes of a streamed download.

### `_build_download_outcome()`

Create a :class:`DownloadOutcome` applying PDF validation rules.

The helper normalises classification labels, performs the terminal ``%%EOF``
check for PDFs (skipping when running in ``--dry-run`` mode), and attaches
bookkeeping metadata such as digests and conditional request headers.

Args:
artifact: Work metadata describing the current OpenAlex record.
classification: Initial classification derived from content sniffing.
dest_path: Final storage path for the artefact (if any).
response: HTTP response object returned by :func:`request_with_retries`.
elapsed_ms: Download duration in milliseconds.
flagged_unknown: Whether heuristics flagged the payload as ambiguous.
sha256: SHA-256 digest of the payload when computed.
content_length: Size of the payload in bytes, if known.
etag: ETag header value supplied by the origin.
last_modified: Last-Modified header value supplied by the origin.
extracted_text_path: Optional path to extracted HTML text artefacts.
tail_bytes: Trailing bytes captured from the streamed download for
corruption detection heuristics.
dry_run: Indicates whether this execution runs in dry-run mode.

Returns:
DownloadOutcome capturing the normalized classification and metadata.

### `_normalize_pmid(pmid)`

Extract the numeric PubMed identifier or return ``None`` when absent.

Args:
pmid: Raw PubMed identifier string which may include prefixes.

Returns:
Normalised numeric PMCID string or ``None`` when not parsable.

### `_normalize_arxiv(arxiv_id)`

Normalize arXiv identifiers by removing prefixes and whitespace.

Args:
arxiv_id: Raw arXiv identifier which may include URL or prefix.

Returns:
Canonical arXiv identifier without prefixes or whitespace.

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

### `download_candidate(session, artifact, url, referer, timeout, context)`

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

### `apply_config_overrides(config, data, resolver_names)`

Apply overrides from configuration data onto a ResolverConfig.

Args:
config: Resolver configuration object to mutate.
data: Mapping loaded from a configuration file.
resolver_names: Known resolver names to seed toggle defaults.

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
extract_html_text: Whether to extract plaintext from HTML artefacts.
previous_lookup: Mapping of work_id/URL to prior manifest entries.
resume_completed: Set of work IDs already processed in resume mode.

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

### `_write(self, payload)`

Append a JSON record to the log file ensuring timestamps are present.

Args:
payload: JSON-serializable mapping to write.

Returns:
None

### `log_attempt(self, record)`

Record a resolver attempt entry.

Args:
record: Attempt metadata captured from the resolver pipeline.
timestamp: Optional override timestamp (ISO format).

Returns:
None

### `log(self, record)`

Compatibility shim mapping to :meth:`log_attempt`.

Args:
record: Attempt record to forward to :meth:`log_attempt`.

Returns:
None

### `log_manifest(self, entry)`

Persist a manifest entry to the JSONL log.

Args:
entry: Manifest entry to write.

Returns:
None

### `log_summary(self, summary)`

Write a summary record to the log.

Args:
summary: Mapping containing summary metrics.

Returns:
None

### `close(self)`

Close the underlying file handle.

Args:
self: Logger instance managing the JSONL file descriptor.

Returns:
None

### `__enter__(self)`

Return ``self`` when used as a context manager.

### `__exit__(self, exc_type, exc, tb)`

Close the file handle on context manager exit.

### `log_attempt(self, record)`

Write an attempt record to both JSONL and CSV outputs.

Args:
record: Attempt record to persist.

Returns:
None

### `log_manifest(self, entry)`

Forward manifest entries to the JSONL logger.

Args:
entry: Manifest entry to forward.

Returns:
None

### `log_summary(self, summary)`

Forward summary entries to the JSONL logger.

Args:
summary: Summary mapping to forward.

Returns:
None

### `log(self, record)`

Compatibility shim mapping to :meth:`log_attempt`.

Args:
record: Attempt record to log.

Returns:
None

### `close(self)`

Close both the JSONL logger and the CSV file handle.

Args:
self: Adapter instance coordinating CSV and JSONL streams.

Returns:
None

### `__enter__(self)`

Return ``self`` when used as a context manager.

### `__exit__(self, exc_type, exc, tb)`

Close the CSV file handle on context manager exit.

### `__post_init__(self)`

Define namespace mappings for output artefact directories.

Args:
self: Instance whose namespace mapping is being initialised.

Returns:
None

### `_append_location(loc)`

Accumulate location URLs from a single OpenAlex location record.

Args:
loc: Location dictionary as returned by OpenAlex (may be None).

### `_session_factory()`

Build a fresh requests session configured with polite headers.

### `_session_factory()`

Build a fresh requests session configured with polite headers.

### `_record_result(res)`

Update aggregate counters based on a single work result.

### `_submit_work(work_item)`

Submit a work item to the executor for asynchronous processing.

### `_runner()`

Process a single work item within a worker-managed session.

## 2. Classes

### `ManifestEntry`

Structured record capturing the outcome of a resolver attempt.

Attributes:
timestamp: ISO timestamp when the manifest entry was created.
work_id: OpenAlex work identifier associated with the download.
title: Human-readable work title.
publication_year: Publication year when available.
resolver: Name of the resolver that produced the asset.
url: Source URL of the downloaded artifact.
path: Local filesystem path to the stored artifact.
classification: Classification label describing the outcome (e.g., 'pdf').
content_type: MIME type reported by the server.
reason: Failure or status reason for non-successful attempts.
html_paths: Paths to any captured HTML artifacts.
sha256: SHA-256 digest of the downloaded content.
content_length: Size of the artifact in bytes.
etag: HTTP ETag header value if provided.
last_modified: HTTP Last-Modified timestamp.
extracted_text_path: Optional path to extracted text content.
dry_run: Flag indicating whether the download was simulated.

Examples:
>>> ManifestEntry(
...     timestamp="2024-01-01T00:00:00Z",
...     work_id="W123",
...     title="Sample Work",
...     publication_year=2024,
...     resolver="unpaywall",
...     url="https://example.org/sample.pdf",
...     path="pdfs/sample.pdf",
...     classification="pdf",
...     content_type="application/pdf",
...     reason=None,
...     dry_run=False,
... )

### `JsonlLogger`

Structured logger that emits attempt, manifest, and summary JSONL records.

Attributes:
_path: Destination JSONL log path.
_file: Underlying file handle used for writes.

Examples:
>>> logger = JsonlLogger(Path("logs/attempts.jsonl"))
>>> logger.log_summary({"processed": 10})
>>> logger.close()

The logger serialises records outside a thread lock and performs atomic
writes under the lock, ensuring well-formed output even when multiple
threads share the instance. It also implements the context manager protocol
for deterministic resource cleanup.

### `CsvAttemptLoggerAdapter`

Adapter that mirrors attempt records to CSV for backward compatibility.

Attributes:
_logger: Underlying :class:`JsonlLogger` instance.
_file: CSV file handle used for writing.
_writer: ``csv.DictWriter`` writing to :attr:`_file`.

Examples:
>>> adapter = CsvAttemptLoggerAdapter(JsonlLogger(Path("attempts.jsonl")), Path("attempts.csv"))
>>> adapter.log_attempt(AttemptRecord(work_id="W1", resolver_name="unpaywall", resolver_order=1,
...                                   url="https://example", status="pdf", http_status=200,
...                                   content_type="application/pdf", elapsed_ms=120.0))
>>> adapter.close()

CSV writes are protected by a lock to ensure rows remain well formed when
multiple worker threads log through the same adapter instance.

### `WorkArtifact`

Normalized artifact describing an OpenAlex work to process.

Attributes:
work_id: OpenAlex work identifier.
title: Work title suitable for logging.
publication_year: Publication year or None.
doi: Canonical DOI string.
pmid: PubMed identifier (normalized).
pmcid: PubMed Central identifier (normalized).
arxiv_id: Normalized arXiv identifier.
landing_urls: Candidate landing page URLs.
pdf_urls: Candidate PDF download URLs.
open_access_url: Open access URL provided by OpenAlex.
source_display_names: Source names for provenance.
base_stem: Base filename stem for local artefacts.
pdf_dir: Directory where PDFs are stored.
html_dir: Directory where HTML assets are stored.
failed_pdf_urls: URLs that failed during resolution.
metadata: Arbitrary metadata collected during processing.

Examples:
>>> artifact = WorkArtifact(
...     work_id="W123",
...     title="Sample Work",
...     publication_year=2024,
...     doi="10.1234/example",
...     pmid=None,
...     pmcid=None,
...     arxiv_id=None,
...     landing_urls=["https://example.org"],
...     pdf_urls=[],
...     open_access_url=None,
...     source_display_names=["Example Source"],
...     base_stem="2024__Sample_Work__W123",
...     pdf_dir=Path("pdfs"),
...     html_dir=Path("html"),
... )

### `DownloadState`

State machine for streaming downloads.

Attributes:
PENDING: Payload type is being sniffed.
WRITING: Payload bytes are being streamed to disk.

Examples:
>>> DownloadState.PENDING is DownloadState.WRITING
False
