# Module: download_pyalex_pdfs

Download PDFs for OpenAlex works with a configurable resolver stack.

## Functions

### `_utc_timestamp()`

Return an ISO 8601 UTC timestamp with a trailing 'Z' suffix.

### `_accepts_argument(func, name)`

*No documentation available.*

### `_has_pdf_eof(path)`

*No documentation available.*

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
path: Directory path to create.

Returns:
None

### `_make_session(headers)`

Create a retry-enabled :class:`requests.Session` configured for polite crawling.

Parameters
----------
headers:
Header dictionary returned by :func:`load_resolver_config`. The mapping **must**
already include the project user agent and mailto contact address. The factory
copies the mapping before applying it to the outgoing session so tests may pass
mutable dictionaries without worrying about side effects.

Returns
-------
requests.Session
A session with exponential backoff retry behaviour suitable for resolver
traffic. Both ``http`` and ``https`` transports share the same
:class:`urllib3.util.retry.Retry` configuration.

Notes
-----
Each worker should call this helper to obtain an isolated session instance. Example
usage::

session = _make_session({"User-Agent": "DocsToKGDownloader/1.0", "mailto": "ops@example.org"})

Subsequent resolver requests automatically include the polite headers and retry
policy.

### `_make_session_for_worker(headers)`

Factory helper for per-worker sessions.

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

Return 'pdf', 'html', or None if undecided.

Args:
head_bytes: Leading bytes from the HTTP payload.
content_type: Content-Type header reported by the server.
url: Source URL of the payload.

Returns:
Optional[str]: Classification string or ``None`` when unknown.

Raises:
UnicodeDecodeError: If payload sniffing encounters invalid encoding when
decoding the tail of the file (rare; captured by fallback logic).
requests.RequestException: Propagated if header or GET requests fail
before protective guards are applied.
OSError: If filesystem writes fail while persisting the payload.
ValueError: If payload metadata cannot be interpreted while classifying.

### `_build_download_outcome()`

Create a :class:`DownloadOutcome` applying PDF validation rules.

The helper normalises classification labels, performs the terminal ``%%EOF``
check for PDFs (skipping when running in ``--dry-run`` mode), and attaches
bookkeeping metadata such as digests and conditional request headers.

### `_normalize_pmid(pmid)`

*No documentation available.*

### `_normalize_arxiv(arxiv_id)`

*No documentation available.*

### `_collect_location_urls(work)`

*No documentation available.*

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
session: HTTP session providing ``head`` and ``get`` methods.
artifact: Work metadata and output directory handles for the current record.
url: Candidate download URL discovered by a resolver.
referer: Optional referer header override provided by the resolver.
timeout: Per-request timeout in seconds.
context: Execution context containing ``dry_run``, ``extract_html_text``,
and ``previous`` manifest lookup data.

Returns:
DownloadOutcome describing the result of the download attempt.

Raises:
requests.RequestException: If HTTP requests fail unexpectedly outside
guarded sections.
OSError: If writing the downloaded payload to disk fails.
ValueError: If response payloads cannot be interpreted during
classification.

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
Iterable yielding work payload dictionaries.

### `attempt_openalex_candidates(session, artifact, logger, metrics, context)`

Attempt downloads for all candidate URLs associated with an artifact.

Args:
session: Requests session configured for resolver usage.
artifact: Work artifact containing candidate URLs.
logger: Attempt logger receiving structured records.
metrics: Resolver metrics collector.
context: Optional context dict (dry-run flags, previous entries).

Returns:
Pair of (DownloadOutcome, URL) on success, otherwise None.

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

CLI entrypoint for the OpenAlex PDF downloader.

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
None

Returns:
None

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
None

Returns:
None

### `__post_init__(self)`

*No documentation available.*

### `_append_location(loc)`

*No documentation available.*

### `_session_factory()`

*No documentation available.*

### `_record_result(res)`

*No documentation available.*

### `_submit_work(work_item)`

*No documentation available.*

### `_runner()`

*No documentation available.*

## Classes

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
