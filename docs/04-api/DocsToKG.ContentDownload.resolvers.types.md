# Module: types

Type definitions and protocols for the resolver pipeline.

## Functions

### `is_event(self)`

Return ``True`` when this result represents an informational event.

Args:
None

Returns:
bool: ``True`` if the resolver emitted an event instead of a URL.

### `get_timeout(self, resolver_name)`

Return the timeout to use for a resolver, falling back to defaults.

Args:
resolver_name: Name of the resolver requesting a timeout.

Returns:
float: Timeout value in seconds.

### `is_enabled(self, resolver_name)`

Return ``True`` when the resolver is enabled for the current run.

Args:
resolver_name: Name of the resolver.

Returns:
bool: ``True`` if the resolver is enabled.

### `__post_init__(self)`

*No documentation available.*

### `log(self, record)`

Log a resolver attempt.

Args:
record: Attempt record describing the resolver execution.

Returns:
None

### `is_pdf(self)`

Return ``True`` when the classification represents a PDF.

Args:
None

Returns:
bool: ``True`` if the outcome corresponds to a PDF download.

### `is_enabled(self, config, artifact)`

Return ``True`` if this resolver should run for the given artifact.

Args:
config: Resolver configuration options.
artifact: Work artifact under consideration.

Returns:
bool: ``True`` when the resolver should run for the artifact.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs or events for the given artifact.

Args:
session: HTTP session used for outbound requests.
config: Resolver configuration.
artifact: Work artifact describing the current item.

Returns:
Iterable[ResolverResult]: Stream of download candidates or events.

### `record_attempt(self, resolver_name, outcome)`

Record a resolver attempt and update success/html counters.

Args:
resolver_name: Name of the resolver that executed.
outcome: Download outcome produced by the resolver.

Returns:
None

### `record_skip(self, resolver_name, reason)`

Record a skip event for a resolver with a reason tag.

Args:
resolver_name: Resolver that was skipped.
reason: Short description explaining the skip.

Returns:
None

### `summary(self)`

Return aggregated metrics summarizing resolver behaviour.

Args:
None

Returns:
Dict[str, Any]: Snapshot of attempts, successes, HTML hits, and skips.

Examples:
>>> metrics = ResolverMetrics()
>>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
>>> metrics.summary()["attempts"]["unpaywall"]
1

## Classes

### `ResolverResult`

Either a candidate download URL or an informational resolver event.

Attributes:
url: Candidate download URL emitted by the resolver (``None`` for events).
referer: Optional referer header to accompany the download request.
metadata: Arbitrary metadata recorded alongside the result.
event: Optional event label (e.g., ``"error"`` or ``"skipped"``).
event_reason: Human-readable reason describing the event.
http_status: HTTP status associated with the event, when available.

Examples:
>>> ResolverResult(url="https://example.org/file.pdf", metadata={"resolver": "core"})

### `ResolverConfig`

Runtime configuration options applied across resolvers.

Attributes:
resolver_order: Ordered list of resolver names to execute.
resolver_toggles: Mapping toggling individual resolvers on/off.
max_attempts_per_work: Maximum number of resolver attempts per work item.
timeout: Default HTTP timeout applied to resolvers.
sleep_jitter: Random jitter added between retries.
polite_headers: HTTP headers to apply for polite crawling.
unpaywall_email: Contact email registered with Unpaywall.
core_api_key: API key used for the CORE resolver.
semantic_scholar_api_key: API key for Semantic Scholar resolver.
doaj_api_key: API key for DOAJ resolver.
resolver_timeouts: Resolver-specific timeout overrides.
resolver_min_interval_s: Minimum interval between resolver requests.
resolver_rate_limits: Deprecated rate limit configuration retained for compat.
mailto: Contact email appended to polite headers and user agent string.

Examples:
>>> config = ResolverConfig()
>>> config.max_attempts_per_work
25

### `AttemptRecord`

Structured log record describing a resolver attempt.

Attributes:
work_id: Identifier of the work being processed.
resolver_name: Name of the resolver that produced the record.
resolver_order: Ordinal position of the resolver in the pipeline.
url: Candidate URL that was attempted.
status: Classification or status string for the attempt.
http_status: HTTP status code (when available).
content_type: Response content type.
elapsed_ms: Approximate elapsed time for the attempt in milliseconds.
reason: Optional descriptive reason for failures or skips.
metadata: Arbitrary metadata supplied by the resolver.
sha256: SHA-256 digest of downloaded content, when available.
content_length: Size of the downloaded content in bytes.
dry_run: Flag indicating whether the attempt occurred in dry-run mode.

Examples:
>>> AttemptRecord(
...     work_id="W1",
...     resolver_name="unpaywall",
...     resolver_order=1,
...     url="https://example.org/pdf",
...     status="pdf",
...     http_status=200,
...     content_type="application/pdf",
...     elapsed_ms=120.5,
... )

### `AttemptLogger`

Protocol for logging resolver attempts.

Examples:
>>> class Collector:
...     def __init__(self):
...         self.records = []
...     def log(self, record: AttemptRecord) -> None:
...         self.records.append(record)
>>> collector = Collector()
>>> isinstance(collector, AttemptLogger)
True

Attributes:
None

### `DownloadOutcome`

Outcome of a resolver download attempt.

Attributes:
classification: Classification label describing the outcome (e.g., 'pdf').
path: Local filesystem path to the stored artifact.
http_status: HTTP status code when available.
content_type: Content type reported by the server.
elapsed_ms: Time spent downloading in milliseconds.
error: Optional error string describing failures.
sha256: SHA-256 digest of the downloaded content.
content_length: Size of the downloaded content in bytes.
etag: HTTP ETag header value when provided.
last_modified: HTTP Last-Modified timestamp.
extracted_text_path: Optional path to extracted text artefacts.

Examples:
>>> DownloadOutcome(classification="pdf", path="pdfs/sample.pdf", http_status=200,
...                 content_type="application/pdf", elapsed_ms=150.0)

### `PipelineResult`

Aggregate result returned by the resolver pipeline.

Attributes:
success: Indicates whether the pipeline found a suitable asset.
resolver_name: Resolver that produced the successful result.
url: URL that was ultimately fetched.
outcome: Download outcome associated with the result.
html_paths: Collected HTML artefacts from the pipeline.
reason: Optional reason string explaining failures.

Examples:
>>> PipelineResult(success=True, resolver_name="unpaywall", url="https://example")

### `Resolver`

Protocol that resolver implementations must follow.

Attributes:
name: Resolver identifier used within configuration.

Examples:
Concrete implementations are provided in classes such as
:class:`UnpaywallResolver` and :class:`CrossrefResolver` below.

### `ResolverMetrics`

Lightweight metrics collector for resolver execution.

Attributes:
attempts: Counter of attempts per resolver.
successes: Counter of successful PDF downloads per resolver.
html: Counter of HTML-only results per resolver.
skips: Counter of skip events keyed by resolver and reason.

Examples:
>>> metrics = ResolverMetrics()
>>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
>>> metrics.summary()["attempts"]["unpaywall"]
1
