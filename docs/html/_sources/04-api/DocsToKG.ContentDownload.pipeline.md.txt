# 1. Module: pipeline

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.pipeline``.

## 1. Overview

Content Download Resolver Orchestration

This module centralises resolver configuration, provider registration,
pipeline orchestration, and cache helpers for the DocsToKG content download
stack. Resolver classes encapsulate provider-specific discovery logic while
the shared pipeline coordinates rate limiting, concurrency, and polite HTTP
behaviour.

Key Features:
- Resolver registry supplying default provider ordering and toggles.
- Shared retry helper integration to ensure consistent network backoff.
- Manifest and attempt bookkeeping for detailed diagnostics.
- Utility functions for cache invalidation and signature normalisation.

Dependencies:
- requests: Outbound HTTP traffic and session management.
- DocsToKG.ContentDownload.resolvers: Resolver registry and implementations.
- DocsToKG.ContentDownload.network: Shared retry and session helpers.

Usage:
    from DocsToKG.ContentDownload import pipeline

    config = pipeline.ResolverConfig()
    active_resolvers = pipeline.default_resolvers()
    runner = pipeline.ResolverPipeline(
        resolvers=active_resolvers,
        config=config,
    )

## 2. Functions

### `_normalise_domain_bytes_budget(data)`

Normalize domain byte budgets to lower-case host keys and integer byte limits.

### `_normalise_domain_content_rules(data)`

Normalize domain content policies to lower-case host keys and canonical values.

### `read_resolver_config(path)`

Read resolver configuration from JSON or YAML files.

### `_seed_resolver_toggle_defaults(config, resolver_names)`

Ensure resolver toggles include defaults for every known resolver.

### `apply_config_overrides(config, data, resolver_names)`

Apply overrides from configuration data onto a ResolverConfig.

### `load_resolver_config(args, resolver_names, resolver_order_override)`

Construct resolver configuration combining CLI, config files, and env vars.

### `_callable_accepts_argument(func, name)`

Return ``True`` when ``func`` accepts an argument named ``name``.

Args:
func: Download function whose call signature should be inspected.
name: Argument name whose presence should be detected.

Returns:
bool: ``True`` when ``func`` accepts the argument or variable parameters.

### `get_timeout(self, resolver_name)`

Return the timeout to use for a resolver, falling back to defaults.

Args:
resolver_name: Name of the resolver requesting a timeout.

Returns:
float: Timeout value in seconds.

### `get_content_policy(self, host)`

Return the normalized content policy for ``host`` when configured.

### `is_enabled(self, resolver_name)`

Return ``True`` when the resolver is enabled for the current run.

Args:
resolver_name: Name of the resolver.

Returns:
bool: ``True`` if the resolver is enabled.

### `__post_init__(self)`

Validate configuration fields and apply defaults for missing values.

Args:
self: Configuration instance requiring validation.

Returns:
None

### `__post_init__(self)`

*No documentation available.*

### `is_pdf(self)`

Return ``True`` when the classification represents a PDF.

Args:
self: Download outcome to evaluate.

Returns:
bool: ``True`` if the outcome corresponds to a PDF download.

### `__post_init__(self)`

*No documentation available.*

### `__post_init__(self)`

*No documentation available.*

### `from_wire(cls, value)`

Coerce serialized event values into :class:`ResolverEvent` members.

### `from_wire(cls, value)`

Coerce serialized reason values into :class:`ResolverEventReason` members.

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
session: HTTP session used for outbound _requests.
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

### `record_failure(self, resolver_name)`

Record a resolver failure occurrence.

Args:
resolver_name: Resolver that raised an exception during execution.

Returns:
None

### `summary(self)`

Return aggregated metrics summarizing resolver behaviour.

Args:
self: Metrics collector instance aggregating resolver statistics.

Returns:
Dict[str, Any]: Snapshot of attempts, successes, HTML hits, and skips.

Examples:
>>> metrics = ResolverMetrics()
>>> metrics.record_attempt("unpaywall", DownloadOutcome("pdf", None, 200, None, 10.0))
>>> metrics.summary()["attempts"]["unpaywall"]
1

### `_emit_attempt(self, record)`

Invoke the configured logger using the structured attempt contract.

Args:
record: Attempt payload emitted by the pipeline.
timestamp: Optional ISO timestamp forwarded to sinks that accept it.

Returns:
None

### `_record_skip(self, resolver_name, reason, detail)`

Normalise and record skip telemetry for resolver events.

### `_respect_rate_limit(self, resolver_name)`

Sleep as required to respect per-resolver rate limiting policies.

The method performs an atomic read-modify-write on
:attr:`_last_invocation` guarded by :attr:`_lock` to ensure that
concurrent threads honour resolver spacing requirements.

Args:
resolver_name: Name of the resolver to rate limit.

Returns:
None

### `_respect_domain_limit(self, url)`

Enforce per-domain throttling when configured.

Args:
url: Candidate URL whose host may be throttled.

Returns:
Total seconds slept enforcing the domain policies.

### `_ensure_host_bucket(self, host)`

*No documentation available.*

### `_resolve_budget_host_key(self, host)`

*No documentation available.*

### `_domain_budget_remaining(self, host_key)`

*No documentation available.*

### `_estimate_outcome_bytes(self, outcome)`

*No documentation available.*

### `_record_domain_bytes(self, host_key, outcome)`

*No documentation available.*

### `_acquire_host_slot(self, host)`

*No documentation available.*

### `_get_existing_host_breaker(self, host)`

*No documentation available.*

### `_ensure_host_breaker(self, host)`

*No documentation available.*

### `_host_breaker_allows(self, host)`

*No documentation available.*

### `_update_breakers(self, resolver_name, host, outcome)`

*No documentation available.*

### `_jitter_sleep(self)`

Introduce a small delay to avoid stampeding downstream services.

Args:
self: Pipeline instance executing resolver scheduling logic.

Returns:
None

### `_should_attempt_head_check(self, resolver_name, url)`

Return ``True`` when a resolver should perform a HEAD preflight request.

Args:
resolver_name: Name of the resolver under consideration.

Returns:
Boolean indicating whether the resolver should issue a HEAD request.

### `_head_precheck_url(self, session, url, timeout, content_policy)`

Delegate to the shared network-layer preflight helper.

Args:
session: Requests session used to issue the HEAD preflight.
url: Candidate URL that may host a downloadable document.
timeout: Maximum duration in seconds to wait for the HEAD response.

Returns:
bool: ``True`` when the URL passes preflight checks, ``False`` otherwise.

### `run(self, session, artifact, context)`

Execute resolvers until a PDF is obtained or resolvers are exhausted.

Args:
session: Requests session used for resolver HTTP calls.
artifact: Work artifact describing the document to resolve.
context: Optional :class:`DownloadContext` (or mapping) containing execution flags.
session_factory: Optional callable returning a thread-local session for
resolver execution. When omitted, the provided ``session`` is reused.

Returns:
PipelineResult capturing resolver attempts and successful downloads.

### `_run_sequential(self, session, session_provider, artifact, context, state)`

Execute resolvers in order using the current thread.

Args:
session: Shared requests session for resolver HTTP calls.
session_provider: Callable returning the session for the active thread.
artifact: Work artifact describing the document being processed.
context: Execution context object.
state: Mutable run state tracking attempts and duplicates.

Returns:
PipelineResult summarising the sequential run outcome.

### `_run_concurrent(self, session, session_provider, artifact, context, state)`

Execute resolvers concurrently using a thread pool.

Args:
session: Shared requests session for resolver HTTP calls.
session_provider: Callable returning the session for the active thread.
artifact: Work artifact describing the document being processed.
context: Execution context object.
state: Mutable run state tracking attempts and duplicates.

Returns:
PipelineResult summarising the concurrent run outcome.

### `_prepare_resolver(self, resolver_name, order_index, artifact, state)`

Return a prepared resolver or log skip events when unavailable.

Args:
resolver_name: Name of the resolver to prepare.
order_index: Execution order index for the resolver.
artifact: Work artifact being processed.
state: Mutable run state tracking skips and duplicates.

Returns:
Resolver instance when available and enabled, otherwise ``None``.

### `_collect_resolver_results(self, resolver_name, resolver, session_provider, artifact)`

Collect resolver results while applying rate limits and error handling.

Args:
resolver_name: Name of the resolver being executed (for logging and limits).
resolver: Resolver instance that will generate candidate URLs.
session_provider: Callable returning the requests session for the current thread.
artifact: Work artifact describing the current document.

Returns:
Tuple of resolver results and the resolver wall time (ms).

### `_process_result(self, session, artifact, resolver_name, order_index, result, context, state)`

Process a single resolver result and return a terminal pipeline outcome.

Args:
session: Requests session used for follow-up download calls.
artifact: Work artifact describing the document being processed.
resolver_name: Name of the resolver that produced the result.
order_index: 1-based index of the resolver in the execution order.
result: Resolver result containing either a URL or event metadata.
context: Execution context object.
state: Mutable run state tracking attempts and duplicates.
resolver_wall_time_ms: Wall-clock time spent in the resolver.

Returns:
PipelineResult when resolution succeeds, otherwise ``None``.

### `_percentile(sorted_values, percentile)`

*No documentation available.*

### `_release()`

*No documentation available.*

### `submit_next(executor, start_index)`

Queue additional resolvers until reaching concurrency limits.

Args:
executor: Thread pool responsible for executing resolver calls.
start_index: Index in ``resolver_order`` where submission should resume.

Returns:
Updated index pointing to the next resolver candidate that has not been submitted.

### `session_provider()`

Return the shared HTTP session when no factory override is provided.

## 3. Classes

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
resolver_min_interval_s: Minimum interval between resolver _requests.
domain_min_interval_s: Optional per-domain rate limits overriding resolver settings.
enable_head_precheck: Toggle applying HEAD filtering before downloads.
resolver_head_precheck: Per-resolver overrides for HEAD filtering behaviour.
host_accept_overrides: Mapping of hostname to Accept header override.
mailto: Contact email appended to polite headers and user agent string.
max_concurrent_resolvers: Upper bound on concurrent resolver threads per work.
max_concurrent_per_host: Upper bound on simultaneous downloads per hostname.
enable_global_url_dedup: Enable global URL deduplication across works when True.
domain_token_buckets: Mapping of hostname to token bucket parameters.
domain_bytes_budget: Mapping of hostname to maximum bytes permitted this run.
domain_content_rules: Mapping of hostname to MIME allow-lists and max-bytes caps.
resolver_circuit_breakers: Mapping of resolver name to breaker thresholds/cooldowns.

Notes:
``enable_head_precheck`` toggles inexpensive HEAD lookups before downloads
to filter obvious HTML responses. ``resolver_head_precheck`` allows
per-resolver overrides when specific providers reject HEAD _requests.
``max_concurrent_resolvers`` bounds the number of resolver threads used
per work while still respecting configured rate limits. ``max_concurrent_per_host``
limits simultaneous downloads hitting the same hostname across workers.

Examples:
>>> config = ResolverConfig()
>>> config.max_attempts_per_work
25

### `AttemptRecord`

Structured log record describing a resolver attempt.

Attributes:
run_id: Identifier shared across a downloader run.
work_id: Identifier of the work being processed.
resolver_name: Name of the resolver that produced the record.
resolver_order: Ordinal position of the resolver in the pipeline.
url: Candidate URL that was attempted.
status: :class:`Classification` describing the attempt result.
http_status: HTTP status code (when available).
content_type: Response content type.
elapsed_ms: Approximate elapsed time for the attempt in milliseconds.
resolver_wall_time_ms: Wall-clock time spent inside the resolver including
rate limiting, measured in milliseconds.
reason: Optional :class:`ReasonCode` describing failure or skip reason.
metadata: Arbitrary metadata supplied by the resolver.
sha256: SHA-256 digest of downloaded content, when available.
content_length: Size of the downloaded content in bytes.
dry_run: Flag indicating whether the attempt occurred in dry-run mode.
retry_after: Optional cooldown seconds suggested by the upstream service.

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

### `DownloadOutcome`

Outcome of a resolver download attempt.

Attributes:
classification: Classification label describing the outcome (e.g., 'pdf').
path: Local filesystem path to the stored artifact.
http_status: HTTP status code when available.
content_type: Content type reported by the server.
elapsed_ms: Time spent downloading in milliseconds.
reason: Optional :class:`ReasonCode` describing failures.
reason_detail: Optional human-readable diagnostic detail.
sha256: SHA-256 digest of the downloaded content.
content_length: Size of the downloaded content in bytes.
etag: HTTP ETag header value when provided.
last_modified: HTTP Last-Modified timestamp.
extracted_text_path: Optional path to extracted text artefacts.
retry_after: Optional cooldown value derived from HTTP ``Retry-After`` headers.

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
failed_urls: Candidate URLs that failed during this run.
reason: Optional reason string explaining failures.

Examples:
>>> PipelineResult(success=True, resolver_name="unpaywall", url="https://example")

### `ResolverEvent`

Canonical resolver event categories emitted via :class:`ResolverResult`.

### `ResolverEventReason`

Structured reason taxonomy for resolver events.

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

### `_RunState`

Mutable pipeline execution state shared across resolvers.

Args:
dry_run: Indicates whether downloads should be skipped.

Attributes:
dry_run: Indicates whether downloads should be skipped.
seen_urls: Set of URLs already attempted.
html_paths: Collected HTML fallback paths.
failed_urls: URLs that failed during resolution.
attempt_counter: Total number of resolver attempts performed.

Examples:
>>> state = _RunState(dry_run=True)
>>> state.dry_run
True

### `ResolverPipeline`

Executes resolvers in priority order until a PDF download succeeds.

The pipeline is safe to reuse across worker threads when
:attr:`ResolverConfig.max_concurrent_resolvers` is greater than one. All
mutable shared state is protected by :class:`threading.Lock` instances and
only read concurrently without mutation. Resolver execution retrieves
thread-local HTTP sessions via the supplied factory so each worker maintains
its own connection pool without cross-thread sharing.

Attributes:
config: Resolver configuration containing ordering and rate limits.
download_func: Callable responsible for downloading resolved URLs.
logger: Structured attempt logger capturing resolver telemetry.
metrics: Metrics collector tracking resolver performance.

Examples:
>>> pipeline = ResolverPipeline([], ResolverConfig(), lambda *args, **kwargs: None, None)  # doctest: +SKIP
>>> isinstance(pipeline.metrics, ResolverMetrics)  # doctest: +SKIP
True
