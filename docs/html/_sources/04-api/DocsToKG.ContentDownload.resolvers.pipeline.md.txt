# 1. Module: pipeline

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers.pipeline``.

Resolver Pipeline Execution Engine

This module coordinates the execution of resolver providers that discover
downloadable artefacts for scholarly works. It encapsulates sequential and
concurrent strategies, rate limiting, duplicate detection, and callback hooks
for logging and metrics collection.

Key Features:
- Sequential and concurrent resolver scheduling with configurable concurrency.
- Rate-limiting enforcement and optional HEAD preflight checks.
- State tracking for seen URLs, HTML fallbacks, and failure metrics.

Usage:
    from DocsToKG.ContentDownload.resolvers.pipeline import ResolverPipeline

    pipeline = ResolverPipeline(
        resolvers=[],
        config=ResolverConfig(),
        download_func=lambda *args, **kwargs: DownloadOutcome("miss"),
        logger=lambda record: None,
        metrics=ResolverMetrics(),
    )

## 1. Functions

### `request_with_retries(session, method, url)`

Proxy to :func:`DocsToKG.ContentDownload.http.request_with_retries`.

The indirection keeps this module compatible with unit tests that monkeypatch
either the pipeline-level attribute or the underlying HTTP helper while also
deferring imports to avoid circular dependencies during runtime initialisation.

Args:
session: Requests session used to execute the outbound HTTP call.
method: HTTP verb such as ``"GET"`` or ``"HEAD"``.
url: Fully qualified URL to fetch.
**kwargs: Additional keyword arguments forwarded to the HTTP helper.

Returns:
requests.Response: Response object produced by the proxied helper.

Raises:
requests.RequestException: Propagated from the underlying retry helper.

### `_callable_accepts_argument(func, name)`

Return ``True`` when ``func`` accepts an argument named ``name``.

Args:
func: Download function whose call signature should be inspected.
name: Argument name whose presence should be detected.

Returns:
bool: ``True`` when ``func`` accepts the argument or variable parameters.

### `_respect_rate_limit(self, resolver_name)`

Sleep as required to respect per-resolver rate limiting policies.

The method performs an atomic read-modify-write on
:attr:`_last_invocation` guarded by :attr:`_lock` to ensure that
concurrent threads honour resolver spacing requirements.

Args:
resolver_name: Name of the resolver to rate limit.

Returns:
None

### `_jitter_sleep(self)`

Introduce a small delay to avoid stampeding downstream services.

Args:
self: Pipeline instance executing resolver scheduling logic.

Returns:
None

### `_should_attempt_head_check(self, resolver_name)`

Return ``True`` when a resolver should perform a HEAD preflight request.

Args:
resolver_name: Name of the resolver under consideration.

Returns:
Boolean indicating whether the resolver should issue a HEAD request.

### `_head_precheck_url(self, session, url, timeout)`

Issue a HEAD request to validate that ``url`` plausibly returns a PDF.

Args:
session: Requests session used for issuing the HEAD request.
url: Candidate URL whose response should be inspected.
timeout: Timeout budget for the preflight request.

Returns:
``True`` when the response appears to represent a PDF download.

### `run(self, session, artifact, context)`

Execute resolvers until a PDF is obtained or resolvers are exhausted.

Args:
session: Requests session used for resolver HTTP calls.
artifact: Work artifact describing the document to resolve.
context: Optional execution context containing flags such as ``dry_run``.

Returns:
PipelineResult capturing resolver attempts and successful downloads.

### `_run_sequential(self, session, artifact, context_data, state)`

Execute resolvers in order using the current thread.

Args:
session: Shared requests session for resolver HTTP calls.
artifact: Work artifact describing the document being processed.
context_data: Execution context dictionary.
state: Mutable run state tracking attempts and duplicates.

Returns:
PipelineResult summarising the sequential run outcome.

### `_run_concurrent(self, session, artifact, context_data, state)`

Execute resolvers concurrently using a thread pool.

Args:
session: Shared requests session for resolver HTTP calls.
artifact: Work artifact describing the document being processed.
context_data: Execution context dictionary.
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

### `_collect_resolver_results(self, resolver_name, resolver, session, artifact)`

Collect resolver results while applying rate limits and error handling.

Args:
resolver_name: Name of the resolver being executed (for logging and limits).
resolver: Resolver instance that will generate candidate URLs.
session: Requests session forwarded to the resolver.
artifact: Work artifact describing the current document.

Returns:
Tuple of resolver results and the resolver wall time (ms).

### `_process_result(self, session, artifact, resolver_name, order_index, result, context_data, state)`

Process a single resolver result and return a terminal pipeline outcome.

Args:
session: Requests session used for follow-up download calls.
artifact: Work artifact describing the document being processed.
resolver_name: Name of the resolver that produced the result.
order_index: 1-based index of the resolver in the execution order.
result: Resolver result containing either a URL or event metadata.
context_data: Execution context dictionary.
state: Mutable run state tracking attempts and duplicates.
resolver_wall_time_ms: Wall-clock time spent in the resolver.

Returns:
PipelineResult when resolution succeeds, otherwise ``None``.

### `submit_next(executor, start_index)`

Queue additional resolvers until reaching concurrency limits.

Args:
executor: Thread pool responsible for executing resolver calls.
start_index: Index in ``resolver_order`` where submission should resume.

Returns:
Updated index pointing to the next resolver candidate that has not been submitted.

## 2. Classes

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
only read concurrently without mutation. HTTP ``requests.Session`` objects
are treated as read-only; callers must avoid mutating shared sessions after
handing them to the pipeline.

Attributes:
config: Resolver configuration containing ordering and rate limits.
download_func: Callable responsible for downloading resolved URLs.
logger: Structured attempt logger capturing resolver telemetry.
metrics: Metrics collector tracking resolver performance.

Examples:
>>> pipeline = ResolverPipeline([], ResolverConfig(), lambda *args, **kwargs: None, None)  # doctest: +SKIP
>>> isinstance(pipeline.metrics, ResolverMetrics)  # doctest: +SKIP
True
