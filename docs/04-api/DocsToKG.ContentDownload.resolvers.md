# 1. Module: resolvers

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.resolvers``.

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
- BeautifulSoup: Optional HTML parsing for resolver implementations.
- DocsToKG.ContentDownload.network: Shared retry and session helpers.

Usage:
    from DocsToKG.ContentDownload import resolvers

    config = resolvers.ResolverConfig()
    active_resolvers = resolvers.default_resolvers()
    pipeline = resolvers.ResolverPipeline(
        resolvers=active_resolvers,
        config=config,
    )

## 1. Functions

### `headers_cache_key(headers)`

Return a deterministic cache key for HTTP header dictionaries.

Args:
headers: Mapping of header names to values.

Returns:
Tuple of lowercase header names paired with their original values,
sorted alphabetically for stable hashing.

### `request_with_retries(session, method, url)`

Proxy to :func:`DocsToKG.ContentDownload.network.request_with_retries`.

Args:
session: Requests session used to perform the HTTP call.
method: HTTP method such as ``"GET"`` or ``"HEAD"``.
url: Fully-qualified URL for the request.
**kwargs: Additional parameters forwarded to the network layer helper.

Returns:
requests.Response: Response returned by the shared network helper.

Notes:
The indirection keeps resolver providers compatible with tests that patch the
network-layer helper while avoiding circular imports during module initialisation.

### `_absolute_url(base, href)`

Resolve relative ``href`` values against ``base`` to obtain absolute URLs.

### `_collect_candidate_urls(node, results)`

Recursively collect HTTP(S) URLs from nested response payloads.

### `_fetch_crossref_data(doi, mailto, timeout, headers_key)`

Retrieve Crossref metadata for ``doi`` with polite header caching.

### `_fetch_unpaywall_data(doi, email, timeout, headers_key)`

Fetch Unpaywall metadata for ``doi`` using polite caching.

### `_fetch_semantic_scholar_data(doi, api_key, timeout, headers_key)`

Fetch Semantic Scholar Graph API metadata for ``doi`` with caching.

### `default_resolvers()`

Instantiate the default resolver stack in priority order.

Args:
None

Returns:
List[Resolver]: Resolver instances ordered according to
``DEFAULT_RESOLVER_ORDER``.

Examples:
>>> from DocsToKG.ContentDownload import resolvers
>>> [resolver.name for resolver in resolvers.default_resolvers()]  # doctest: +ELLIPSIS
['openalex', 'unpaywall', ...]

### `_callable_accepts_argument(func, name)`

Return ``True`` when ``func`` accepts an argument named ``name``.

Args:
func: Download function whose call signature should be inspected.
name: Argument name whose presence should be detected.

Returns:
bool: ``True`` when ``func`` accepts the argument or variable parameters.

### `clear_resolver_caches()`

Clear resolver-level HTTP caches to force fresh lookups.

This utility resets the internal LRU caches used by the Unpaywall,
Crossref, and Semantic Scholar resolvers. It should be called before
executing resolver pipelines when deterministic behaviour across runs is
required (for example, in unit tests or benchmarking scenarios).

Args:
None

Returns:
None

### `__getattr__(name)`

Return legacy exports while emitting :class:`DeprecationWarning`.

### `is_event(self)`

Return ``True`` when this result represents an informational event.

Args:
self: Resolver result instance under inspection.

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

Validate configuration fields and apply defaults for missing values.

Args:
self: Configuration instance requiring validation.

Returns:
None

### `log(self, record)`

Log a resolver attempt.

Args:
record: Attempt record describing the resolver execution.

Returns:
None

### `is_pdf(self)`

Return ``True`` when the classification represents a PDF.

Args:
self: Download outcome to evaluate.

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

### `register(cls, resolver_cls)`

Register a resolver class under its declared ``name`` attribute.

Args:
resolver_cls: Resolver implementation to register.

Returns:
Type[Resolver]: The registered resolver class for chaining.

### `create_default(cls)`

Instantiate resolver instances in priority order.

Args:
None

Returns:
List[Resolver]: Resolver instances ordered by default priority.

Raises:
None.

### `__init_subclass__(cls, register)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `_lookup_pmcids(self, session, identifiers, config)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

### `is_enabled(self, config, artifact)`

Return ``True`` when resolver prerequisites are met for the artifact.

Args:
config: Resolver configuration containing runtime toggles and credentials.
artifact: Work artifact capturing document metadata and identifiers.

Returns:
bool: ``True`` when the resolver should attempt to resolve the artifact.

### `iter_urls(self, session, config, artifact)`

Yield resolver results discovered for the supplied artifact.

Args:
session: Requests session used to communicate with upstream providers.
config: Resolver configuration supplying timeouts and headers.
artifact: Work artifact describing the document under resolution.

Returns:
Iterator[ResolverResult]: Stream of candidate download URLs or resolver events.

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
url: Resolver URL whose host may be subject to throttling.

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
resolver_min_interval_s: Minimum interval between resolver _requests.
domain_min_interval_s: Optional per-domain rate limits overriding resolver settings.
resolver_rate_limits: Deprecated rate limit configuration retained for compat.
enable_head_precheck: Toggle applying HEAD filtering before downloads.
resolver_head_precheck: Per-resolver overrides for HEAD filtering behaviour.
mailto: Contact email appended to polite headers and user agent string.
max_concurrent_resolvers: Upper bound on concurrent resolver threads per work.
enable_global_url_dedup: Enable global URL deduplication across works when True.

Notes:
``enable_head_precheck`` toggles inexpensive HEAD lookups before downloads
to filter obvious HTML responses. ``resolver_head_precheck`` allows
per-resolver overrides when specific providers reject HEAD _requests.
``max_concurrent_resolvers`` bounds the number of resolver threads used
per work while still respecting configured rate limits.

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
resolver_wall_time_ms: Wall-clock time spent inside the resolver including
rate limiting, measured in milliseconds.
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

Attributes:
None: The protocol formalises the callable surface without storing state.

Examples:
>>> class Collector:
...     def __init__(self):
...         self.records = []
...     def log(self, record: AttemptRecord) -> None:
...         self.records.append(record)
>>> collector = Collector()
>>> isinstance(collector, AttemptLogger)
True

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
failed_urls: Candidate URLs that failed during this run.
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

### `ResolverRegistry`

Registry tracking resolver classes by their ``name`` attribute.

Attributes:
_providers: Mapping of resolver names to resolver classes.

Examples:
>>> ResolverRegistry.register(type("Tmp", (RegisteredResolver,), {"name": "tmp"}))  # doctest: +SKIP
<class 'Tmp'>

### `RegisteredResolver`

Mixin ensuring subclasses register with :class:`ResolverRegistry`.

Attributes:
None: Subclasses inherit registration behaviour automatically.

Examples:
>>> class ExampleResolver(RegisteredResolver):
...     name = "example"
...     def is_enabled(self, config, artifact):
...         return True
...     def iter_urls(self, session, config, artifact):
...         yield ResolverResult(url="https://example.org")

### `ArxivResolver`

Resolve arXiv preprints using arXiv identifier lookups.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> ArxivResolver().name
'arxiv'

### `CoreResolver`

Resolve PDFs using the CORE API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> CoreResolver().name
'core'

### `CrossrefResolver`

Resolve candidate URLs from the Crossref metadata API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> CrossrefResolver().name
'crossref'

### `DoajResolver`

Resolve Open Access links using the DOAJ API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> DoajResolver().name
'doaj'

### `EuropePmcResolver`

Resolve Open Access links via the Europe PMC REST API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> EuropePmcResolver().name
'europe_pmc'

### `FigshareResolver`

Resolve Figshare repository metadata into download URLs.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> FigshareResolver().name
'figshare'

### `HalResolver`

Resolve publications from the HAL open archive.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> HalResolver().name
'hal'

### `LandingPageResolver`

Attempt to scrape landing pages when explicit PDFs are unavailable.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> LandingPageResolver().name
'landing_page'

### `OpenAireResolver`

Resolve URLs using the OpenAIRE API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> OpenAireResolver().name
'openaire'

### `OpenAlexResolver`

Resolve OpenAlex work metadata into candidate download URLs.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> OpenAlexResolver().name
'openalex'

### `OsfResolver`

Resolve artefacts hosted on the Open Science Framework.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> OsfResolver().name
'osf'

### `PmcResolver`

Resolve PubMed Central articles via identifiers and lookups.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> PmcResolver().name
'pmc'

### `SemanticScholarResolver`

Resolve PDFs via the Semantic Scholar Graph API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> SemanticScholarResolver().name
'semantic_scholar'

### `UnpaywallResolver`

Resolve PDFs via the Unpaywall API.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> UnpaywallResolver().name
'unpaywall'

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> WaybackResolver().name
'wayback'

### `ZenodoResolver`

Resolve Zenodo records into downloadable open-access PDF URLs.

Attributes:
name: Resolver identifier registered with the pipeline.

Examples:
>>> ZenodoResolver().name
'zenodo'

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
only read concurrently without mutation. HTTP ``_requests.Session`` objects
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
