# Module: __init__

Resolver pipeline and provider implementations for the OpenAlex downloader.

The pipeline is intentionally lightweight so it can be reused by both the
command-line entrypoint and tests.  Resolvers yield candidate URLs (and
associated metadata) which are attempted in priority order until a confirmed
PDF is downloaded.

## Functions

### `_sleep_backoff(attempt, base)`

*No documentation available.*

### `_headers_cache_key(headers)`

*No documentation available.*

### `_fetch_unpaywall_data(doi, email, timeout, headers_key)`

*No documentation available.*

### `_fetch_crossref_data(doi, mailto, timeout, headers_key)`

*No documentation available.*

### `_fetch_semantic_scholar_data(doi, api_key, timeout, headers_key)`

*No documentation available.*

### `_collect_candidate_urls(node, results)`

*No documentation available.*

### `_request_with_retries(session, method, url)`

Invoke `session.request` with exponential backoff on transient errors.

### `_callable_accepts_argument(func, name)`

*No documentation available.*

### `_absolute_url(base, href)`

*No documentation available.*

### `clear_resolver_caches()`

Clear resolver-level LRU caches to avoid stale results.

Args:
None

Returns:
None

### `default_resolvers()`

Return the default resolver instances in priority order.

Args:
None

Returns:
List[Resolver]: Resolver instances in execution order.

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

Yield candidate PMC download URLs derived from identifiers.

Args:
session: Requests session used to query PMC utilities.
config: Resolver configuration supplying headers/timeouts.
artifact: Work artifact containing PMC/PMID/DOI identifiers.

Returns:
Iterable[ResolverResult]: Candidate PMC download URLs.

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

### `_respect_rate_limit(self, resolver_name)`

*No documentation available.*

### `_jitter_sleep(self)`

*No documentation available.*

### `run(self, session, artifact, context)`

Execute resolvers sequentially until a PDF is obtained or exhausted.

Args:
session: Requests session shared across resolver invocations.
artifact: Work artifact describing the current work item.
context: Optional context dictionary (dry-run flags, previous manifest).

Returns:
PipelineResult summarizing the pipeline outcome.

### `is_enabled(self, config, artifact)`

Return ``True`` when Unpaywall is configured and the work has a DOI.

Args:
config: Resolver configuration containing Unpaywall credentials.
artifact: Work artifact whose identifiers are being considered.

Returns:
bool: ``True`` if the resolver should run for this artifact.

### `iter_urls(self, session, config, artifact)`

Yield candidate PDF URLs discovered via the Unpaywall API.

Args:
session: Requests session used to query the Unpaywall API.
config: Resolver configuration containing credentials.
artifact: Work artifact providing a DOI for lookup.

Returns:
Iterable[ResolverResult]: Resolver results with candidate URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI available for lookup.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if the resolver should execute.

### `iter_urls(self, session, config, artifact)`

Yield URLs discovered via the Crossref API for a given artifact.

Args:
session: Requests session used for API requests.
config: Resolver configuration (polite headers, mailto info).
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate URL results.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact exposes landing page URLs.

Args:
config: Resolver configuration (unused).
artifact: Work artifact with landing page URLs.

Returns:
bool: ``True`` if landing page URLs are available.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered by scraping landing pages.

Args:
session: Requests session used for HTTP calls.
config: Resolver configuration.
artifact: Work artifact providing landing page URLs.

Returns:
Iterable[ResolverResult]: Candidate URL results.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has an arXiv identifier.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if an arXiv ID is present.

### `iter_urls(self, session, config, artifact)`

Yield candidate arXiv download URLs.

Args:
session: Requests session (unused for static URLs).
config: Resolver configuration (unused).
artifact: Work artifact containing an arXiv identifier.

Returns:
Iterable[ResolverResult]: Candidate URLs for the arXiv PDF.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has PMC, PMID, or DOI identifiers.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if identifier data is available.

### `_lookup_pmcids(self, session, identifiers, config)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

Yield candidate PMC download URLs derived from identifiers.

Args:
session: Requests session used for PMC utility calls.
config: Resolver configuration supplying headers/timeouts.
artifact: Work artifact containing PMC, PMID, or DOI identifiers.

Returns:
Iterable[ResolverResult]: Candidate PMC URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI suitable for lookup.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing metadata.

Returns:
bool: ``True`` if a DOI is present.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs from the Europe PMC API.

Args:
session: Requests session used for API calls.
config: Resolver configuration.
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate Europe PMC URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when a CORE API key and DOI are available.

Args:
config: Resolver configuration containing credentials.
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if the resolver should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs returned by the CORE API.

Args:
session: Requests session used for API requests.
config: Resolver configuration containing API keys.
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate CORE URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for DOAJ lookup.

Args:
config: Resolver configuration with optional API key.
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if the resolver should run for this artifact.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered via DOAJ article metadata.

Args:
session: Requests session used for DOAJ API calls.
config: Resolver configuration containing optional API key.
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate DOAJ URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for lookup.

Args:
config: Resolver configuration containing API credentials.
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if lookup should be attempted.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs returned from the Semantic Scholar API.

Args:
session: Requests session (unused; API call uses cached helper).
config: Resolver configuration containing API key.
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate Semantic Scholar URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if the resolver should run.

### `iter_urls(self, session, config, artifact)`

Yield candidate URLs discovered via OpenAIRE search.

Args:
session: Requests session for API calls.
config: Resolver configuration.
artifact: Work artifact with DOI metadata.

Returns:
Iterable[ResolverResult]: Candidate OpenAIRE URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for HAL lookup.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` when the resolver should execute.

### `iter_urls(self, session, config, artifact)`

Yield candidate HAL download URLs.

Args:
session: Requests session.
config: Resolver configuration with polite headers.
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate HAL URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when the artifact has a DOI for OSF lookup.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing identifiers.

Returns:
bool: ``True`` if the resolver should run for this artifact.

### `iter_urls(self, session, config, artifact)`

Yield candidate download URLs from the OSF API.

Args:
session: Requests session for OSF API calls.
config: Resolver configuration with polite headers.
artifact: Work artifact containing a DOI.

Returns:
Iterable[ResolverResult]: Candidate OSF URLs.

### `is_enabled(self, config, artifact)`

Return ``True`` when previous resolvers have recorded failed URLs.

Args:
config: Resolver configuration (unused).
artifact: Work artifact containing resolver metadata.

Returns:
bool: ``True`` if failed URLs exist for the artifact.

### `iter_urls(self, session, config, artifact)`

Yield archived URLs from the Internet Archive when available.

Args:
session: Requests session for Wayback API calls.
config: Resolver configuration.
artifact: Work artifact with previously failed URLs.

Returns:
Iterable[ResolverResult]: Candidate archived URLs.

### `_yield_unique(candidates)`

*No documentation available.*

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

### `ResolverPipeline`

Executes resolvers in priority order until a PDF download succeeds.

Attributes:
config: Resolver configuration shared across executions.
download_func: Callable used to download candidate URLs.
logger: Attempt logger receiving structured records.
metrics: Metrics collector tracking resolver performance.

Examples:
>>> pipeline = ResolverPipeline(
...     [UnpaywallResolver()],
...     ResolverConfig(),
...     download_candidate,
...     JsonlLogger(Path('attempts.jsonl')),
... )
>>> isinstance(pipeline, ResolverPipeline)
True

### `UnpaywallResolver`

Resolve PDFs via the Unpaywall API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = UnpaywallResolver()
>>> resolver.name
'unpaywall'

### `CrossrefResolver`

Resolve candidate URLs from the Crossref metadata API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = CrossrefResolver()
>>> resolver.name
'crossref'

### `LandingPageResolver`

Attempt to scrape landing pages when explicit PDFs are unavailable.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = LandingPageResolver()
>>> resolver.name
'landing_page'

### `ArxivResolver`

Resolve arXiv preprints using arXiv identifier lookups.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = ArxivResolver()
>>> resolver.name
'arxiv'

### `PmcResolver`

Resolve PubMed Central articles via identifiers and lookups.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = PmcResolver()
>>> resolver.name
'pmc'

### `EuropePmcResolver`

Resolve Open Access links via the Europe PMC REST API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = EuropePmcResolver()
>>> resolver.name
'europe_pmc'

### `CoreResolver`

Resolve PDFs using the CORE API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = CoreResolver()
>>> resolver.name
'core'

### `DoajResolver`

Resolve Open Access links using the DOAJ API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = DoajResolver()
>>> resolver.name
'doaj'

### `SemanticScholarResolver`

Resolve PDFs using the Semantic Scholar Graph API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = SemanticScholarResolver()
>>> resolver.name
'semantic_scholar'

### `OpenAireResolver`

Resolve URLs using the OpenAIRE API.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = OpenAireResolver()
>>> resolver.name
'openaire'

### `HalResolver`

Resolve publications from the HAL open archive.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = HalResolver()
>>> resolver.name
'hal'

### `OsfResolver`

Resolve artefacts hosted on the Open Science Framework.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = OsfResolver()
>>> resolver.name
'osf'

### `WaybackResolver`

Fallback resolver that queries the Internet Archive Wayback Machine.

Attributes:
name: Resolver identifier used in configuration.

Examples:
>>> resolver = WaybackResolver()
>>> resolver.name
'wayback'
