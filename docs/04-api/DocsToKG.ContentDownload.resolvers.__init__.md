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

### `_absolute_url(base, href)`

*No documentation available.*

### `clear_resolver_caches()`

*No documentation available.*

### `default_resolvers()`

*No documentation available.*

### `is_event(self)`

*No documentation available.*

### `get_timeout(self, resolver_name)`

*No documentation available.*

### `is_enabled(self, resolver_name)`

*No documentation available.*

### `log(self, record)`

*No documentation available.*

### `is_pdf(self)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `record_attempt(self, resolver_name, outcome)`

*No documentation available.*

### `record_skip(self, resolver_name, reason)`

*No documentation available.*

### `summary(self)`

*No documentation available.*

### `_respect_rate_limit(self, resolver_name)`

*No documentation available.*

### `_jitter_sleep(self)`

*No documentation available.*

### `run(self, session, artifact, context)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `_lookup_pmcids(self, session, identifiers, config)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `is_enabled(self, config, artifact)`

*No documentation available.*

### `iter_urls(self, session, config, artifact)`

*No documentation available.*

### `_yield_unique(candidates)`

*No documentation available.*

## Classes

### `ResolverResult`

Represents either a candidate URL or an informational event.

### `ResolverConfig`

*No documentation available.*

### `AttemptRecord`

*No documentation available.*

### `AttemptLogger`

*No documentation available.*

### `DownloadOutcome`

*No documentation available.*

### `PipelineResult`

*No documentation available.*

### `Resolver`

*No documentation available.*

### `ResolverMetrics`

*No documentation available.*

### `ResolverPipeline`

Executes resolvers in priority order until a PDF download succeeds.

### `UnpaywallResolver`

*No documentation available.*

### `CrossrefResolver`

*No documentation available.*

### `LandingPageResolver`

*No documentation available.*

### `ArxivResolver`

*No documentation available.*

### `PmcResolver`

*No documentation available.*

### `EuropePmcResolver`

*No documentation available.*

### `CoreResolver`

*No documentation available.*

### `DoajResolver`

*No documentation available.*

### `SemanticScholarResolver`

*No documentation available.*

### `OpenAireResolver`

*No documentation available.*

### `HalResolver`

*No documentation available.*

### `OsfResolver`

*No documentation available.*

### `WaybackResolver`

*No documentation available.*
