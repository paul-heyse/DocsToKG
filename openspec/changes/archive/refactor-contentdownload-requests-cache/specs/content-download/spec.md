# ContentDownload Specification Deltas

## ADDED Requirements

### Requirement: Central Cached Session for Metadata Requests
`DocsToKG.ContentDownload.networking` SHALL expose a single `requests_cache.CachedSession` factory that every metadata HTTP call uses, while binary download flows explicitly bypass caching.

#### Scenario: Resolver Requests Use Cached Session
- **GIVEN** any resolver issues an HTTP GET/HEAD for metadata (e.g., Crossref, Unpaywall, Semantic Scholar)
- **WHEN** the resolver calls the networking helper
- **THEN** the request SHALL be executed through the shared `CachedSession`
- **AND** repeated calls with identical parameters SHALL return responses flagged with `from_cache=True` without re-contacting the origin.

#### Scenario: PDF Streams Stay Uncached
- **GIVEN** `download.stream_candidate_payload` fetches a PDF or other binary artifact
- **WHEN** the request is executed
- **THEN** the helper SHALL mark the call `expire_after=DO_NOT_CACHE` (or use an equivalent disabled context) so no binary body is stored in the cache backend.

#### Scenario: Offline Mode Serves Cache Only
- **GIVEN** the offline cache mode is enabled via configuration or environment variable
- **WHEN** a resolver attempts a metadata request whose cache entry exists
- **THEN** the response SHALL be served from cache without hitting the network
- **AND** if no cache entry exists, the request SHALL raise the documented 504-style error surfaced by `requests-cache`.

### Requirement: Cache Policy Enforcement
The cached session SHALL enforce a central policy: respect HTTP validators (`cache_control=True`, `always_revalidate=True`), apply per-host TTL overrides, exclude binary/oversized content, honour `stale_if_error`/`stale_while_revalidate`, and ignore transient query parameters when building cache keys.

#### Scenario: Per-Host TTL Applied
- **GIVEN** the TTL map assigns `Crossref` endpoints a 48-hour expiration
- **WHEN** the first request stores a response
- **AND** a subsequent request occurs within 48 hours without server revalidation
- **THEN** the cached response SHALL be returned immediately with `from_cache=True`.

#### Scenario: Stale If Error
- **GIVEN** a cached response has expired but `stale_if_error=10 minutes`
- **AND** the origin returns HTTP 503 during revalidation
- **WHEN** the request is retried
- **THEN** the cached response SHALL be served despite being expired, and telemetry SHALL record that stale content was delivered because of upstream failure.

#### Scenario: Binary Response Excluded
- **GIVEN** a metadata probe unexpectedly returns `Content-Type: application/pdf`
- **WHEN** the cached session processes the response
- **THEN** the configured filter SHALL refuse to cache it, ensuring no binary payload is persisted.

#### Scenario: Ignored Query Parameters
- **GIVEN** two requests differ only by a `utm_source` query parameter
- **WHEN** the cached session calculates the cache key
- **THEN** the ignored parameter SHALL be removed, resulting in a cache hit for the second request.

### Requirement: Cache Telemetry
ContentDownload SHALL record cache effectiveness metrics (hit/miss counts, stale-served events, response age) within run summaries and attempt telemetry.

#### Scenario: Manifest Captures Cache Metrics
- **GIVEN** a run completes with both cache hits and misses
- **WHEN** the manifest summary is written
- **THEN** the summary SHALL include counters for hits, misses, stale responses, and offline-only requests, allowing operators to assess cache performance.
