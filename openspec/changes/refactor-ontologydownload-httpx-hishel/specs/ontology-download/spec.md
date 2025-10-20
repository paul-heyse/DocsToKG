# Ontology Download Specification Deltas

## ADDED Requirements

### Requirement: Shared HTTPX Client with Hishel Cache
OntologyDownload SHALL maintain a singleton `httpx.Client` decorated by a Hishel disk cache (`CACHE_DIR/http/`) and reuse it for every ontology HTTP request (planner probes, checksum fetches, streaming downloads).

#### Scenario: Conditional GET served from cache
- **WHEN** `download_stream` revalidates a URL whose previous manifest contains `etag` and `last_modified`
- **AND** the origin responds `304 Not Modified`
- **THEN** the shared HTTPX client (via Hishel) SHALL surface the cached body without re-downloading
- **AND** `DownloadResult.status` SHALL be `"cached"` while `etag`, `last_modified`, `content_type`, and `content_length` reflect the cached metadata.

#### Scenario: Client enforces pooled timeouts and SSL context
- **WHEN** the shared client is constructed
- **THEN** it SHALL configure explicit connect/read/write/pool timeouts, `httpx.Limits` for connection pooling, `http2` derived from configuration, and an `ssl.SSLContext` seeded from Certifi
- **AND** `trust_env` SHALL remain true so proxy environment variables are honoured when set.

#### Scenario: Redirects remain opt-in
- **WHEN** the shared client issues requests on behalf of planner or downloader helpers
- **THEN** `follow_redirects` SHALL default to `False`
- **AND** redirect handling SHALL continue to be enforced by `request_with_redirect_audit`, which validates every Location header via `validate_url_security` before following it.

#### Scenario: Telemetry hook stamps headers and errors early
- **WHEN** any request is issued through the shared client
- **THEN** event hooks SHALL inject the polite `User-Agent`/correlation headers computed by `DownloadConfiguration.polite_http_headers`
- **AND** `response.raise_for_status()` SHALL be invoked before higher layers process the response so HTTP errors propagate consistently.

### Requirement: Injectable HTTPX Client
OntologyDownload SHALL expose helper APIs that let tests and configuration swap or wrap the shared HTTPX client without mutating callers.

#### Scenario: Tests install MockTransport
- **WHEN** test utilities call `configure_http_client(mock_client=httpx.Client(transport=MockTransport(...)))`
- **THEN** subsequent planner/download invocations SHALL use that client
- **AND** restoring the previous client SHALL leave production defaults intact (timeouts, Hishel cache, hooks).

#### Scenario: DownloadConfiguration session factory bridges to HTTPX
- **WHEN** `DownloadConfiguration.set_session_factory` is provided with a callable returning an HTTPX client (or transport overrides)
- **THEN** `get_http_client()` SHALL invoke that factory once and reuse the resulting client
- **AND** the factory SHALL be able to return `None` to fall back to the default Hishel-backed client.

## MODIFIED Requirements

### Requirement: Consistent networking primitives
Planner probes, checksum fetchers, and downloads MUST use the same HTTPX client helpers, rate limiting, and URL validation.

#### Scenario: Planner probes reuse HTTPX client
- **WHEN** `planner_http_probe` prepares to issue HEAD/GET requests
- **THEN** it SHALL call `get_http_client()` once per probe
- **AND** the returned client SHALL execute requests while respecting `get_bucket` consumption, `validate_url_security`, and retry/backoff semantics previously applied to `SessionPool`.

#### Scenario: Checksum fetchers stream via HTTPX
- **WHEN** `_fetch_checksum_from_url` streams checksum bytes
- **THEN** it SHALL call `get_http_client().stream("GET", ...)` instead of leasing a `requests.Session`
- **AND** retries SHALL continue to use `retry_with_backoff` with `is_retryable_error` adapted for HTTPX exceptions.

#### Scenario: Streaming downloader delegates network IO to HTTPX
- **WHEN** `StreamingDownloader` performs HEAD/GET operations
- **THEN** it SHALL issue them through the shared HTTPX client (available via helper injection)
- **AND** no code path inside OntologyDownload SHALL instantiate `requests.Session` or access `SESSION_POOL`.
- **AND** conditional GETs SHALL rely on Hishel for validator negotiation, with downloader logic only interpreting the resulting status/headers.
