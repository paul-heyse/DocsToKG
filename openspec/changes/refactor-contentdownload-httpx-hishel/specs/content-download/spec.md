# ContentDownload Specification Deltas

## ADDED Requirements

### Requirement: HTTPX Client With Hishel Cache
`DocsToKG.ContentDownload` SHALL expose a singleton `httpx.Client` decorated with a Hishel RFC-9111 cache so all ContentDownload HTTP traffic shares one configured transport, cache policy, and telemetry surface.

#### Scenario: Client Configures Timeouts and Pool Limits
- **WHEN** `DocsToKG.ContentDownload.httpx_transport.get_http_client()` provisions the client
- **THEN** it SHALL apply explicit per-phase timeouts (connect, read, write, pool) and connection limits (`max_connections`, `max_keepalive_connections`, `keepalive_expiry`) aligned with ContentDownload concurrency defaults.

#### Scenario: Client Enables HTTP/2 and Secure SSL Context
- **GIVEN** the client is constructed
- **THEN** it SHALL enable HTTP/2 support
- **AND** it SHALL use an `ssl.SSLContext` rooted in the bundled trust store (e.g., Certifi) for verification, with optional proxy mounts governed by configuration.

#### Scenario: Hishel Cache Stores Responses Under Cache Directory
- **WHEN** the client issues cacheable GET/HEAD requests
- **THEN** Hishel SHALL persist entries under `CACHE_DIR/http/`
- **AND** cached revalidation SHALL respect `ETag`/`Last-Modified` semantics without requiring manual header management in call sites.

### Requirement: HTTPX Transport Powers ContentDownload Networking
All ContentDownload networking entry points SHALL consume the shared HTTPX client, ensuring Tenacity retries operate on `httpx` responses/exceptions and that no direct `requests` session leasing remains.

#### Scenario: Tenacity Helper Retries HTTPX Responses
- **GIVEN** `DocsToKG.ContentDownload.networking.request_with_retries` executes
- **WHEN** HTTP calls fail with retryable status codes or `httpx` timeout/transport exceptions
- **THEN** the Tenacity controller SHALL operate on `httpx.Response` objects, closing previous responses before sleeping, and preserving the existing behaviours for retry-after caps, max retry duration, and final-response return semantics.

#### Scenario: Resolvers and Robots Use Shared Client
- **WHEN** resolvers issue JSON metadata requests, the pipeline performs head pre-checks, or `RobotsCache._fetch` retrieves `robots.txt`
- **THEN** each helper SHALL obtain the shared HTTPX client (directly or via `request_with_retries`) and SHALL NOT construct or lease `requests.Session` instances.

#### Scenario: Configuration Permits Client Injection for Tests
- **WHEN** tests or tooling call `configure_http_client(...)` (or equivalent hook)
- **THEN** they SHALL be able to inject an `httpx.Client`/transport override without mutating global state outside the provided API, allowing deterministic tests via `httpx.MockTransport`.

### Requirement: Streaming Downloads Use HTTPX With Atomic Writes
ContentDownload streaming SHALL rely on HTTPX streaming APIs while maintaining existing manifest semantics, atomic file writes, content-policy enforcement, and cached result handling.

#### Scenario: Streaming Download Writes Atomically
- **WHEN** `stream_candidate_payload` downloads a payload
- **THEN** it SHALL use the HTTPX streaming iterator to write to a temporary path that is atomically promoted on success
- **AND** it SHALL emit progress callbacks and telemetry identical to the pre-refactor behaviour.

#### Scenario: Cached 304 Responses Yield CachedResult
- **GIVEN** Hishel returns a validated 304 response for a conditional request
- **WHEN** ContentDownload interprets the response
- **THEN** it SHALL populate `CachedResult` with the previously cached artifact metadata, consistent with current conditional download semantics.

#### Scenario: Content Policy Validation Uses HTTPX Headers
- **WHEN** `_enforce_content_policy` evaluates response metadata
- **THEN** it SHALL inspect the `httpx.Response` headers (`Content-Type`, `Content-Length`, etc.)
- **AND** violations SHALL raise `ContentPolicyViolation` exactly as they do today.
