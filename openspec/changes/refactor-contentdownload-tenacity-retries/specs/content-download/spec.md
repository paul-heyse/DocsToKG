# ContentDownload Specification Deltas

## ADDED Requirements

### Requirement: Tenacity-Governed HTTP Retries
`DocsToKG.ContentDownload.networking.request_with_retries` SHALL delegate all retry logic to a Tenacity controller that respects HTTP retryable status codes, `Retry-After` headers, and configured attempt/time budgets while closing retryable responses before sleeping.

#### Scenario: Retryable Status Honours Retry-After
- **GIVEN** `request_with_retries` executes with `respect_retry_after=True`, `retry_after_cap=30`, and `backoff_max=120`
- **AND** the initial attempt returns HTTP `503` with header `Retry-After: 60`
- **WHEN** the Tenacity policy schedules the next attempt
- **THEN** the computed wait duration SHALL be capped to `30` seconds (the smaller of the header value, `retry_after_cap`, and `backoff_max`)
- **AND** the `requests.Response` from the failed attempt SHALL be closed before the delay begins.

#### Scenario: Non-Retryable Status Returns Immediately
- **GIVEN** `request_with_retries` is invoked with the default retry status set `{429, 500, 502, 503, 504}`
- **WHEN** the first attempt returns HTTP `404`
- **THEN** the helper SHALL return that response immediately without additional Tenacity attempts or sleeps.

#### Scenario: Exhausted HTTP Retries Return Last Response
- **GIVEN** `request_with_retries` runs with `max_retries=2`
- **AND** each attempt receives HTTP `503`
- **WHEN** the retry budget is exhausted
- **THEN** the helper SHALL return the final `requests.Response` to the caller (without raising `RetryError`) so downstream code can log or inspect the failure.

#### Scenario: Exhausted Network Failures Raise
- **GIVEN** `request_with_retries` runs with `max_retries=1`
- **WHEN** both attempts raise `requests.Timeout`
- **THEN** the helper SHALL re-raise the last timeout exception after Tenacity exhausts the retry budget.

### Requirement: Single Tenacity Retry Surface
All HTTP requests initiated by the ContentDownload pipeline (resolvers, head pre-check, robots policy fetches) SHALL run through the Tenacity-backed `DocsToKG.ContentDownload.networking.request_with_retries`, ensuring consistent retry semantics and shared observability.

#### Scenario: Resolver JSON Requests Use Shared Policy
- **GIVEN** a resolver implementing `ApiResolverBase._request_json`
- **WHEN** it issues an HTTP request for metadata
- **THEN** the request SHALL be executed via `DocsToKG.ContentDownload.networking.request_with_retries`
- **AND** no resolver-specific retry loops SHALL be present around that call.

#### Scenario: Head Precheck Uses Shared Policy
- **GIVEN** `ResolverPipeline` performs a HEAD preflight for a candidate URL
- **WHEN** it probes the URL (including the GET fallback)
- **THEN** both requests SHALL be routed through the shared Tenacity entrypoint with `max_retries=1` for the HEAD attempt.

#### Scenario: Robots Cache Uses Shared Policy
- **GIVEN** the pipeline fetches `robots.txt` for a host
- **WHEN** `RobotsCache._fetch` obtains the policy
- **THEN** it SHALL call `DocsToKG.ContentDownload.networking.request_with_retries` (with its single retry) instead of opening raw `session.get` connections.
