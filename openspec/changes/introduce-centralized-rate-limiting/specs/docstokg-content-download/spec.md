## ADDED Requirements

### Requirement: Centralized Rate Limiting Transport
The networking layer SHALL route every outbound HTTP request through a `RateLimitedTransport` built on `pyrate_limiter.Limiter`, positioned beneath the Hishel cache transport so that only cache misses and revalidation attempts consume limiter capacity. Each request SHALL declare a role (`metadata`, `landing`, `artifact`) via `request.extensions["role"]`, defaulting to `metadata` when omitted, and the transport SHALL select the `(host, role)` limiter accordingly. The transport SHALL respect per-role `max_delay_ms` budgets defined in policy; if the limiter cannot acquire within that budget it SHALL raise `RateLimitError` carrying host, role, elapsed wait, and next-available timestamps. HEAD requests SHALL bypass acquisition when the policy marks `count_head=False`.

#### Scenario: Rate-limited cache miss
- **WHEN** a metadata request for `api.openalex.org` is a cache miss and the limiter has capacity
- **THEN** the transport SHALL acquire tokens for `(api.openalex.org, metadata)` before issuing the network call

#### Scenario: Cache hit bypasses limiter
- **WHEN** Hishel serves a landing-page response from cache without hitting the transport
- **THEN** no limiter acquisition SHALL occur and limiter metrics SHALL remain unchanged

#### Scenario: Excess wait raises RateLimitError
- **WHEN** an artifact download exhausts its policy allowance and the computed wait exceeds `artifact.max_delay_ms`
- **THEN** the transport SHALL raise `RateLimitError` with host, role, waited duration, and next-allowed metadata, and the request SHALL NOT be retried by Tenacity

#### Scenario: Tenacity does not retry RateLimitError
- **WHEN** `RateLimitedTransport` raises `RateLimitError` while `request_with_retries` orchestrates a call
- **THEN** the Tenacity controller SHALL stop retrying immediately and bubble the exception so callers can log the limiter block

### Requirement: Rate Limiting Configuration and Overrides
The CLI and environment configuration SHALL expose host-level rate policies in the form of ordered rate lists (e.g., `--rate api.crossref.org=10/s,5000/h`), limiter modes (`wait:<ms>` or `raise`), backend selection (`--rate-backend={memory,multiprocess,sqlite,redis,postgres}`), and per-role `max_delay_ms`. Defaults SHALL be defined in a policy registry per `(host, role)` and validated at startup with `pyrate_limiter.validate_rate_list`. Legacy throttling flags (`--sleep`, `--domain-token-bucket`, `--max-concurrent-per-host`) SHALL either map deterministically onto the new policy model or emit deprecation warnings documenting the replacement. Effective policies SHALL be logged at startup in a structured table listing host, role, rates, mode, max delay, backend, and whether HEAD requests are counted.

#### Scenario: CLI override replaces defaults
- **WHEN** the user passes `--rate api.openalex.org=5/s,300/h --rate-mode api.openalex.org=wait:200`
- **THEN** the registry SHALL reflect those values for both metadata and landing roles before any limiter is constructed

#### Scenario: Invalid rate ordering fails fast
- **WHEN** a rate override provides an incorrectly ordered rate list
- **THEN** startup SHALL abort with a validation error describing the host, role, and offending rate sequence

#### Scenario: Legacy flag emits deprecation mapping
- **WHEN** the user supplies `--domain-token-bucket 3/second`
- **THEN** the CLI SHALL translate the value into the new metadata policy and emit a warning indicating the equivalent `--rate` syntax

### Requirement: Rate Limiting Telemetry and Reporting
Limiter activity SHALL be surfaced through telemetry counters and histograms keyed by host and role: counts of acquisitions, cumulative wait durations, and total blocks. Each HTTP attempt record SHALL annotate whether it was served from cache, waited for limiter tokens (with wait milliseconds), or failed due to rate limiting. Run summaries SHALL include aggregated limiter statistics, and manifest or summary logs SHALL embed host, role, wait, and next-allowed details whenever a `RateLimitError` occurs.

#### Scenario: Telemetry records wait duration
- **WHEN** a metadata request waits 120 ms for limiter capacity
- **THEN** telemetry SHALL increment the `rate_limiter_acquire_total` counter for `(host=api.openalex.org, role=metadata)` and add 120 ms to the wait histogram

#### Scenario: Summary surfaces limiter outcomes
- **WHEN** a run completes with limiter waits or blocks
- **THEN** the run summary SHALL list total waits, average wait duration, and total blocks per host/role alongside existing retry metrics

#### Scenario: RateLimitError manifests metadata
- **WHEN** an artifact request raises `RateLimitError`
- **THEN** the manifest entry SHALL record host, role, waited duration, and next-allowed timestamp fields for downstream analysis

### Requirement: Pipeline Role Attribution and Legacy Throttle Removal
Resolver pipelines and download helpers SHALL set an explicit limiter role when constructing HTTP requests and SHALL remove bespoke rate-limiting constructs such as `TokenBucket` sleeps and per-host semaphores. Role assignment MUST follow the canonical mapping: metadata/API calls → `"metadata"`, HTML landing-page fetches → `"landing"`, and artifact downloads/streams → `"artifact"`. Any legacy options that previously configured token buckets or host semaphores SHALL translate into the centralized limiter configuration or log deprecation warnings directing operators to the new `--rate*` flags.

#### Scenario: Pipeline request assigns role
- **WHEN** the download pipeline schedules an artifact GET
- **THEN** it SHALL call `request_with_retries(..., role="artifact")` so the limiter applies the artifact policy

#### Scenario: Legacy token bucket removed
- **WHEN** a user runs with prior `--domain-token-bucket` settings
- **THEN** the pipeline SHALL not instantiate a `TokenBucket` or sleep loop; instead the configuration SHALL convert to limiter policies and emit a deprecation warning describing the new flags
