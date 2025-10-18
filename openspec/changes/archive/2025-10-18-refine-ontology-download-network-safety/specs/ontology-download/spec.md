# Ontology Download Specification

## MODIFIED Requirements

### Requirement: Planner URL hygiene
Planner metadata probes MUST apply the same URL validation policy as download execution to prevent contacting disallowed hosts.

#### Scenario: Planner validates URLs before probing
- **WHEN** `_fetch_last_modified` (or any planner helper) prepares to issue an HTTP request
- **THEN** it SHALL invoke `validate_url_security(plan.url, active_config.defaults.http)`
- **AND** the request SHALL only proceed if validation succeeds
- **AND** failures SHALL raise the same security exception used by download-time validation.

#### Scenario: Disallowed URLs never reach the network
- **WHEN** a planner encounter a URL targeting a forbidden host or scheme
- **THEN** no HTTP request SHALL be dispatched
- **AND** telemetry/log entries SHALL record the blocked probe with planner context.

### Requirement: Consistent networking primitives
Planner HTTP probes MUST reuse the polite networking stack (headers, session pooling, rate limiting, telemetry) to remain compliant with configured limits.

#### Scenario: Session reuse and polite headers
- **WHEN** planner metadata helpers issue HTTP requests
- **THEN** they SHALL obtain sessions from `SessionPool` and headers from `polite_http_headers`
- **AND** the headers SHALL match the download pipeline’s polite signature.

#### Scenario: Rate limiting parity
- **WHEN** planner probes target a host with configured rate limits
- **THEN** the probes SHALL acquire the same per-host bucket as download requests
- **AND** throttling behaviour (token exhaustion, backoff) SHALL mirror the download pipeline
- **AND** telemetry SHALL capture planner probe metrics using the same event type and schema fields as downloads.

### Requirement: Safe ontology index maintenance
Ontology index updates MUST be serialised across versions to prevent concurrent runs from clobbering entries.

#### Scenario: Ontology-scoped locking
- **WHEN** `_append_index_entry` (or equivalent) writes to `index.json`
- **THEN** the operation SHALL acquire an ontology-level lock before mutating the file
- **AND** concurrent writers for different versions SHALL queue until the lock is released
- **AND** no completed entry SHALL be lost due to interleaved writes.

#### Scenario: Concurrency observability
- **WHEN** lock contention occurs during index updates
- **THEN** logs/telemetry SHALL record wait durations and lock ownership
- **AND** operators SHALL be able to diagnose contention impacting throughput.

### Requirement: Planner probing configurability
Operators MUST be able to disable planner probing entirely after URL validation in environments where metadata requests are disallowed.

#### Scenario: Probing disabled via configuration
- **WHEN** the probe opt-out configuration or CLI flag is set
- **THEN** planner helpers SHALL skip network metadata requests after successful URL validation
- **AND** telemetry/log output SHALL note that probing was intentionally disabled
- **AND** planning SHALL proceed using cached metadata or defaults without HTTP calls.

#### Scenario: Default behaviour remains opt-in
- **WHEN** the configuration is not set
- **THEN** planner probes SHALL execute as normal
- **AND** operators SHALL receive documentation describing the trade-offs before enabling the opt-out.
