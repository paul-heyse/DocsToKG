# Ontology Download Capability Deltas

## ADDED Requirements

### Requirement: Automatic Resolver Fallback on Download Failure

The system SHALL automatically attempt alternative resolvers when the primary resolver download fails due to retryable errors, without requiring manual configuration changes or operator intervention.

#### Scenario: Primary resolver returns service unavailable

- **WHEN** a download from the primary resolver fails with HTTP 503 Service Unavailable status
- **AND** alternative resolvers are configured for the same ontology
- **THEN** the system SHALL automatically attempt download from the next available resolver
- **AND** the system SHALL preserve polite headers and user agent for the fallback attempt
- **AND** the system SHALL record the complete fallback chain in the manifest

#### Scenario: Primary resolver authentication temporarily fails

- **WHEN** a download from the primary resolver fails with HTTP 403 Forbidden status
- **AND** the failure is classified as potentially retryable
- **AND** alternative resolvers are available
- **THEN** the system SHALL attempt download from the next resolver
- **AND** the manifest SHALL record which resolver ultimately succeeded

#### Scenario: Network timeout during primary download

- **WHEN** a download operation times out connecting to the primary resolver
- **AND** additional resolver candidates exist in the plan
- **THEN** the system SHALL fall back to the next candidate without operator intervention
- **AND** the system SHALL log the timeout and fallback attempt for observability

### Requirement: Streaming Normalization for Large Ontologies

The system SHALL support memory-bounded normalization of large ontology files using streaming disk-based processing to prevent memory exhaustion while ensuring deterministic canonical output.

#### Scenario: Normalizing multi-gigabyte ontology

- **WHEN** an ontology file exceeds the configured streaming normalization threshold
- **THEN** the system SHALL serialize the ontology to N-Triples format in a temporary file
- **AND** the system SHALL use external sort to produce deterministically ordered triples
- **AND** the system SHALL compute the SHA-256 hash while streaming the sorted output
- **AND** memory usage SHALL remain bounded regardless of ontology size

#### Scenario: Small ontology uses fast path

- **WHEN** an ontology file is below the streaming normalization threshold
- **THEN** the system SHALL use in-memory normalization for optimal performance
- **AND** the output hash SHALL be identical to what streaming normalization would produce
- **AND** the choice of normalization path SHALL be recorded in the manifest fingerprint

#### Scenario: Cross-platform determinism

- **WHEN** the same ontology is normalized on different platforms
- **THEN** the computed SHA-256 hash SHALL be identical across Linux and macOS systems
- **AND** the normalized output SHALL be byte-for-byte identical regardless of platform

### Requirement: Parallel Resolver Planning with Service Limits

The system SHALL execute resolver planning operations concurrently while respecting per-service rate limits to reduce wall-clock time for batch ontology planning.

#### Scenario: Planning ten ontologies concurrently

- **WHEN** planning is requested for ten different ontologies
- **AND** the concurrent planning limit is configured to eight workers
- **THEN** the system SHALL issue up to eight concurrent API calls to resolver services
- **AND** results SHALL be returned as they complete without waiting for slowest resolver
- **AND** total wall-clock time SHALL be significantly less than sequential planning

#### Scenario: Respecting per-service rate limits during concurrent planning

- **WHEN** multiple ontologies use the same resolver service
- **AND** per-service rate limits are configured
- **THEN** the system SHALL not exceed the configured concurrent request limit per service
- **AND** requests to different services SHALL not be artificially serialized
- **AND** token bucket limits SHALL prevent overwhelming individual services

#### Scenario: Handling partial failures during concurrent planning

- **WHEN** some resolver API calls fail during concurrent planning
- **AND** continue-on-error is enabled in configuration
- **THEN** the system SHALL continue planning remaining ontologies
- **AND** the system SHALL return successful plans while reporting failures
- **AND** exceptions SHALL be logged without terminating the entire batch

### Requirement: CLI Concurrency Control

The system SHALL allow operators to override concurrency limits via command-line flags without requiring configuration file modifications for operational flexibility.

#### Scenario: Override concurrent downloads via CLI flag

- **WHEN** operator invokes pull command with `--concurrent-downloads 3` flag
- **THEN** the system SHALL limit simultaneous download operations to three
- **AND** the CLI value SHALL take precedence over configuration file setting
- **AND** the override SHALL apply only to the current invocation

#### Scenario: Override concurrent planning via CLI flag

- **WHEN** operator invokes plan command with `--concurrent-plans 5` flag
- **THEN** the system SHALL limit simultaneous planning operations to five
- **AND** the thread pool SHALL respect the CLI-specified worker limit
- **AND** configuration file value SHALL remain unchanged for future invocations

#### Scenario: Invalid concurrency value rejected

- **WHEN** operator provides non-positive integer for concurrency flag
- **THEN** the system SHALL reject the argument with descriptive error message
- **AND** the command SHALL not execute with invalid concurrency setting

### Requirement: CLI Host Allowlist Override

The system SHALL allow operators to temporarily add permitted hosts via command-line flags without editing configuration files for ad-hoc operational needs.

#### Scenario: Add host to allowlist via CLI

- **WHEN** operator invokes pull command with `--allowed-hosts example.org,test.com` flag
- **THEN** the system SHALL merge CLI-specified hosts with configuration allowlist
- **AND** downloads from the merged host set SHALL be permitted
- **AND** configuration file SHALL remain unchanged

#### Scenario: Wildcard domain in CLI allowlist

- **WHEN** operator provides `--allowed-hosts *.example.org` flag
- **THEN** the system SHALL accept downloads from any subdomain of example.org
- **AND** wildcard matching SHALL function identically to configuration-based wildcards

#### Scenario: CLI and configuration hosts deduplicated

- **WHEN** the same host appears in both CLI argument and configuration file
- **THEN** the system SHALL deduplicate hosts in the merged allowlist
- **AND** no duplicate entries SHALL exist in the effective allowlist

### Requirement: Version Pruning Management

The system SHALL provide command to delete surplus ontology versions while retaining specified number of most recent versions to manage storage consumption.

#### Scenario: Prune keeping two most recent versions

- **WHEN** operator invokes `prune --keep 2` command
- **AND** an ontology has five stored versions
- **THEN** the system SHALL delete the three oldest versions
- **AND** the system SHALL preserve the two most recent versions
- **AND** latest version marker SHALL continue pointing to newest retained version

#### Scenario: Dry run preview deletions

- **WHEN** operator invokes `prune --keep 3 --dry-run` command
- **THEN** the system SHALL display which versions would be deleted
- **AND** the system SHALL show freed disk space estimate
- **AND** no files SHALL actually be deleted from storage

#### Scenario: Prune specific ontologies only

- **WHEN** operator invokes `prune --keep 1 --ids CHEBI HP` command
- **THEN** the system SHALL prune only CHEBI and HP ontologies
- **AND** other ontologies SHALL remain untouched
- **AND** the system SHALL report pruning actions per ontology

### Requirement: Enhanced System Diagnostics

The system SHALL provide comprehensive diagnostic command reporting system health, tool availability, disk space, configuration validity, and network connectivity for operational troubleshooting.

#### Scenario: Check ROBOT tool availability

- **WHEN** operator invokes `doctor` command
- **THEN** the system SHALL check for ROBOT tool in system PATH
- **AND** when ROBOT is found the system SHALL report its version
- **AND** when ROBOT is missing the system SHALL report clear indication

#### Scenario: Validate rate limit configuration

- **WHEN** operator invokes `doctor` command
- **AND** configuration contains rate limit strings
- **THEN** the system SHALL parse each rate limit against expected pattern
- **AND** the system SHALL report any invalid rate limit with explanation
- **AND** valid rate limits SHALL be confirmed in output

#### Scenario: Test network connectivity to resolvers

- **WHEN** operator invokes `doctor` command
- **THEN** the system SHALL attempt HEAD request to representative endpoint for each resolver
- **AND** the system SHALL use three-second timeout to avoid blocking
- **AND** the system SHALL report connectivity status for OLS, BioPortal, and Bioregistry services

#### Scenario: Check available disk space

- **WHEN** operator invokes `doctor` command
- **THEN** the system SHALL report free disk space in gigabytes for ontology directory
- **AND** when free space drops below ten gigabytes the system SHALL emit warning
- **AND** when free space below ten percent of total the system SHALL emit warning

### Requirement: Planning Introspection

The system SHALL provide commands to filter planning by date and compare plans to identify changes in resolver metadata over time for operational visibility.

#### Scenario: Filter plan by last modified date

- **WHEN** operator invokes `plan --since 2024-01-15` command
- **THEN** the system SHALL exclude ontologies where last-modified predates specified date
- **AND** the system SHALL include ontologies modified on or after specified date
- **AND** date comparison SHALL use timezone-aware datetime handling

#### Scenario: Compare current plan with baseline

- **WHEN** operator invokes `plan diff --baseline previous-plan.json` command
- **THEN** the system SHALL load baseline plan from specified file
- **AND** the system SHALL generate current plan for comparison
- **AND** the system SHALL report added ontologies not present in baseline
- **AND** the system SHALL report removed ontologies absent from current plan
- **AND** the system SHALL report modified ontologies where URL, version, or license changed

#### Scenario: Plan diff with structured output

- **WHEN** operator invokes `plan diff --json` command
- **THEN** the system SHALL emit JSON output with structured change representation
- **AND** JSON output SHALL be consumable by automation tools
- **AND** output SHALL distinguish additions, removals, and modifications clearly

## MODIFIED Requirements

### Requirement: Archive Extraction Security

The system SHALL extract ontology archives using centralized implementation with uniform path traversal prevention and compression bomb detection across all archive formats.

#### Scenario: Reject path traversal in ZIP archive

- **WHEN** a ZIP archive contains member with path traversal attempt
- **THEN** the system SHALL reject the member before extraction
- **AND** the system SHALL log the security violation
- **AND** extraction SHALL fail with descriptive error

#### Scenario: Reject path traversal in TAR archive

- **WHEN** a TAR archive contains member with absolute path or parent directory reference
- **THEN** the system SHALL reject the member before extraction
- **AND** the system SHALL raise configuration error with path details

#### Scenario: Detect compression bomb

- **WHEN** an archive has compression ratio exceeding ten-to-one threshold
- **THEN** the system SHALL reject extraction before consuming disk space
- **AND** the system SHALL log compression ratio and limits
- **AND** error message SHALL indicate compression bomb detection

#### Scenario: Support multiple archive formats uniformly

- **WHEN** ontology is distributed as ZIP, TAR, TGZ, or TXZ archive
- **THEN** the system SHALL detect format from file extension
- **AND** the system SHALL apply identical security checks regardless of format
- **AND** extraction SHALL succeed with consistent output structure

### Requirement: Retry Logic Consistency

The system SHALL use unified retry mechanism with exponential backoff and jitter across resolver API calls and download operations for consistent error handling behavior.

#### Scenario: Retry resolver API timeout

- **WHEN** a resolver API call times out
- **AND** maximum retry attempts not exceeded
- **THEN** the system SHALL wait using exponential backoff before retry
- **AND** random jitter SHALL be added to prevent thundering herd
- **AND** retry attempt SHALL be logged with attempt number

#### Scenario: Retry download connection failure

- **WHEN** a download operation fails with connection error
- **AND** error is classified as retryable
- **THEN** the system SHALL retry using same backoff formula as resolver retries
- **AND** backoff duration SHALL increase exponentially with attempt count

#### Scenario: Non-retryable error fails immediately

- **WHEN** an operation fails with authentication error
- **THEN** the system SHALL classify error as non-retryable
- **AND** the system SHALL raise exception immediately without retry attempts
- **AND** no backoff delay SHALL be applied

### Requirement: Manifest Fingerprint Stability

The system SHALL compute manifest fingerprint including schema version, normalized target formats, and normalization mode to ensure proper cache invalidation on parameter changes.

#### Scenario: Fingerprint includes schema version

- **WHEN** manifest fingerprint is computed
- **THEN** the fingerprint input SHALL include manifest schema version constant
- **AND** changing schema version SHALL produce different fingerprint
- **AND** schema version SHALL be first component in fingerprint

#### Scenario: Fingerprint includes sorted target formats

- **WHEN** target formats are specified in configuration
- **THEN** the fingerprint input SHALL include formats sorted alphabetically
- **AND** format order in configuration SHALL not affect fingerprint
- **AND** changing target formats SHALL produce different fingerprint

#### Scenario: Fingerprint includes normalization mode

- **WHEN** streaming normalization is used versus in-memory normalization
- **THEN** the fingerprint input SHALL include normalization mode identifier
- **AND** changing normalization approach SHALL produce different fingerprint
- **AND** manifest reader can detect when normalization parameters changed
