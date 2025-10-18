# Content Download Specification

## MODIFIED Requirements

### Requirement: Download outcome construction
Download outcomes MUST accurately encode reason codes and telemetry classifications for every resolver attempt.

#### Scenario: Successful downloads omit reason code
- **WHEN** an artifact is freshly downloaded without hitting conditional logic
- **THEN** the emitted `DownloadOutcome.reason_code` SHALL be `None`
- **AND** the telemetry record SHALL serialize a JSON `null`
- **AND** manifest persistence SHALL record `reason_code` as `NULL`

#### Scenario: Conditional responses set reason code
- **WHEN** a resolver receives a 304 (or equivalent conditional-not-modified signal)
- **THEN** the emitted download outcome SHALL set `ReasonCode.CONDITIONAL_NOT_MODIFIED`
- **AND** the byte counts and hashes SHALL reflect the cached artifact
- **AND** telemetry SHALL count the event as a conditional hit

#### Scenario: Voluntary skip reason code
- **WHEN** a user-configured `skip_large_downloads` threshold prevents a fetch
- **THEN** the download outcome SHALL set `ReasonCode.SKIP_LARGE_DOWNLOAD`
- **AND** the domain budget counter SHALL remain unchanged
- **AND** telemetry dashboards SHALL report voluntary skips separately from budget violations

### Requirement: Cached artifact validation
Cached artifact validation MUST avoid unnecessary full-file hashing while preserving correctness guarantees.

#### Scenario: Fast-path validation
- **WHEN** a cached artifact matches the stored byte size and filesystem mtime
- **THEN** `_validate_cached_artifact` SHALL skip SHA-256 recomputation
- **AND** the function SHALL return a validation success in under 5% of the baseline hashing time
- **AND** telemetry SHALL record the validation mode as `fast_path`

#### Scenario: Forced digest verification
- **WHEN** cache digest verification is explicitly requested (via configuration or CLI flag)
- **THEN** `_validate_cached_artifact` SHALL recompute the digest
- **AND** validation SHALL fail if the recomputed digest mismatches stored metadata
- **AND** the failure SHALL trigger a full re-download attempt

## ADDED Requirements

### Requirement: Range resume deprecation
Range resume MUST remain disabled to prevent truncation until append semantics can be safely implemented in a future change.

#### Scenario: Range requests suppressed
- **WHEN** callers enable resume flags or resolvers advertise range support
- **THEN** the downloader SHALL perform a full download without issuing range requests
- **AND** a structured warning SHALL inform the operator that resume is deprecated
- **AND** telemetry SHALL note the resume request was ignored

#### Scenario: Partial artifact recovery
- **WHEN** a download is interrupted mid-transfer
- **THEN** the subsequent retry SHALL discard the partial file and re-download the artifact from the beginning
- **AND** the resulting artifact hash SHALL match the baseline uninterrupted download hash
- **AND** no truncated files SHALL be persisted to disk

### Requirement: Lazy manifest warm-up
Manifest warm-up MUST scale to large datasets by deferring work until entries are requested.

#### Scenario: Lazy iteration default
- **WHEN** `resolve_config` prepares manifest access for a run
- **THEN** the system SHALL expose a lazy iterator or paginated accessor instead of loading all manifest rows
- **AND** memory usage during startup SHALL remain below 200 MB for manifests with ≥250k rows
- **AND** the first download SHALL start within 20% of baseline startup time

#### Scenario: Opt-in eager warm-up
- **WHEN** the operator enables eager warm-up via CLI flag or configuration
- **THEN** the system SHALL execute the legacy warm-up path
- **AND** telemetry SHALL annotate the run with `manifest_warmup="eager"`
- **AND** documentation SHALL warn that eager warm-up may impact large deployments
