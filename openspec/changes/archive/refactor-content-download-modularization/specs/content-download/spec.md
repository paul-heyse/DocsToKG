# Content Download Specification

## ADDED Requirements

### Requirement: Modular Resolver Architecture

The system SHALL organize content resolvers into focused, independently testable modules using a plugin-based registry pattern.

#### Scenario: Resolver discovery and registration

- **WHEN** the system initializes the resolver pipeline
- **THEN** all resolver modules under `resolvers/` SHALL be automatically discovered
- **AND** each resolver SHALL register with the `ResolverRegistry` via the `RegisteredResolver` mixin
- **AND** the registry SHALL return resolver instances in configured priority order

#### Scenario: Adding a new resolver

- **WHEN** a contributor creates a new resolver module in `resolvers/`
- **THEN** the resolver SHALL inherit from `RegisteredResolver` or `ApiResolverBase`
- **AND** the resolver SHALL declare a unique `name` attribute
- **AND** the resolver SHALL implement the `Resolver` protocol (`is_enabled`, `iter_urls`)
- **AND** the resolver SHALL be automatically discovered without modifying the registry

#### Scenario: Resolver module isolation

- **WHEN** a resolver implementation is modified
- **THEN** the change SHALL NOT require modifications to other resolver modules
- **AND** the resolver SHALL be testable in isolation without instantiating the full pipeline
- **AND** the resolver SHALL access shared utilities via `resolvers/base.py`

#### Scenario: Backward-compatible imports

- **WHEN** downstream code imports from `ContentDownload.pipeline`
- **THEN** the import SHALL succeed via re-exports from `pipeline.py`
- **AND** a deprecation warning SHALL be logged for internal module imports
- **AND** the warning SHALL reference the new import path in `resolvers/`

### Requirement: Pure Configuration Resolution

The system SHALL separate configuration computation from I/O side effects, returning an immutable configuration dataclass.

#### Scenario: Side-effect-free configuration

- **WHEN** `resolve_config(args, parser)` is called
- **THEN** the function SHALL return a frozen `ResolvedConfig` dataclass
- **AND** the function SHALL NOT write to the filesystem
- **AND** the function SHALL NOT mutate the `args` namespace
- **AND** the function SHALL NOT pull credentials from environment variables (except via explicit credential resolution)

#### Scenario: Filesystem initialization

- **WHEN** `bootstrap_run_environment(resolved)` is called
- **THEN** the function SHALL create the PDF output directory
- **AND** the function SHALL create the HTML output directory
- **AND** the function SHALL create the XML output directory
- **AND** the function SHALL initialize telemetry state (if applicable)

#### Scenario: Configuration immutability

- **WHEN** a `ResolvedConfig` instance is created
- **THEN** all fields SHALL be immutable (frozen dataclass)
- **AND** attempts to modify fields SHALL raise `FrozenInstanceError`
- **AND** the configuration SHALL be serializable to JSON/YAML
- **AND** the configuration SHALL be deserializable from JSON/YAML

#### Scenario: Configuration reuse

- **WHEN** a `ResolvedConfig` is passed to multiple run invocations
- **THEN** each run SHALL execute with identical configuration
- **AND** the configuration SHALL NOT accumulate state between runs
- **AND** the configuration SHALL be safe to share across threads

### Requirement: Composable Runner Orchestration

The system SHALL decompose the download pipeline into composable stages managed by a `DownloadRun` class.

#### Scenario: Stage-based initialization

- **WHEN** a `DownloadRun` instance is created with `ResolvedConfig`
- **THEN** the runner SHALL provide a `setup_sinks()` method to initialize telemetry sinks
- **AND** the runner SHALL provide a `setup_resolver_pipeline()` method to create the resolver pipeline
- **AND** the runner SHALL provide a `setup_work_provider()` method to create the OpenAlex work provider
- **AND** the runner SHALL provide a `setup_download_state()` method to initialize download tracking
- **AND** the runner SHALL provide a `setup_worker_pool()` method to create the thread pool

#### Scenario: Independent stage testing

- **WHEN** a stage method is tested in isolation
- **THEN** the stage SHALL be callable without executing other stages
- **AND** the stage SHALL return its configured component (sink, pipeline, provider, etc.)
- **AND** the stage SHALL be mockable for integration tests

#### Scenario: Run orchestration

- **WHEN** `DownloadRun.run()` is invoked
- **THEN** the runner SHALL call `setup_sinks()` to initialize telemetry
- **AND** the runner SHALL call `setup_resolver_pipeline()` to create resolvers
- **AND** the runner SHALL call `setup_work_provider()` to create the work source
- **AND** the runner SHALL call `setup_download_state()` to initialize tracking
- **AND** the runner SHALL call `setup_worker_pool()` to create workers
- **AND** the runner SHALL iterate over work items, calling `process_work_item()` for each
- **AND** the runner SHALL return a `RunResult` summarizing outcomes

### Requirement: Strategy-Based Download Processing

The system SHALL use a strategy pattern to isolate artifact-specific download logic (PDF, HTML, XML).

#### Scenario: Strategy protocol definition

- **WHEN** a `DownloadStrategy` is defined
- **THEN** the strategy SHALL implement `should_download(artifact, context) -> bool`
- **AND** the strategy SHALL implement `process_response(response, artifact, context) -> Classification`
- **AND** the strategy SHALL implement `finalize_artifact(artifact, classification, context) -> DownloadOutcome`

#### Scenario: Strategy selection

- **WHEN** a work item is processed
- **THEN** the system SHALL select a strategy based on expected classification
- **AND** the strategy SHALL determine if download should proceed via `should_download()`
- **AND** the strategy SHALL process the HTTP response via `process_response()`
- **AND** the strategy SHALL finalize the artifact via `finalize_artifact()`

#### Scenario: PDF download strategy

- **WHEN** `PdfDownloadStrategy` is applied
- **THEN** `should_download()` SHALL check for existing PDF artifacts
- **AND** `process_response()` SHALL validate PDF magic bytes and EOF markers
- **AND** `finalize_artifact()` SHALL create content-addressed symlinks (if enabled)
- **AND** `finalize_artifact()` SHALL log PDF-specific metadata to telemetry

#### Scenario: HTML download strategy

- **WHEN** `HtmlDownloadStrategy` is applied
- **THEN** `should_download()` SHALL check for existing HTML artifacts
- **AND** `process_response()` SHALL detect HTML content via magic bytes or Content-Type
- **AND** `finalize_artifact()` SHALL extract text content (if `--extract-text=html` enabled)
- **AND** `finalize_artifact()` SHALL log HTML-specific metadata to telemetry

#### Scenario: XML download strategy

- **WHEN** `XmlDownloadStrategy` is applied
- **THEN** `should_download()` SHALL check for existing XML artifacts
- **AND** `process_response()` SHALL detect XML content via magic bytes or Content-Type
- **AND** `finalize_artifact()` SHALL validate XML structure (if enabled)
- **AND** `finalize_artifact()` SHALL log XML-specific metadata to telemetry

#### Scenario: Adding a new artifact strategy

- **WHEN** a contributor adds a new artifact type (e.g., DOCX)
- **THEN** the contributor SHALL create a new `DocxDownloadStrategy` class
- **AND** the strategy SHALL implement the `DownloadStrategy` protocol
- **AND** the strategy SHALL be registered in the strategy selection logic
- **AND** the strategy SHALL NOT require modifications to existing strategies

### Requirement: Focused Download Helper Functions

The system SHALL extract download processing concerns into focused, testable functions.

#### Scenario: Classification validation

- **WHEN** `validate_classification(classification, artifact, options)` is called
- **THEN** the function SHALL return `ValidationResult` indicating success or failure
- **AND** the function SHALL check for expected vs. actual classification mismatch
- **AND** the function SHALL validate minimum file size constraints
- **AND** the function SHALL be testable without downloading actual content

#### Scenario: Resume logic handling

- **WHEN** `handle_resume_logic(artifact, previous_index, options)` is called
- **THEN** the function SHALL return `ResumeDecision` indicating whether to skip download
- **AND** the function SHALL check for cached artifacts in `previous_index`
- **AND** the function SHALL honor `--force` overrides
- **AND** the function SHALL be testable with mock previous attempts

#### Scenario: Sidecar cleanup

- **WHEN** `cleanup_sidecar_files(artifact, classification, options)` is called
- **THEN** the function SHALL remove `.part` temporary files
- **AND** the function SHALL remove classification-mismatched artifacts
- **AND** the function SHALL preserve successfully downloaded artifacts
- **AND** the function SHALL log cleanup actions to telemetry

#### Scenario: Download outcome construction

- **WHEN** `build_download_outcome(artifact, classification, attempts)` is called
- **THEN** the function SHALL return a `DownloadOutcome` dataclass
- **AND** the outcome SHALL include the final classification
- **AND** the outcome SHALL include all resolver attempts
- **AND** the outcome SHALL include download statistics (bytes, duration)
- **AND** the function SHALL be testable without downloading actual content

### Requirement: Backward Compatibility

The system SHALL maintain backward compatibility with existing CLI usage, configuration files, and telemetry formats.

#### Scenario: CLI interface preservation

- **WHEN** a user invokes the CLI with existing arguments
- **THEN** the CLI SHALL behave identically to the pre-refactoring implementation
- **AND** all flags SHALL retain their existing semantics
- **AND** output paths SHALL match the pre-refactoring structure
- **AND** telemetry logs SHALL use the same format

#### Scenario: Configuration file compatibility

- **WHEN** a user provides a resolver configuration file (YAML/JSON)
- **THEN** the configuration SHALL be parsed identically to the pre-refactoring implementation
- **AND** resolver toggles SHALL apply correctly
- **AND** domain-specific rules SHALL be enforced
- **AND** credential overrides SHALL work as before

#### Scenario: Telemetry format preservation

- **WHEN** download outcomes are logged to telemetry sinks
- **THEN** the JSONL format SHALL match the pre-refactoring schema
- **AND** the CSV format SHALL match the pre-refactoring schema
- **AND** the SQLite schema SHALL match the pre-refactoring schema
- **AND** existing telemetry parsing scripts SHALL continue to work

### Requirement: Performance Neutrality

The system SHALL maintain performance within 5% of the pre-refactoring baseline.

#### Scenario: Resolver execution performance

- **WHEN** resolvers are executed for a batch of 1,000 works
- **THEN** the total execution time SHALL be within 5% of the baseline
- **AND** the number of HTTP requests SHALL match the baseline
- **AND** memory usage SHALL not increase by more than 10%

#### Scenario: Download processing performance

- **WHEN** 1,000 PDFs are downloaded and classified
- **THEN** the total processing time SHALL be within 5% of the baseline
- **AND** the throughput (PDFs/second) SHALL match the baseline
- **AND** the classification accuracy SHALL match the baseline

#### Scenario: Configuration resolution performance

- **WHEN** `resolve_config()` is invoked
- **THEN** the resolution time SHALL be within 5% of the baseline
- **AND** the function SHALL NOT introduce additional network calls
- **AND** the function SHALL NOT introduce additional filesystem reads

### Requirement: Test Coverage Maintenance

The system SHALL maintain integration test coverage at >= 85% across the refactored modules.

#### Scenario: Unit test coverage for resolvers

- **WHEN** resolver modules are tested
- **THEN** each resolver SHALL have unit tests for `is_enabled()` logic
- **AND** each resolver SHALL have unit tests for `iter_urls()` logic
- **AND** each resolver SHALL have tests for error handling (HTTP errors, JSON errors, timeouts)

#### Scenario: Integration test coverage for runner

- **WHEN** the `DownloadRun` class is tested
- **THEN** integration tests SHALL verify end-to-end download pipeline execution
- **AND** integration tests SHALL verify resumable downloads, telemetry emission, and throttling behaviour
- **AND** integration tests SHALL verify telemetry logging
- **AND** integration tests SHALL verify worker pool concurrency

#### Scenario: Strategy test coverage

- **WHEN** download strategies are tested
- **THEN** each strategy SHALL have unit tests for `should_download()` logic
- **AND** each strategy SHALL have unit tests for `process_response()` logic
- **AND** each strategy SHALL have unit tests for `finalize_artifact()` logic
- **AND** integration tests SHALL verify strategy selection logic

## MODIFIED Requirements

None (this is a new capability specification).

## REMOVED Requirements

None (this is a new capability specification).

## RENAMED Requirements

None (this is a new capability specification).
