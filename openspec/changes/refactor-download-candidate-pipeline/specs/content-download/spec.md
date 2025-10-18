## ADDED Requirements
### Requirement: Modular download execution
The downloader SHALL orchestrate candidate fetches via discrete phases (preflight/cache evaluation, streaming & persistence, post-processing) so each phase can be invoked and tested independently.

#### Scenario: Helper orchestration
- **WHEN** a resolver hands `download_candidate` a URL
- **THEN** the orchestrator delegates to the preflight helper before streaming
- **AND** the streaming helper returns a structured result object consumed by the finalization helper
- **AND** unit tests cover each helper with mocks for HTTP and filesystem dependencies.

### Requirement: Partial artifact cleanup
The downloader SHALL remove partial artifact files (including `.part` suffixes) whenever streaming aborts before completion, unless range resume is explicitly enabled and supported.

#### Scenario: Size limit abort
- **WHEN** streaming exceeds the configured max-bytes limit for a PDF
- **THEN** the system calls `cleanup_sidecar_files` for the artifact
- **AND** no `.part` file remains in the artifact directory after the outcome is returned.

#### Scenario: Resume supported retain partial
- **WHEN** range resume is explicitly enabled and the server returns `206 Partial Content`
- **THEN** the system preserves the partial file required for resume
- **AND** the cleanup logic documents that decision and leaves existing artifacts untouched.

### Requirement: Unified download configuration
The system SHALL expose a single source of truth for download configuration so CLI, pipeline, and resumable workflows share identical option fields and defaults.

#### Scenario: Host accept override survives
- **WHEN** a caller sets `host_accept_overrides` on the configuration object
- **THEN** the same value is present inside the per-download context without manual re-mapping
- **AND** new flags added to the central configuration automatically appear in both CLI and pipeline execution paths.

#### Scenario: CLI to pipeline parity
- **WHEN** CLI arguments enable `skip_head_precheck` and provide a progress callback hook
- **THEN** the unified configuration propagates these settings to the pipeline execution without bespoke wiring
- **AND** tests cover the round-trip via `.to_context()` and `.from_cli_args()`.

### Requirement: Authoritative outcome classification
The runner SHALL trust downloader outcomes as authoritative, avoiding duplicate classification validation while preserving manifest reason codes produced by the downloader.

#### Scenario: Runner manifest logging
- **WHEN** `build_download_outcome` returns a MISS due to HTML tail detection
- **THEN** `process_one_work` records the manifest using the provided reason and detail
- **AND** no secondary validation overwrites those fields.
