## ADDED Requirements

### Requirement: Stage Modules Are Decomposed
DocParsing SHALL expose each stage (chunking, embedding) as a package with dedicated modules for configuration, CLI wiring, and runtime orchestration while preserving the legacy public imports.

#### Scenario: Stage CLI re-exports
- **WHEN** a caller imports `DocsToKG.DocParsing.embedding.main` or `DocsToKG.DocParsing.chunking.main`
- **THEN** the import SHALL succeed via a thin wrapper module
- **AND** the wrapper SHALL re-export the stage package's `cli.main` entrypoint
- **AND** the stage package SHALL also expose `build_parser` and the config dataclass without importing runtime workers at import time.

#### Scenario: Runtime import isolation
- **WHEN** a unit test imports `DocsToKG.DocParsing.embedding.runtime` (or `chunking.runtime`)
- **THEN** the module SHALL load without triggering CLI parser construction
- **AND** it SHALL offer a stage runner class or function that requires explicit configuration instances
- **AND** its top-level imports SHALL exclude heavy dependencies (e.g., vLLM, tokenizer models) until runtime methods are invoked.

### Requirement: Configuration Layer Validation
Stage configuration layering SHALL surface misconfigurations and support explicit clearing of optional fields.

#### Scenario: Unknown keys error
- **WHEN** `StageConfigBase.update_from_file` reads a configuration mapping containing keys not present on the dataclass
- **THEN** it SHALL raise a `ValueError` (or structured error) that lists the offending keys and the file path
- **AND** the error SHALL log or emit telemetry so operators understand the failure.

#### Scenario: Explicit clears
- **WHEN** an operator passes an empty string CLI argument (e.g., `--splade-model-dir ""`) or YAML `null`
- **THEN** `EmbedCfg`/`ChunkerCfg` SHALL treat the value as an explicit request to clear the field
- **AND** the resulting configuration SHALL set the field to `None` and record the override
- **AND** the manifest snapshot SHALL reflect the cleared value.

### Requirement: Scoped Runtime Context
DocParsing runtime SHALL avoid process-wide mutable singletons for HTTP sessions and telemetry.

#### Scenario: Stage-scoped HTTP session
- **WHEN** the embedding runtime requires an HTTP client
- **THEN** it SHALL obtain the session via a stage-scoped factory or context manager
- **AND** headers or timeouts applied by one stage SHALL NOT leak into another stage executed in the same process
- **AND** tests SHALL be able to inject a fake factory without patching module globals.

#### Scenario: Telemetry lifecycle
- **WHEN** a stage runner starts and finishes
- **THEN** it SHALL create a telemetry context object that exposes `log_success`/`log_failure`
- **AND** the context SHALL be disposed automatically (context manager or explicit close) even on exceptions
- **AND** manifest helpers SHALL require an explicit telemetry handle instead of reading `_STAGE_TELEMETRY`.

### Requirement: Bootstrap Error Reporting
Stage CLIs SHALL surface bootstrap failures so operators and automation can react.

#### Scenario: Missing prerequisite directories
- **WHEN** `data_doctags` or `data_chunks` raises during CLI bootstrap
- **THEN** the CLI SHALL log the failure with stage context and details about the missing path
- **AND** it SHALL exit with a non-zero status
- **AND** no work SHALL proceed after the failure is recorded.

#### Scenario: Successful bootstrap remains unchanged
- **WHEN** the bootstrap succeeds
- **THEN** the CLI SHALL continue into the existing runtime flow without new prompts
- **AND** the success path SHALL emit the same telemetry events as before (aside from added run metadata)
- **AND** the CLI usage (`--help`) SHALL remain unchanged for existing scripts.

## MODIFIED Requirements

None.

## REMOVED Requirements

None.

## RENAMED Requirements

None.
