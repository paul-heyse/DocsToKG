# DocParsing Specification Deltas

## ADDED Requirements

### Requirement: Embedding Format Negotiation
DocParsing embedding runs SHALL emit vector artifacts whose file suffix and serialization match the requested format (`jsonl` or `parquet`) while preserving schema contents and manifest bookkeeping.

#### Scenario: Default JSONL Emission
- **GIVEN** an operator runs `docparse embed` without providing a format override
- **WHEN** the stage completes successfully
- **THEN** each vectors file SHALL be written as `<doc>.vectors.jsonl` containing the existing JSONL schema
- **AND** manifest entries SHALL record `vector_format: "jsonl"` in the configuration snapshot

#### Scenario: Parquet Opt-in
- **GIVEN** an operator sets `--format parquet` (or config/env equivalent)
- **WHEN** vectors are written
- **THEN** each vectors file SHALL be written atomically as `<doc>.vectors.parquet`
- **AND** the payload SHALL contain the same logical fields (BM25, SPLADE, dense vector, metadata) encoded as Parquet columns
- **AND** manifests SHALL reference the parquet path and record `vector_format: "parquet"`

#### Scenario: Missing Dependency Error
- **GIVEN** `pyarrow` is not importable
- **WHEN** a run requests `--format parquet`
- **THEN** the CLI SHALL exit with a validation error describing the missing dependency and how to install it

### Requirement: Embedding Format-Aware Validation
DocParsing validation, planning, and resume flows SHALL respect the requested vector format to avoid stale outputs or false positives.

#### Scenario: Validate-Only Reads Parquet
- **GIVEN** a corpus with `<doc>.vectors.parquet`
- **WHEN** `docparse embed --validate-only --format parquet` executes
- **THEN** the validator SHALL stream parquet rows, enforce the vector schema, and report success without rewriting files

#### Scenario: Resume Skips Format-Specific Outputs
- **GIVEN** a prior parquet run succeeded and manifests contain `vector_format: "parquet"`
- **WHEN** a follow-up run uses `--resume --format parquet`
- **THEN** unchanged documents SHALL be skipped using the parquet hashes without rewriting JSONL files

#### Scenario: Format Mismatch Forces Regeneration
- **GIVEN** a manifest entry records `vector_format: "jsonl"`
- **WHEN** an operator re-runs `docparse embed --format parquet --resume`
- **THEN** the stage SHALL regenerate vectors for affected documents instead of skipping, and the updated manifest SHALL reference the new parquet outputs

#### Scenario: Planner Uses Correct Suffix
- **GIVEN** `docparse plan embed --format parquet` is invoked
- **WHEN** the plan enumerates process/skip entries
- **THEN** each output path preview SHALL end with `.vectors.parquet`
- **AND** the same command with `--format jsonl` SHALL end with `.vectors.jsonl`
