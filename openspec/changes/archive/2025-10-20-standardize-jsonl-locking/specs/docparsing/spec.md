# DocParsing Specification Deltas

## ADDED Requirements

### Requirement: Standardised File Locking
DocParsing file writes SHALL coordinate through `filelock.FileLock` helpers derived from the target path, replacing bespoke `.lock` sentinel logic.

#### Scenario: Lock Acquisition Guards Telemetry Append
- **WHEN** two processes attempt to append to the same manifest simultaneously
- **THEN** the second process SHALL block on the shared `FileLock`
- **AND** the manifest SHALL remain consistent after both appends complete

#### Scenario: Manual Lock Sentinels Rejected
- **WHEN** new DocParsing code attempts to create or manage `.lock` files directly
- **THEN** the CI guard SHALL fail, instructing contributors to use the shared lock helper

### Requirement: Library-Backed JSONL Iteration
DocParsing streaming readers SHALL delegate to a library-backed iterator while preserving existing interfaces (`iter_jsonl`, `iter_jsonl_batches`) and semantics (`skip_invalid`, `max_errors`, `start`, `end`).

#### Scenario: Skip Invalid Rows Honors Error Budget
- **GIVEN** a JSONL file containing three malformed rows
- **WHEN** `iter_jsonl(..., skip_invalid=True, max_errors=3)` executes
- **THEN** the iterator SHALL yield all valid rows
- **AND** SHALL raise once a fourth malformed row is encountered

#### Scenario: Batch Iterator Respects Bounds
- **GIVEN** `iter_jsonl_batches(paths, batch_size=128, start=256, end=512)`
- **WHEN** the iterator is consumed
- **THEN** all yielded rows SHALL have indices in `[256, 512)`
- **AND** the final batch SHALL contain only the remaining subset

### Requirement: Telemetry Writer Abstraction
Stage telemetry SHALL append manifest entries through an injectable writer that encapsulates locking and JSONL append behavior, defaulting to `FileLock` + `jsonl_append_iter(..., atomic=True)`.

#### Scenario: Default Writer Uses Lock + Atomic Append
- **WHEN** telemetry emits a manifest entry without an override
- **THEN** it SHALL acquire the shared `FileLock`
- **AND** SHALL append the row atomically using `jsonl_append_iter`

#### Scenario: Custom Writer Supported
- **GIVEN** a telemetry sink initialised with a custom writer callable
- **WHEN** manifest entries are emitted
- **THEN** the callable SHALL be invoked exactly once per append with the manifest path and payload

## MODIFIED Requirements

### Requirement: JSONL Append Atomicity
JSONL append helpers SHALL continue to provide atomic writes while delegating iteration to the library-backed adapter.

#### Scenario: Append Failure Leaves File Intact
- **WHEN** `jsonl_append_iter(..., atomic=True)` encounters an I/O failure mid-write
- **THEN** the existing JSONL file SHALL remain unchanged
- **AND** the helper SHALL surface the exception to the caller

## REMOVED Requirements

None.
