## MODIFIED Requirements

### Requirement: Validation and Calibration Suite
An automated validation harness SHALL cover ingest integrity (field presence, dimension checks), dense self-hit accuracy (≥0.95 @1 for IVF or 1.00 for Flat), sparse relevance sanity (≥90% self-match @10), namespace filtering, pagination stability, fusion efficacy, highlight packaging, and calibration sweeps for `nprobe`/PQ parameters. The harness SHALL support both lightweight synthetic fixtures and a curated real-vector dataset that includes full-dimension Qwen embeddings and SPLADE weights, verify that chunk/vector UUIDs align, ensure BM25/SPLADE features deserialize without loss, and emit timestamped JSON + text reports under `reports/validation/<timestamp>/` for all runs.

#### Scenario: Run end-to-end validation on sample corpus
- **GIVEN** the lightweight JSONL dataset of sparse and dense features
- **WHEN** the validation command executes
- **THEN** all ingest, search, fusion, pagination, and backup checks pass with thresholds defined above
- **AND** calibration results are recorded for operational tuning
- **AND** the harness writes both JSON and human-readable summaries under `reports/validation/<timestamp>/`

#### Scenario: Run validation on real embeddings fixture
- **GIVEN** the curated real-vector dataset with paired chunk/vector JSONL and query expectations
- **WHEN** the validation command executes with the dataset flag set to the real fixture
- **THEN** ingest integrity confirms 2560-d embeddings, UUID alignment, and SPLADE/BM25 term parity
- **AND** dense self-hit, fusion, and namespace checks achieve thresholds against the real corpus
- **AND** calibration results and report artifacts are written for audit

## ADDED Requirements

### Requirement: Real Vector Fixture Generation
The system SHALL provide deterministic tooling that samples a configurable set of documents from production chunk/vector artifacts, strips or redacts sensitive metadata, and emits a reproducible real-vector fixture under `Data/HybridScaleFixture/` containing aligned chunk JSONL, vector JSONL, a manifest with per-document metadata, and canonical query expectations. The process SHALL record the source artifact hashes and random seed used for sampling to preserve auditability.

#### Scenario: Build reproducible real-vector fixture
- **GIVEN** the production `Data/ChunkedDocTagFiles` and `Data/Vectors` directories and a target sample size
- **WHEN** the fixture builder script executes
- **THEN** it validates input availability, samples the configured number of documents with a fixed seed, redacts configured metadata fields, and writes chunk/vector files plus `manifest.json` and `queries.json` into `Data/HybridScaleFixture/`
- **AND** the manifest records the source file hashes, sample seed, and any redacted fields
- **AND** rerunning the script with the same seed produces byte-identical fixtures

### Requirement: Real Vector Regression Execution
The regression suite SHALL support an opt-in `real-vectors` execution mode (exposed via pytest marker/CLI flag and documented make/tox targets) that ingests the curated fixture, exercises re-ingest/delete flows, runs the public API and validation harness, and asserts that validation reports are emitted under `reports/validation/<timestamp>/`. CI SHALL run this suite on a scheduled cadence, archive the generated reports as build artifacts, and fail the job if ingest, retrieval, fusion, or serialization checks do not meet thresholds.

#### Scenario: Execute real-vector regression suite locally
- **GIVEN** the real-vector fixture is present on disk
- **WHEN** a developer runs the documented command (e.g., `pytest --real-vectors tests/test_hybrid_search_real.py`)
- **THEN** the suite ingests the fixture, verifies vector dimensions, UUID alignment, BM25/SPLADE parity, and passes re-ingest/delete, API, validator, and serialization tests without manual setup beyond enabling the flag
- **AND** skipped markers are reported clearly when the flag is not provided

#### Scenario: Archive validation artifacts in CI
- **GIVEN** the scheduled CI job executes the real-vector suite
- **WHEN** the validation harness runs as part of the tests
- **THEN** the job uploads the produced `reports/validation/<timestamp>/` directory as a build artifact
- **AND** the pipeline fails if any validation report is marked as failed or missing

### Requirement: FAISS Destructive Operation Fallback
The system SHALL detect when GPU-promoted FAISS indexes lack support for `remove_ids` (or equivalent destructive operations), transparently downgrade the affected index to CPU for the duration of the operation, perform the deletion, and re-promote to GPU without data loss. Metrics SHALL reflect the downgrade event, and `ntotal` SHALL update consistently regardless of execution path.

#### Scenario: Remove vectors when GPU index lacks remove_ids
- **GIVEN** a FAISS index running on GPU that raises `remove_ids not implemented`
- **WHEN** the ingestion pipeline re-ingests a batch that requires deleting existing vector IDs
- **THEN** the manager downgrades the index to CPU, removes the IDs, and re-promotes to GPU
- **AND** the deletion metrics and `ntotal` values match the removed chunk count
- **AND** subsequent searches succeed without requiring a service restart
