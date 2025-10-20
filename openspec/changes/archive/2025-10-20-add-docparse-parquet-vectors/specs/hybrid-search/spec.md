# HybridSearch Specification Deltas

## ADDED Requirements

### Requirement: HybridSearch Accepts Multiple DocParsing Vector Formats
HybridSearch ingestion SHALL ingest DocParsing vector artifacts produced in JSONL or Parquet formats without requiring downstream operators to normalize files manually.

#### Scenario: JSONL Backward Compatibility
- **GIVEN** an ingestion job references `*.vectors.jsonl` files
- **WHEN** the job processes the dataset
- **THEN** vectors SHALL be parsed using the existing JSONL reader
- **AND** the ingestion summary SHALL report success without regressions

#### Scenario: Parquet Dataset Ingestion
- **GIVEN** an ingestion job references `*.vectors.parquet` files
- **WHEN** HybridSearch loads the dataset
- **THEN** the pipeline SHALL read the parquet files, extract sparse/dense features, and materialize the same in-memory structures as JSONL ingestion

#### Scenario: Mixed Dataset Guardrail
- **GIVEN** a dataset mixes JSONL and Parquet vector files for the same namespace
- **WHEN** the ingestion job starts
- **THEN** HybridSearch SHALL raise a descriptive error indicating the mismatched formats so operators can reconcile the corpus before retrying
