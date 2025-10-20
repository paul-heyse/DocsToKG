# Add Parquet Vector Output Support to DocParsing

## Why
Operators rely on `docparse embed --format` to negotiate vector artifacts, but the runtime only emits JSONL despite advertising Parquet. Downstream services (HybridSearch, analytics) expect columnar outputs for faster ingestion and compression, so the discrepancy blocks parquet-based workflows and erodes trust in CLI help and docs.

## What Changes
- Implement a production-ready Parquet writer in `DocParsing.embedding.runtime` that mirrors JSONL durability (atomic writes, crash simulation hooks, manifest integration).
- Plumb format awareness through discovery, planning, resume, and validation paths so manifests, quarantine logic, and `--validate-only` work for both JSONL and Parquet (including format-mismatch regeneration safeguards).
- Extend embedding CLI/config/docs to truthfully advertise selectable formats and document the new behavior, including dependency guidance (`pyarrow`).
- Teach HybridSearch ingestion utilities and fixtures to consume parquet vectors without regressions for existing JSONL datasets while rejecting mixed-format corpora with clear errors.
- Expand unit/integration tests to cover both formats, including manifest resume, validation, and downstream ingestion.

## Impact
- **Affected specs:** docparsing, hybrid-search.
- **Affected code:** `src/DocsToKG/DocParsing/embedding/{cli.py,config.py,runtime.py}`, `core/{cli.py,discovery.py,planning.py}`, resume/validation helpers, manifest writers, fixtures under `tests/docparsing/**`, HybridSearch ingestion (`src/DocsToKG/HybridSearch/{pipeline.py,service.py}` and supporting scripts), and shared docs (`DocParsing/README.md`, AGENTS runbooks).
- **New dependencies:** reuse existing `pyarrow` optional extra; ensure graceful error when missing.
