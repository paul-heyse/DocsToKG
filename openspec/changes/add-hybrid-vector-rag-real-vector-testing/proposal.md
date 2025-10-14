## Why

Our hybrid search suite only exercises ingestion, validation, and retrieval against synthetic 16‑dimension embeddings. Real production vectors already exist in `Data/ChunkedDocTagFiles` and `Data/Vectors`, yet no automated job loads them. When running the current tests with native FAISS bindings, the ingestion pipeline fails because promoted GPU indexes do not implement `remove_ids`, masking bugs that will surface once real corpora are connected. We need a scoped change that formalizes real-vector regression coverage and hardens the FAISS manager so the stack can be validated with production-like artifacts.

## What Changes

- Real fixture curation: deterministic tooling to sample, sanitize, and publish a reproducible dataset under `tests/data/real_hybrid_dataset/` with paired chunk/vector JSONL and query expectations.
- Test harness updates: gated pytest marker/flag for real-vector suites, extended ingestion/reingest/API/assertion coverage, validator invocation with report verification, and serialization roundtrip tests.
- FAISS manager hardening: graceful CPU fallback for destructive operations (`remove_ids`) when GPU indexes lack the method, with regression coverage that exercises the downgraded path and validates `ntotal`.
- Validation CLI and docs: default dataset pointing at the real fixture, updated runbook instructions, and calibration narrative aligned to production embeddings.
- CI / selective execution: pytest configuration that skips vendored FAISS/vLLM suites by default, documented make/tox targets for hybrid testing, and a scheduled job that enables real-vector coverage and publishes validation artifacts.

## Impact

- Affected specs: `hybrid-search`
- Affected code: `DocsToKG/HybridSearch/dense.py`, `DocsToKG/HybridSearch/validation.py`, real-vector test modules, fixture tooling under `scripts/`, `pytest.ini` / test runners, docs runbooks.

## Current State

- Test fixtures rely on hashed toy embeddings; calibration never touches real SPLADE/Qwen payloads.
- `ChunkIngestionPipeline.delete_chunks` triggers `RuntimeError: remove_ids not implemented` when FAISS indexes run on GPU.
- Validation CLI lacks a baked real dataset and there is no automated assertion on generated reports.
- Pytest attempts to collect upstream FAISS/vLLM test suites, slowing runs and producing unrelated failures.

## Success Metrics & Acceptance

- Real-vector regression suite finishes within 5 minutes locally with <2 GB RSS.
- Real-vector ingestion, re-ingest, API, validator, and serialization tests pass without manual intervention.
- FAISS removal flows no longer raise `remove_ids not implemented`; CPU fallback is covered by automated tests.
- Validation artifacts include real document IDs and calibration metrics and are archived by CI jobs.
- Documentation covers fixture refresh steps and test execution guidance.

## Open Questions

1. Do we need additional sampling or redaction to remove PHI/licensed metadata before committing fixtures?
2. Should the real dataset live in git (curated sample) or be generated on demand from `Data/` to prevent drift?
3. Are there GPU deployments where we must preserve the promoted index even during delete operations, or is CPU fallback sufficient for now?
