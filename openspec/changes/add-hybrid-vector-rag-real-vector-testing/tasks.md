## 1. Real Vector Fixture
- [x] 1.1 Implement `scripts/build_real_hybrid_fixture.py` that validates source directories, samples documents with a fixed seed, redacts configured metadata fields, and writes aligned chunk/vector JSONL plus `manifest.json` and `queries.json`.
- [x] 1.2 Add automated checks in the script for UUID alignment, embedding dimensionality, and hash recording so reruns with the same seed are deterministic.
- [x] 1.3 Materialize `Data/HybridScaleFixture/` (chunk files, vector files, manifest, queries, README) and ensure the README documents refresh workflow and redaction policy.

## 2. Hybrid Search Test Coverage
- [x] 2.1 Introduce a `real_vectors` pytest marker, CLI flag (`--real-vectors`), and default skip configuration in `pytest.ini`.
- [x] 2.2 Create dedicated real-vector regression tests that load the fixture, assert vector dimensionality, UUID matches, BM25/SPLADE parity, and re-ingest/delete behaviour (including FAISS fallback).
- [x] 2.3 Extend API and validator tests to execute against the real fixture, verify top-k expectations, and assert that reports are written under `reports/validation/<timestamp>/`.
- [x] 2.4 Update developer tooling (e.g., scripts) with commands for synthetic-only and real-vector test runs.

## 3. FAISS Manager Hardening
- [x] 3.1 Implement CPU fallback and re-promotion logic for destructive operations (`remove_ids`) including observability hooks/metrics.
- [x] 3.2 Add unit/integration tests that force the fallback path, validate `ntotal` changes, and ensure subsequent searches succeed.
- [x] 3.3 Confirm serialization/restore handles 2560-d vectors and large payloads without truncation or ordering drift.

## 4. Tooling & CI Integration
- [x] 4.1 Update `DocsToKG.HybridSearch.validation` CLI to default to the real fixture path while retaining overrides, and surface a `--dataset` help example for both fixtures.
- [x] 4.2 Provide tooling for running synthetic-only, real-only, and combined suites.
- [x] 4.3 Configure CI/nightly automation hook by supplying `scripts/run_real_vector_ci.py` which enables the real-vector flag, collects validation outputs, and propagates failures.
- [x] 4.4 Ensure CI uploads validation artifacts and links them from job summaries for manual inspection.
