## 1. Real Vector Fixture
- [ ] 1.1 Implement `scripts/build_real_hybrid_fixture.py` that validates source directories, samples documents with a fixed seed, redacts configured metadata fields, and writes aligned chunk/vector JSONL plus `manifest.json` and `queries.json`.
- [ ] 1.2 Add automated checks in the script for UUID alignment, embedding dimensionality, and hash recording so reruns with the same seed are deterministic.
- [ ] 1.3 Materialize `tests/data/real_hybrid_dataset/` (chunk files, vector files, manifest, queries, README) and ensure the README documents refresh workflow and redaction policy.

## 2. Hybrid Search Test Coverage
- [ ] 2.1 Introduce a `real_vectors` pytest marker, CLI flag (`--real-vectors`), and default skip configuration in `pytest.ini`.
- [ ] 2.2 Create dedicated real-vector regression tests that load the fixture, assert vector dimensionality, UUID matches, BM25/SPLADE parity, and re-ingest/delete behaviour (including FAISS fallback).
- [ ] 2.3 Extend API and validator tests to execute against the real fixture, verify top-k expectations, and assert that reports are written under `reports/validation/<timestamp>/`.
- [ ] 2.4 Update developer tooling (e.g., `Makefile`, `tox.ini`, or README) with commands for synthetic-only and real-vector test runs.

## 3. FAISS Manager Hardening
- [ ] 3.1 Implement CPU fallback and re-promotion logic for destructive operations (`remove_ids`) including observability hooks/metrics.
- [ ] 3.2 Add unit/integration tests that force the fallback path, validate `ntotal` changes, and ensure subsequent searches succeed.
- [ ] 3.3 Confirm serialization/restore handles 2560-d vectors and large payloads without truncation or ordering drift.

## 4. Tooling & CI Integration
- [ ] 4.1 Update `DocsToKG.HybridSearch.validation` CLI to default to the real fixture path while retaining overrides, and surface a `--dataset` help example for both fixtures.
- [ ] 4.2 Provide make/tox targets (or equivalent) for running synthetic-only, real-only, and combined suites.
- [ ] 4.3 Configure CI/nightly job to enable the real-vector flag, collect `reports/validation` outputs, and fail on missing/failed reports.
- [ ] 4.4 Ensure CI uploads validation artifacts and links them from job summaries for manual inspection.
