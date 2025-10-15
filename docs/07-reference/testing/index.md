# 1. Testing Strategies

Testing spans ingestion utilities, ontology download workflows, and hybrid search correctness.

## 2. Test Suites

- `tests/test_hybrid_search.py` – Core unit coverage for HybridSearch service contracts.
- `tests/test_hybrid_search_real_vectors.py` – Optional vector-backed tests (enable with `--real-vectors` marker).
- `tests/test_pipeline_behaviour.py` – End-to-end validation of document processing pipelines.
- `tests/ontology_download/` – Validates ontology resolver logic and manifest handling.

Run locally with:

```bash
pytest -q
pytest -m real_vectors --real-vectors  # requires fixture data
```

## 3. Documentation Validation

Integrate documentation checks into CI:

```bash
python docs/scripts/validate_docs.py
python docs/scripts/check_links.py --timeout 10
python docs/scripts/generate_api_docs.py
```

Warnings should gate merges for documentation changes.

## 4. Benchmark Datasets

- `tests/data/hybrid_dataset.jsonl` – Reference dataset for recall/latency validation.
- `tests/data/ontology/` – Fixture ontologies for deterministic download/validation tests.

Keep these datasets updated when schemas evolve, and document changes in `docs/05-development/index.md`.

## 5. Continuous Integration

Recommended steps for CI pipelines:

1. Install project dependencies (`pip install -e .`).
2. Install GPU wheels conditionally (skip on CPU-only runners).
3. Run `pytest` with markers to skip heavy suites on PR builds.
4. Execute documentation validation scripts.
5. Publish coverage reports or store as build artifacts.

See `docs/06-operations/index.md` for pipeline templates and scheduler guidance.
