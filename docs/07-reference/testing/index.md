# 1. Testing Strategies

DocsToKG ships an extensive pytest suite that spans content download resiliency, DocParsing pipelines, ontology workflows, and hybrid search quality. The sections below highlight the primary test packages, key markers, and supporting validation scripts.

## 2. Test Suite Map

| Path | Focus | Notes |
|------|-------|-------|
| `tests/content_download/` | Resolver pipeline, networking retries, manifest sinks. | Requires optional dependencies (`pyalex`, `beautifulsoup4`, `trafilatura`). Tests fall back to stubs where available. |
| `tests/docparsing/` | Chunker CLI, embedding pipeline, docstring and NAVMAP validation. | Includes smoke tests for `docparse chunk` / `docparse embed` and the underlying `_chunking` / `_embedding` runtimes. |
| `tests/hybrid_search/test_suite.py` | Hybrid search ranking, fusion, FAISS integration. | Uses custom markers `real_vectors` and `scale_vectors` for GPU-backed assertions. |
| `tests/ontology_download/` | CLI, configuration, storage backends, validation flows. | Skips gracefully when `pydantic-settings` or fsspec extras are missing. |
| `tests/pipeline/` | End-to-end orchestration and retry controllers. | Marked with `@pytest.mark.integration` for slower runs. |

Run the default unit suite with:

```bash
direnv exec . pytest -q
```

Optional suites:

- `direnv exec . pytest -m real_vectors --real-vectors` – Loads FAISS-backed fixtures and real embeddings.
- `direnv exec . pytest -m scale_vectors --scale-vectors` – Stress tests on large vector sets (GPU recommended).
- `direnv exec . pytest -m "not integration"` – Skip network-heavy integration checks on pull requests.

Markers are defined in `pyproject.toml` under `[tool.pytest.ini_options]`.

## 3. Documentation & Annotation Validation

Keep generated documentation in sync with the codebase before merging:

```bash
direnv exec . python docs/scripts/validate_docs.py
direnv exec . python docs/scripts/generate_api_docs.py
direnv exec . python docs/scripts/check_links.py --timeout 10
direnv exec . python docs/scripts/validate_code_annotations.py
```

`validate_code_annotations.py` enforces NAVMAP ordering and docstring completeness, reducing surprises in `docs/04-api/` rebuilds.

## 4. Benchmark & Fixture Data

- `tests/data/hybrid_dataset.jsonl` – Baseline for hybrid search recall and latency comparisons.
- `tests/data/ontology/` – Controlled ontologies for checksum, normalization, and storage tests.
- `tests/fixtures/` – Synthetic DocTags, chunk manifests, and resolver payloads used across pipelines.

When schemas evolve (for example new manifest fields), regenerate or update these fixtures and document the change in `docs/05-development/index.md`.

## 5. Continuous Integration Recommendations

1. `./scripts/bootstrap_env.sh` (installs `.venv` with bundled wheels).
2. `direnv exec . python docs/scripts/validate_code_annotations.py`.
3. `direnv exec . python docs/scripts/validate_docs.py`.
4. `direnv exec . pytest -m "not (real_vectors or scale_vectors)"`.
5. Publish coverage reports or store `pytest` JUnit XML for diagnostics.

Nightly/weekly pipelines should additionally exercise:

- `direnv exec . pytest -m real_vectors --real-vectors`.
- `direnv exec . pytest -m scale_vectors --scale-vectors`.
- `direnv exec . python -m DocsToKG.HybridSearch.validation` for regression dashboards.

See `docs/06-operations/index.md` for scheduler templates and operational checklists.
