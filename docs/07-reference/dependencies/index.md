# 1. Dependency Reference

DocsToKG targets modern Python runtimes and relies on a focussed set of libraries for
ontology management, hybrid search, and document processing. Use this reference when
auditing environments or troubleshooting dependency conflicts.

## 2. Runtime Requirements

- **Python**: 3.12 or newer (the `bootstrap_env.sh` script provisions CPython 3.13 inside `.venv`)
- **pip**: 25.x+ recommended; `bootstrap_env.sh` upgrades pip automatically
- **direnv**: 2.32 or newer so `.envrc` can expose the virtualenv PATH and `PYTHONPATH`
- **System libraries**: `git-lfs` (for bundled wheels) and `libopenblas0` (runtime dependency for FAISS)
- **CUDA toolchain** *(optional)*: CUDA 12.9 drivers when running GPU-backed chunking or embedding workflows locally

## 3. Core Python Dependencies

| Subsystem | Key packages | Notes |
|-----------|--------------|-------|
| Ontology download | `oaklib==0.6.23`, `ontoportal-client`, `ols-client`, `bioregistry`, `rdflib`, `owlready2`, `pronto`, `arelle` | Power `DocsToKG.OntologyDownload.*` workflows, including validators that exercise multiple ontology formats. |
| Content acquisition | `pyalex==0.18`, `requests`, `beautifulsoup4`, `trafilatura` *(optional)*, `tenacity`, `pydantic` | Used by `DocsToKG.ContentDownload` resolver pipeline, manifest telemetry, and HTML fallback extraction. |
| Doc parsing & chunking | `docling-core`, `transformers`, `tokenizers`, `sentencepiece`, `pydantic`, `structlog` | Back the DocTags ingestion, hybrid chunker, and telemetry instrumentation. |
| Hybrid search & embeddings | `faiss`, `numpy`, `scikit-learn`, `torch`, `xformers`, `cupy`, `cupy-cuda12x` | Deployed by `DocsToKG.HybridSearch` to build FAISS indexes and run embedding evaluation. |
| Services & CLI | `typer`, `click`, `rich`, `uvicorn`, `fastapi`, `pydantic-settings` | Provide command-line helpers and service endpoints across the project. |

All pins live in `requirements.txt`; use `./scripts/bootstrap_env.sh` to install the curated set together with the project in editable mode.

## 4. GPU and Numerical Stack

Custom wheels are stored under `ci/wheels/` and installed automatically when the environment is bootstrapped:

- `torch==2.8.0+cu129`, `torchvision==0.23.0+cu129`, `torchaudio==2.8.0+cu129`
- `vllm==0.11.0rc2.dev449+g134f70b3e.d20251014.cu129`
- `faiss==1.12.0` (Python wheel)
- `cupy==14.0.0a1` and `cupy-cuda12x==14.0.0a1`

Because these wheels are bundled, installations do **not** hit external package indexes—ensure `git lfs pull` has populated the wheel directory before bootstrapping.

## 5. Optional Tooling

| Tool | Why | Install |
|------|-----|---------|
| `pre-commit` + `ruff` | Match repository linting and formatting defaults | `pip install pre-commit ruff` then `pre-commit install` |
| `vale` | Documentation prose linting used in CI | `brew install vale` or download from [vale.sh](https://vale.sh) |
| `mkdocs-material` | Experimental markdown preview site | `pip install mkdocs-material` |
| `sphinx-rtd-theme` | HTML theme for Sphinx builds | `pip install -r docs/build/sphinx/requirements.txt` |

## 6. External Services

- **OpenAlex (via `pyalex`)** – Primary source for scholarly metadata in `DocsToKG.ContentDownload`.
- **Crossref, Unpaywall, PubMed Central** – Resolver backstops invoked when OpenAlex lacks PDFs.
- **BioPortal / OBO / Ontology Lookup Service** – Ontology fetch targets configured in `DocsToKG/OntologyDownload/settings.py`.
- **Object storage (S3, GCS, or local filesystem)** – Backing store referenced by `DocsToKG.OntologyDownload.storage` and FAISS snapshot utilities.

For deployment-specific provisioning guidance see `docs/06-operations/index.md`; environment setup walkthroughs live in `docs/02-setup/index.md`.
