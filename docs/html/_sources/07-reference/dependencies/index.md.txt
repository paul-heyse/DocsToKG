# 1. Dependency Reference

DocsToKG targets modern Python runtimes and relies on a focussed set of libraries for
ontology management, hybrid search, and document processing. Use this reference when
auditing environments or troubleshooting dependency conflicts.

## 2. Runtime Requirements

- **Python**: 3.12 or newer (see `pyproject.toml`)
- **pip**: 23.0+ recommended for editable installs
- **Optional GPU stack**: CUDA 12.9 for the pinned FAISS and Torch wheels in `requirements.in`

## 3. Core Python Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `rdflib` | RDF parsing and serialization | Required by ontology validators |
| `oaklib`, `ols-client`, `ontoportal-client` | Ontology retrieval from community endpoints | Power the ontology download CLI |
| `bioregistry` | Identifier resolution and metadata enrichment | Cached under `~/.data/ontology-fetcher` |
| `owlready2`, `pronto`, `arelle` | Validation backends for ontology formats | Selectively enabled via CLI flags |
| `requests`, `pooch`, `pystow` | HTTP fetching, asset caching, storage paths | Shared across downloader and content ingestion |
| `pyyaml` | Configuration loading for pipelines | Used by both ingestion and ontology tooling |

## 4. GPU and Numerical Stack

The project keeps GPU builds separate to shorten cold installs:

- `torch==2.8.0+cu129` – pinned for CUDA 12.9 support.
- `vllm==0.11.0rc2.dev429+ga6049be73.cu129` – leveraged by DocParsing accelerators.
- `faiss-1.12.0-py3-none-any.whl` – reference wheel located in `docs/07-reference/faiss/resources/`.

Install these with `pip install -r requirements.in` after activating your environment.

## 5. Optional Tooling

| Tool | Why | Install |
|------|-----|---------|
| `vale` | Documentation prose linting | `brew install vale` or download from vale.sh |
| `mkdocs-material` | Alternative static site builds (experimental) | `pip install mkdocs-material` |
| `sphinx-rtd-theme` | HTML theme for Sphinx builds | Provided in `docs/build/sphinx/requirements.txt` |

## 6. External Services

- **Pyalex API** – Source for publication metadata (`DocsToKG.ContentDownload`).
- **BioPortal / OBO / OLS endpoints** – Ontology fetch targets configured via `sources.yaml`.
- **Object storage** – Persistent location for ontology artifacts and FAISS snapshots.

Refer to `docs/06-operations/index.md` for provisioning guidance and to `docs/02-setup/index.md` for step-by-step installation instructions.
