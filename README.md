# DocsToKG

DocsToKG turns raw documents into searchable knowledge artefacts by combining document acquisition, Docling-based parsing, ontology downloads, and a FAISS-backed hybrid search engine.

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

- **Content acquisition**: Fetch PDFs and metadata from external services such as Pyalex.
- **Doc parsing & chunking**: Convert documents into DocTags, chunked Markdown, and embeddings using Docling pipelines.
- **Hybrid search**: Fuse BM25, SPLADE, and FAISS dense retrieval with configurable ranking.
- **Ontology management**: Download and validate ontologies for consistent terminology and enrichment.
- **Documentation tooling**: Automated scripts generate API docs, run validations, and perform link checks.

See `docs/01-overview/` for a high-level introduction and `docs/03-architecture/` for subsystem details.

## Quick Start

```bash
git clone https://github.com/paul-heyse/DocsToKG.git
cd DocsToKG

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
pip install -r requirements.in         # optional GPU / ML stack
pip install -r docs/build/sphinx/requirements.txt  # documentation tooling
```

Create a `.env` file (or export variables manually):

```bash
cat > .env <<'EOF'
DOCSTOKG_DATA_ROOT=./Data
HYBRID_SEARCH_CONFIG=./config/hybrid_config.json
ONTOLOGY_FETCHER_CONFIG=./configs/sources.yaml
PA_ALEX_KEY=replace-with-api-key
EOF
```

Verify the environment:

```bash
pytest -q
python docs/scripts/generate_api_docs.py
python docs/scripts/validate_docs.py
```

## Example Usage

### Hybrid Search Service

```python
from DocsToKG.HybridSearch.retrieval import HybridSearchRequest
from my_project.hybrid import build_hybrid_service  # see docs/06-operations/index.md

request = HybridSearchRequest(query="ontology alignment best practices", page_size=3)
service = build_hybrid_service()
response = service.search(request)
for result in response.results:
    print(result.doc_id, round(result.score, 3), result.highlights)
```

### Ontology Download CLI

```bash
python -m DocsToKG.OntologyDownload.cli pull --spec configs/sources.yaml --force --json
python -m DocsToKG.OntologyDownload.cli validate hp latest
```

More operational examples are available in `docs/06-operations/index.md`.

## Development Workflow

```bash
# Formatting and linting
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/ --strict

# Tests
pytest -q
pytest -m real_vectors --real-vectors  # optional vector-backed suite

# Documentation
python docs/scripts/generate_all_docs.py
python docs/scripts/check_links.py --timeout 10
```

Refer to `docs/05-development/index.md` for contribution guidelines and review checklists.

## Project Structure

```
DocsToKG/
├── src/DocsToKG/                 # Core packages
│   ├── ContentDownload/          # Document acquisition utilities
│   ├── DocParsing/               # Docling pipelines and embedding scripts
│   ├── HybridSearch/             # Hybrid search configuration, storage, retrieval, API
│   └── OntologyDownload/         # Ontology downloader CLI and validators
├── docs/                         # Documentation framework (guides, templates, scripts)
├── tests/                        # Automated tests and fixtures
├── openspec/                     # Spec-driven change proposals and tasks
├── requirements.in               # Optional GPU / ML dependencies
├── pyproject.toml                # Project metadata and tooling configuration
└── docs/scripts/                 # Documentation automation scripts
```

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/paul-heyse/DocsToKG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paul-heyse/DocsToKG/discussions)
- **License**: MIT (see [LICENSE](LICENSE))

Read `CONTRIBUTING.md` before submitting changes, and keep documentation in sync with implementation using the provided automation scripts.
