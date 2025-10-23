# 1. DocsToKG

DocsToKG turns raw documents into searchable knowledge artefacts by combining document acquisition, Docling-based parsing, ontology downloads, and a FAISS-backed hybrid search engine.

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 2. Overview

- **Content acquisition**: Fetch PDFs and metadata from external services such as Pyalex.
- **Doc parsing & chunking**: Convert documents into DocTags, chunked Markdown, and embeddings using Docling pipelines.
- **Hybrid search**: Fuse BM25, SPLADE, and FAISS dense retrieval with configurable ranking.
- **Ontology management**: Download and validate ontologies for consistent terminology and enrichment.
- **Documentation tooling**: Automated scripts generate API docs, run validations, and perform link checks.

See `docs/01-overview/` for a high-level introduction and `docs/03-architecture/` for subsystem details.

## 3. Quick Start

```bash
git clone https://github.com/paul-heyse/DocsToKG.git
cd DocsToKG

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install requirements.txt

```

> ℹ️ The repository ships with a preconfigured `.envrc` that adds `.venv/bin` to `PATH`,
> exports `VIRTUAL_ENV`, and appends `src/` to `PYTHONPATH`. Any shell (or AI agent) that
> runs commands via `direnv exec . …` automatically picks up the project virtual environment
> without additional configuration.
