# 1. DocsToKG

DocsToKG turns raw documents into searchable knowledge artefacts by combining document acquisition, Docling-based parsing, ontology downloads, and a FAISS-backed hybrid search engine.

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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

## 4. Example Usage

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

## 5. Parallel Execution

Use the ``--workers`` flag to enable bounded parallelism when downloading content via
the OpenAlex pipeline:

```bash
# Sequential (default, safest)
python -m DocsToKG.ContentDownload.download_pyalex_pdfs --workers 1 --topic "oncology" --year-start 2020 --year-end 2024

# Parallel (2-5x throughput)
python -m DocsToKG.ContentDownload.download_pyalex_pdfs --workers 3 --topic "oncology" --year-start 2020 --year-end 2024
```

**Recommendations:**

- Start with ``--workers=3`` for production workloads.
- Monitor rate limit compliance with resolver APIs while scaling.
- Higher values (>5) may overwhelm resolver providers despite per-resolver rate limiting.
- Each worker maintains its own HTTP session with retry logic.

### Additional CLI Flags

- ``--dry-run``: compute resolver coverage without writing files.
- ``--resume-from <manifest.jsonl>``: skip works already recorded as successful.
- ``--extract-html-text``: save plaintext alongside HTML fallbacks (requires ``trafilatura``).
- ``--enable-resolver openaire`` (and ``hal``/``osf``): opt into additional EU/preprint resolvers.
- ``--resolver-config config.yaml``: load advanced options such as
  ``max_concurrent_resolvers`` and ``resolver_head_precheck`` (see
  ``docs/resolver-configuration.md``).

### Resolver Enhancements

- **Zenodo and Figshare support** – the default resolver order now queries
  Zenodo and Figshare APIs, expanding coverage of institutional repositories.
- **Bounded intra-work concurrency** – configure
  ``max_concurrent_resolvers`` to execute independent resolvers in parallel
  while preserving per-resolver rate limits.
- **HEAD pre-check filtering** – enable ``enable_head_precheck`` (default) to
  perform lightweight HEAD requests that skip HTML and zero-byte responses
  before downloading.
- **Migration resources** – see ``docs/migration-modularize-resolvers.md`` for
  a full list of import path changes, configuration defaults, and testing tips.

### Troubleshooting Content Downloads

- **Partial files remain (``*.part``)** – rerun with fewer workers or check network
  stability before retrying.
- **Resolver rate limit warnings** – lower ``--workers`` or increase per-resolver
  ``resolver_min_interval_s``.
- **High memory usage** – reduce ``--workers`` to limit in-flight downloads.

### Logging and Exports

- Attempts log to JSONL by default. Convert to CSV with
  ``python scripts/export_attempts_csv.py attempts.jsonl attempts.csv``.
- Alternatively, use ``jq``:
  ``jq -r '[.timestamp,.work_id,.status,.url] | @csv' attempts.jsonl > attempts.csv``.

## 6. Development

### 6.1 Development Workflow

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

## 7. Project Structure

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

## 8. DocParsing Pipeline CLI

The DocParsing toolkit now exposes first-class module entrypoints for each stage
of the pipeline. The legacy scripts remain available but emit a
``DeprecationWarning`` on direct invocation.

```bash
# Convert HTML or PDF corpora to DocTags (auto-detects mode when possible)
python -m DocsToKG.DocParsing.cli.doctags_convert --input Data/HTML

# Chunk DocTags with topic-aware coalescence
python -m DocsToKG.DocParsing.cli.chunk_and_coalesce --min-tokens 256 --max-tokens 512

# Generate hybrid embeddings (BM25 + SPLADE + Qwen)
python -m DocsToKG.DocParsing.cli.embed_vectors --resume
```

Use ``--help`` on each command for the full set of flags, including data-root
overrides, resume/force controls, and tokenizer configuration.

## 9. Support & Community

- **Issues**: [GitHub Issues](https://github.com/paul-heyse/DocsToKG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paul-heyse/DocsToKG/discussions)
- **License**: MIT (see [LICENSE](LICENSE))

Read `CONTRIBUTING.md` before submitting changes, and keep documentation in sync with implementation using the provided automation scripts.
