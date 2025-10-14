# Setup Guide

Follow this guide to configure DocsToKG on a local workstation or development server.

## Prerequisites

- **Operating system**: Linux or macOS preferred (Windows users should use WSL2).
- **Python**: 3.12 or newer (see `pyproject.toml`).
- **Memory**: 16 GB RAM recommended for parsing and validation workloads.
- **GPU (optional)**: CUDA 12.9+ for FAISS and PyTorch acceleration (required when installing `requirements.in`).
- **Git**: 2.30 or newer.

> ℹ️  Direnv is supported via `.envrc`, but optional. Install with `brew install direnv` or follow instructions at [direnv.net](https://direnv.net).

## 1. Clone the Repository

```bash
git clone https://github.com/paul-heyse/DocsToKG.git
cd DocsToKG
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

If you use direnv, run `direnv allow` after activation so `.envrc` can automatically load the environment next time.

## 3. Install Dependencies

```bash
# Core project dependencies (editable install for development)
pip install -e .

# Optional GPU/ML stack (requires CUDA 12.9 toolchain)
pip install -r requirements.in
```

Install documentation tooling when you plan to regenerate docs:

```bash
pip install -r docs/build/sphinx/requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file in the repository root (or export variables manually):

```bash
cat > .env <<'EOF'
# Storage locations
DOCSTOKG_DATA_ROOT=./Data
HYBRID_SEARCH_CONFIG=./config/hybrid_config.json
ONTOLOGY_FETCHER_CONFIG=./configs/sources.yaml

# Third-party credentials
PA_ALEX_KEY=your-pyalex-api-key
BIOPORTAL_API_KEY=optional-bioportal-token
EOF
```

Load the file with `source .env` or rely on direnv to ingest it automatically.

## 5. Prepare Local Directories

```bash
mkdir -p Data/DocTagsFiles Data/ChunkedDocTagFiles Data/Embeddings artifacts logs
```

These directories store DocTags inputs, chunked outputs, embedding payloads, FAISS snapshots, and log output.

## 6. Verify the Installation

Run the automated test suite:

```bash
pytest -q
```

Generate and validate documentation:

```bash
python docs/scripts/generate_api_docs.py
python docs/scripts/validate_docs.py
```

For asynchronous link checks (optional):

```bash
python docs/scripts/check_links.py --timeout 10
```

## 7. Optional: Build Sphinx HTML Docs

```bash
python docs/scripts/build_docs.py --format html
open docs/build/_build/html/index.html  # macOS
```

## Next Steps

- Ingest sample documents using the content downloaders (`DocsToKG.ContentDownload`).
- Convert assets into chunked DocTags and embeddings (`DocsToKG.DocParsing`).
- Populate a FAISS index and serve hybrid search queries (`DocsToKG.HybridSearch`).

Refer to the **Architecture Guide** (`docs/03-architecture/index.md`) for subsystem details and the **Operations Guide** (`docs/06-operations/index.md`) for day-two workflows.
