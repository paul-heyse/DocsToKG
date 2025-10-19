# 1. Setup Guide

Follow this guide to configure DocsToKG on a local workstation or development server.

## 2. Overview

- Provision a Python 3.13 environment with optional CUDA 12.9 support.
- Install the curated dependency set (core, GPU wheels, documentation tooling).
- Configure environment variables and data directories.
- Run smoke checks to confirm the installation.

## 3. Prerequisites

- **Operating system**: Linux or macOS (Windows users should use WSL2).
- **Git**: 2.30+ with Git LFS enabled (`git lfs install && git lfs pull` is required to fetch bundled wheels under `ci/wheels/`).
- **Direnv** *(recommended)*: `brew install direnv` or follow [direnv.net](https://direnv.net/) instructions to let `.envrc` manage the virtualenv automatically.
- **Memory**: 16 GB RAM recommended for Docling parsing workloads.
- **GPU (optional)**: CUDA 12.9+ drivers for FAISS, PyTorch, vLLM, and CuPy acceleration.

## 4. Installation

### 4.1 Clone the Repository

```bash
git clone https://github.com/paul-heyse/DocsToKG.git
cd DocsToKG
git lfs install
git lfs pull
```

### 4.2 Bootstrap the Environment (Recommended)

```bash
./scripts/bootstrap_env.sh
direnv allow           # loads .envrc and activates .venv automatically
```

The bootstrap script:

- Creates `.venv/` with CPython 3.13.
- Installs DocsToKG in editable mode and applies the pinned `requirements.txt`.
- Reuses local wheels (`torch`, `torchaudio`, `torchvision`, `faiss`, `vllm`, `cupy-cuda12x`) from `ci/wheels/`.

Use `./scripts/dev.sh exec <command>` in environments where `direnv` cannot be installed.

### 4.3 Manual Setup (Alternative)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt        # includes bundled wheels
```

Install documentation tooling on-demand:

```bash
pip install -r docs/build/sphinx/requirements.txt
```

### 4.4 Configure Environment Variables

Create a `.env` file (loaded by `.envrc`) or export variables through your secrets manager:

```bash
cat > .env <<'EOF'
DOCSTOKG_DATA_ROOT=./Data
HYBRID_SEARCH_CONFIG=./config/hybrid_config.json
ONTOLOGY_FETCHER_CONFIG=./configs/sources.yaml

PA_ALEX_KEY=your-pyalex-api-key
BIOPORTAL_API_KEY=optional-bioportal-token
EOF
```

Direnv automatically loads `.env`; otherwise run `source .env`.

### 4.5 Prepare Local Directories

```bash
mkdir -p Data/DocTagsFiles Data/ChunkedDocTagFiles Data/Embeddings artifacts logs
```

These directories store DocTags inputs, chunked outputs, embeddings, FAISS snapshots, and telemetry.

### 4.6 Smoke Checks

```bash
direnv exec . pytest -q
direnv exec . python docs/scripts/generate_api_docs.py
direnv exec . python docs/scripts/validate_docs.py
direnv exec . python docs/scripts/validate_code_annotations.py
```

Optional link check:

```bash
direnv exec . python docs/scripts/check_links.py --timeout 10
```

### 4.7 Build Sphinx HTML Docs (Optional)

```bash
direnv exec . python docs/scripts/build_docs.py --format html
```

Open `docs/build/_build/html/index.html` in your browser once the build completes.

## 5. Next Steps

- Run content download pipelines via `python -m DocsToKG.ContentDownload.cli`.
- Generate DocTags and embeddings (`docparse chunk`, `docparse embed`).
- Ingest data into the hybrid search service and expose APIs (`DocsToKG.HybridSearch`).

Refer to the **Architecture Guide** (`docs/03-architecture/index.md`) for subsystem details and the **Operations Guide** (`docs/06-operations/index.md`) for day-two workflows.
