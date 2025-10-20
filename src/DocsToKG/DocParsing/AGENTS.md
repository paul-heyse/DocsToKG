
## Table of Contents

- [0) Guard rails (set once per session)](#0-guard-rails-set-once-per-session)
- [1) Verify the environment exists (no install)](#1-verify-the-environment-exists-no-install)
- [2) Run commands strictly from the project `.venv`](#2-run-commands-strictly-from-the-project-venv)
- [3) Quick health checks (no network)](#3-quick-health-checks-no-network)
- [4) Typical tasks (all no-install)](#4-typical-tasks-all-no-install)
- [5) Troubleshooting (stay no-install)](#5-troubleshooting-stay-no-install)
- [6) “Absolutely no installs” policy (what you may do)](#6-absolutely-no-installs-policy-what-you-may-do)
- [7) Fallback (only with **explicit approval** to install)](#7-fallback-only-with-explicit-approval-to-install)
- [8) One-page quick reference (copy/paste safe)](#8-one-page-quick-reference-copy-paste-safe)
- [Mission & Scope](#mission-scope)
- [Prerequisites & Optional Dependencies (aligns with README)](#prerequisites-optional-dependencies-aligns-with-readme)
- [Quickstart (from README)](#quickstart-from-readme)
- [Core Capabilities & Flow](#core-capabilities-flow)
- [Configuration Surfaces](#configuration-surfaces)
- [Data Contracts & Schemas](#data-contracts-schemas)
- [Observability & Operations](#observability-operations)
- [Performance & Profiling](#performance-profiling)
- [Invariants & Safe Change Surfaces](#invariants-safe-change-surfaces)
- [Test Matrix & Quality Gates](#test-matrix-quality-gates)
- [Failure Modes & Debugging](#failure-modes-debugging)
- [Canonical Commands](#canonical-commands)
- [Reference Docs](#reference-docs)
- [Coding Standards & Module Organization](#coding-standards-module-organization)

# Project Environment — **No-Install** Runbook (for AI agents)

> **Prime directive:** **Use the packages already installed in `./.venv`.**
> Do **not** fetch or install anything unless you are explicitly told to. The project `.venv` includes custom wheels and GPU-specific builds that must not be disturbed.

---

## 0) Guard rails (set once per session)

To make accidental installs impossible:

```bash
# Always run inside the project venv (enforced)
export PIP_REQUIRE_VIRTUALENV=1

# Never hit the network/package index unless explicitly allowed
export PIP_NO_INDEX=1

# Don’t read user site-packages (avoid leakage)
export PYTHONNOUSERSITE=1
```

> If you later receive explicit approval to install, temporarily unset `PIP_NO_INDEX` (and only follow the “Fallback (with approval)” section at the end).

---

## 1) Verify the environment exists (no install)

```bash
# from repo root
test -x .venv/bin/python || { echo "ERROR: .venv is missing — STOP (no installs)."; exit 1; }
```

If missing or broken: **stop and report**. Do **not** create or modify the environment without authorization.

---

## 2) Run commands strictly from the project `.venv`

Pick **one** method below. All of them resolve **imports and console scripts from `./.venv`** and avoid installs.

### A) Most explicit (activation-free; recommended for agents)

```bash
# Call tools by absolute path inside the venv
./.venv/bin/python -m pip --version      # proves you're on ./.venv/bin/python
./.venv/bin/python -m DocsToKG.ContentDownload.cli --help
./.venv/bin/pytest -q
./.venv/bin/ruff check .
./.venv/bin/mypy src
```

### B) `direnv` (auto-env; if available)

```bash
direnv allow                             # trust once per machine
direnv exec . python -m pip --version
direnv exec . python -m DocsToKG.ContentDownload.cli --help
direnv exec . pytest -q
```

### C) `./scripts/dev.sh` (portable wrapper; no direnv needed)

```bash
./scripts/dev.sh doctor                  # prints interpreter/env and importability
./scripts/dev.sh python -m DocsToKG.ContentDownload.cli --help
./scripts/dev.sh exec pytest -q
./scripts/dev.sh pip list                # safe: listing does not install
```

### D) Classic activation (if explicitly requested)

```bash
# Linux/macOS
source .venv/bin/activate
export PYTHONPATH="\$PWD/src:${PYTHONPATH:-}"    # mirrors project behavior
python -m pip --version
python -m DocsToKG.ContentDownload.cli --help
pytest -q
```

> Prefer **A–C** for automation. **D** is acceptable in interactive shells but easier to get wrong.

---

## 3) Quick health checks (no network)

Run these **before** heavy work:

```bash
# 1) Interpreter identity (must be the project venv)
./.venv/bin/python - <<'PY'
import sys
assert sys.executable.endswith("/.venv/bin/python"), sys.executable
print("OK: using", sys.executable)
PY

# 2) Package presence WITHOUT installing (examples)
./.venv/bin/python -c "import DocsToKG, pkgutil; print('DocsToKG OK');"
./.venv/bin/python -c "import faiss; print('FAISS OK')"
./.venv/bin/python -c "import cupy; import numpy; print('CuPy OK', cupy.__version__)"
```

If any import fails: **do not install**. Go to Troubleshooting.

---

## 4) Typical tasks (all no-install)

```bash
# CLIs (module form)
./.venv/bin/python -m DocsToKG.ContentDownload.cli --help

# Tests
./.venv/bin/pytest -q

# Lint/format
./.venv/bin/ruff check .
./.venv/bin/black --check .

# Type check
./.venv/bin/mypy src
```

> Always prefer `python -m <module>` and `.venv/bin/<tool>` — these guarantee resolution from the project environment.

---

## 5) Troubleshooting (stay no-install)

**Symptom → Action (no installs):**

- **`ModuleNotFoundError`**
  You’re not using the project interpreter. Re-run via one of §2 methods, then re-check `sys.executable`.

- **GPU/FAISS/CuPy errors** (e.g., missing `.so`/DLL)
  Do **not** build or fetch wheels. Report the exact error. These packages are customized; replacing them may break GPU paths.

- **`pip` tries to fetch**
  You forgot the guard rails. Ensure `PIP_REQUIRE_VIRTUALENV=1` and `PIP_NO_INDEX=1` are set. Never pass `-U/--upgrade`.

---

## 6) “Absolutely no installs” policy (what you may do)

- You **may**:

  - Inspect environment: `./.venv/bin/pip list`, `./.venv/bin/pip show <pkg>`.
  - Run any console script from `./.venv/bin/…`.
  - Read code and run module CLIs with `python -m …`.

- You **must not**:

  - Run `pip install`, `pip wheel`, `pip cache purge`, or `pip uninstall`.
  - Upgrade/downgrade packages (including `pip` itself).
  - Recreate or modify `./.venv` without explicit approval.

---

## 7) Fallback (only with **explicit approval** to install)

If (and only if) you have written approval to modify the environment, apply the **smallest necessary** action **inside** the venv:

```bash
# ensure you are in the project venv first:
source .venv/bin/activate  # or use ./.venv/bin/python -m pip ...
unset PIP_NO_INDEX         # allow index access if instructed

# project code (editable) and pinned deps ONLY:
pip install -e .
pip install -r requirements.txt

# If a local wheelhouse exists (to avoid network):
# pip install --no-index --find-links ./ci/wheels -r requirements.txt
```

> Never “try versions” or compile GPU libs. If a wheel is missing, escalate.

---

## 8) One-page quick reference (copy/paste safe)

```bash
# Guard rails (no accidental installs)
export PIP_REQUIRE_VIRTUALENV=1 PIP_NO_INDEX=1 PYTHONNOUSERSITE=1

# Verify venv exists (stop if missing)
test -x .venv/bin/python || { echo "Missing .venv — STOP (no installs)."; exit 1; }

# Preferred run patterns (choose ONE)
./.venv/bin/python -m DocsToKG.ContentDownload.cli --help
./.venv/bin/pytest -q
# or
direnv exec . python -m DocsToKG.ContentDownload.cli --help
direnv exec . pytest -q
# or
./scripts/dev.sh doctor
./scripts/dev.sh python -m DocsToKG.ContentDownload.cli --help
./scripts/dev.sh exec pytest -q

# Health checks (no network)
./.venv/bin/python - <<'PY'
import sys; assert sys.executable.endswith("/.venv/bin/python"); print("OK:", sys.executable)
PY
./.venv/bin/python -c "import DocsToKG, faiss, cupy; print('Core imports OK')"
```

---

### Final note for agents

This repository’s environment includes **custom wheels and GPU-optimized packages**. Treat the `.venv` as **immutable** unless you are explicitly told to modify it. Your default posture is **execute only**: run what’s already installed, verify, and report issues rather than “fixing” them by installing.

# Agents Guide - DocParsing

Last updated: 2025-10-20

## Mission & Scope

- **Mission**: Convert raw documents into DocTags, chunks, and embeddings with resumable manifests and deterministic hashing so downstream search pipelines can rely on consistent outputs.
- **Scope**: DocTags conversion (PDF/HTML), chunk coalescence, embedding generation (dense/sparse/lexical), telemetry/manifests, staging utilities.
- **Out-of-scope**: Vector-store ingestion, downstream ranking orchestration, training new embedding/DocTags models.

## Prerequisites & Optional Dependencies (aligns with README)

- Python 3.10+, Linux recommended; GPU strongly suggested for PDF DocTags (vLLM) and Qwen embeddings.
- Extras (`pip install`):
  - Core pipeline: `"DocsToKG[docparse]"`
  - PDF DocTags (vLLM + Docling extras): `"DocsToKG[docparse-pdf]"`
  - SPLADE sparse embeddings: `sentence-transformers`
  - Qwen dense embeddings: `vllm` + CUDA 12 libraries (`libcudart.so.12`, `libcublas.so.12`, `libopenblas.so.0`, `libjemalloc.so.2`, `libgomp.so.1`)
  - Parquet vector export/validation: `"DocsToKG[docparse-parquet]"` (installs `pyarrow`)
- Model caches: DocTags `granite-docling-258M` under `${DOCSTOKG_MODEL_ROOT}`; SPLADE/Qwen weights under `${DOCSTOKG_SPLADE_DIR}` / `${DOCSTOKG_QWEN_DIR}` (legacy `${DOCSTOKG_QWEN_MODEL_DIR}` still honoured).
- Data directories (defaults derived from `${DOCSTOKG_DATA_ROOT}` / `--data-root`):
  - `Data/PDFs`, `Data/HTML` inputs
  - `Data/DocTagsFiles`, `Data/ChunkedDocTagFiles`, `Data/Embeddings`
  - `Data/Manifests/docparse.*.manifest.jsonl`
- Environment overrides: `DOCSTOKG_DOCTAGS_*`, `DOCSTOKG_CHUNK_*`, `DOCSTOKG_EMBED_*`, `DOCSTOKG_HASH_ALG`, etc. See configuration section below.

## Quickstart (from README)

```bash
./scripts/bootstrap_env.sh
direnv allow  # or source .venv/bin/activate

direnv exec . python -m DocsToKG.DocParsing.core.cli plan \
  --data-root Data \
  --mode auto \
  --limit 10

direnv exec . python -m DocsToKG.DocParsing.core.cli doctags \
  --mode pdf \
  --input Data/PDFs \
  --output Data/DocTagsFiles

direnv exec . python -m DocsToKG.DocParsing.core.cli chunk \
  --in-dir Data/DocTagsFiles \
  --out-dir Data/ChunkedDocTagFiles

direnv exec . python -m DocsToKG.DocParsing.core.cli embed \
  --chunks-dir Data/ChunkedDocTagFiles \
  --out-dir Data/Embeddings
#   --format parquet  # optional: emit columnar vectors when pyarrow is available
```

- `docparse all` coordinates stages end-to-end; `--resume` reuses manifests, `--force` regenerates outputs.

## Core Capabilities & Flow

- `core.cli` entry point with subcommands: `doctags`, `chunk`, `embed`, `plan`, `manifest`, `token-profiles`, `all`.
- DocTags conversion (`doctags.py`, Docling/vLLM integration) emits `docparse.doctags-*.manifest.jsonl`.
- Chunking (`chunking.runtime`) performs structural + token-aware coalescence with deterministic span hashing.
- Embedding runtime (`embedding.runtime`) supports dense (Qwen/vLLM), sparse (SPLADE), and lexical (BM25) backends with quarantine + optional dependency checks.
- Telemetry/IO (`telemetry.py`, `io.py`) enforce append-only manifests and advisory locks for atomic writes.

```mermaid
flowchart LR
  A[Raw documents (PDF/HTML/ZIP)] --> B[DocTags conversion]
  B --> C[Chunking & coalescence]
  C --> D[Embedding generation]
  B -.-> MB[DocTags manifest]
  C -.-> MC[Chunk manifest]
  D -.-> ME[Embedding manifest]
  classDef boundary stroke:#f00;
  B:::boundary
  C:::boundary
  D:::boundary
```

## Configuration Surfaces

- Config sources: environment (`DOCSTOKG_DATA_ROOT`, `DOCSTOKG_MODEL_ROOT`, stage-specific `DOCSTOKG_DOCTAGS_*`, `DOCSTOKG_CHUNK_*`, `DOCSTOKG_EMBED_*`), CLI flags, optional YAML/TOML via `config_loaders`.
- Shared CLI flags: `--resume`, `--force`, `--log-level`, `--data-root`, `--manifest-dir`.
- Stage-specific highlights:
  - `doctags`: `--mode {pdf,html,auto}`, `--vllm-wait-timeout`, `--port`, `--workers`.
  - `chunk`: `--min-tokens`, `--max-tokens`, `--merge-threshold`, `--shard-count/index`, `--validate-only`.
  - `embed`: `--backend {qwen,splade,bm25}`, `--batch-size-*`, `--validate-only`, `--device`, `--quantization`.
- Content hashing defaults: `DOCSTOKG_HASH_ALG` (default SHA-256). Switching to SHA-1 is only for legacy resumes; expect manifest diff (`input_hash`, chunk UUID).
- Validate config with `core.cli chunk --validate-only` or `embed --validate-only`.
- Vector format defaults: set `DOCSTOKG_EMBED_VECTOR_FORMAT=parquet` (or pass `--format`) to opt into parquet outputs; leave unset for JSONL.

## Data Contracts & Schemas

- Manifests: `docparse.doctags|chunk|embeddings.manifest.jsonl` (append-only, idempotent).
- Schemas: `formats.CHUNK_ROW_SCHEMA`, `formats.VECTOR_ROW_SCHEMA`, manifest payloads in `telemetry.ManifestEntry`.
- Outputs use consistent doc IDs/hashes across stages so resume + downstream ingestion remain deterministic.
- Advisory locks in `telemetry.StageTelemetry` prevent concurrent writers from corrupting manifests.

## Observability & Operations

- Logs: `logging.py` outputs structured JSONL (`${DOCSTOKG_LOG_DIR:-Data/Logs}/docparse-*.jsonl`) plus console; fields include `stage`, `doc_id`, durations, correlation IDs.
- Telemetry: `telemetry.StageTelemetry` emits manifest attempts/summaries per stage; parse with `docparse manifest --stage <stage>` or `jq`. Manifest entries now include `vector_format` for success, skip, and validate-only records to monitor parquet adoption.
- SLO guidance: maintain ≥99.5 % manifest success; keep `embed --validate-only` P50 ≤2.2 s/doc (per README).
- Operational tooling: `core.cli plan` (stage preview), `manifest` (tail JSONL), `token-profiles` (chunk diagnostics), `all --resume` (pipeline orchestrator).

## Performance & Profiling

- Pipeline baselines (README): HTML→DocTags 30–50 docs/min (CPU), PDF→DocTags 5–10 docs/min (A100), chunking 10–20 docs/min (CPU), embeddings 5–8 docs/min (A100).
- Profiling recipes:

  ```bash
  direnv exec . python -m cProfile -m DocsToKG.DocParsing.core.cli chunk --in-dir Data/DocTagsFiles --out-dir /tmp/chunks --limit 50
  direnv exec . pyinstrument -r html -o profile.html python -m DocsToKG.DocParsing.core.cli embed --chunks-dir Data/ChunkedDocTagFiles --out-dir /tmp/embeddings --limit 50
  direnv exec . pytest tests/docparsing/test_synthetic_benchmark.py -q
  ```

- Optimisation levers: streaming IO, batching token/embedding workloads, caching tokenizer/model instances per worker, tuning `--shard-count` and batch size, keeping merges linear.

## Invariants & Safe Change Surfaces

- Directory hierarchy for DocTags/chunks/embeddings mirrors input structure; resume tooling assumes this layout.
- Manifests are append-only; always include `doc_id`, `input_hash`, `status`, `attempts`.
- Deterministic chunk IDs/embedding hashes rely on DocTags ordering and selected hash algorithm.
- Use `chunking/`, `embedding/`, and `formats` modules for heuristic/schema changes; coordinate schema bumps with README/AGENTS updates.
- GPU handling uses spawn semantics; avoid manual forked CUDA processes.

## Test Matrix & Quality Gates

```bash
direnv exec . ruff check src/DocsToKG/DocParsing tests/docparsing
direnv exec . mypy src/DocsToKG/DocParsing
direnv exec . pytest tests/docparsing -q
direnv exec . pytest tests/docparsing/test_synthetic_benchmark.py -q  # performance smoke
direnv exec . pytest tests/docparsing/test_vector_writers.py -q        # parquet writer coverage
```

- Golden fixtures: `tests/data/docparsing/golden/` (DocTags/chunk/vector JSONL).
- High-signal suites: `tests/docparsing/test_cli_and_tripwires.py`, `test_docparsing_core.py`, `test_synthetic_benchmark.py`, `test_chunk_manifest_resume.py`.

## Failure Modes & Debugging

| Symptom | Likely Cause | Checks |
| --- | --- | --- |
| `CUDA error: reinitializing context` | Forked child touching CUDA before spawn | Ensure PDF DocTags workers use spawn; limit workers; configure `CUDA_VISIBLE_DEVICES`. |
| Chunk count mismatch | Resume skipped DocTags or stale hash | Inspect `docparse.chunks.manifest.jsonl`; rerun chunk with `--force` for affected docs. |
| Embedding dim mismatch | Wrong backend config or model upgrade | Run `embed --validate-only`; confirm vector dimension vs config. |
| Validate-only reports zero files | Vector format/dimension mismatch versus existing outputs | Provide `--format`/`DOCSTOKG_EMBED_VECTOR_FORMAT` and explicit `--qwen-dim` when revalidating freshly generated vectors; omit the override to accept historical artifacts. |
| Slow chunking | Quadratic merges from markers | Profile `HybridChunker.generate_chunks`; adjust merge thresholds. |
| Manifest corruption | Manual edits or partial writes | Rebuild via CLI; avoid editing JSONL by hand. |

## Canonical Commands

```bash
direnv exec . python -m DocsToKG.DocParsing.core.cli doctags --mode pdf --input Data/PDFs --output Data/DocTagsFiles --resume
direnv exec . python -m DocsToKG.DocParsing.core.cli chunk --in-dir Data/DocTagsFiles --out-dir Data/ChunkedDocTagFiles --min-tokens 256 --max-tokens 512
direnv exec . python -m DocsToKG.DocParsing.core.cli embed --chunks-dir Data/ChunkedDocTagFiles --out-dir Data/Embeddings --batch-size-qwen 24
direnv exec . python -m DocsToKG.DocParsing.core.cli embed --validate-only --chunks-dir Data/ChunkedDocTagFiles
direnv exec . python -m DocsToKG.DocParsing.core.cli embed --format parquet --validate-only --chunks-dir Data/ChunkedDocTagFiles
direnv exec . python -m DocsToKG.DocParsing.core.cli manifest --stage chunk --tail 20
```

## Reference Docs

- `src/DocsToKG/DocParsing/README.md`
- `docs/06-operations/docparsing-changelog.md`
- OpenSpec archives under `openspec/changes/` for historical design notes

## Coding Standards & Module Organization

- Follow the documentation in [CODE_ANNOTATION_STANDARDS.md](../../../docs/CODE_ANNOTATION_STANDARDS.md) when adding or updating inline documentation and NAVMAP headers.
- Structure modules according to [MODULE_ORGANIZATION_GUIDE.md.txt](../../../docs/html/_sources/MODULE_ORGANIZATION_GUIDE.md.txt), ensuring imports, type aliases, dataclasses, and public API sections remain predictable for downstream agents.
