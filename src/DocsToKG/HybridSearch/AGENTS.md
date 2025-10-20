
## Table of Contents

- [🚨 Mandatory Pre-Read: FAISS/cuVS Stack Walkthrough](#-mandatory-pre-read-faisscuvs-stack-walkthrough)
- [0) Guard rails (set once per session)](#0-guard-rails-set-once-per-session)
- [1) Verify the environment exists (no install)](#1-verify-the-environment-exists-no-install)
- [2) Run commands strictly from the project `.venv`](#2-run-commands-strictly-from-the-project-venv)
- [3) Quick health checks (no network)](#3-quick-health-checks-no-network)
- [4) Typical tasks (all no-install)](#4-typical-tasks-all-no-install)
- [5) Troubleshooting (stay no-install)](#5-troubleshooting-stay-no-install)
- [6) “Absolutely no installs” policy (what you may do)](#6-absolutely-no-installs-policy-what-you-may-do)
- [7) Fallback (only with **explicit approval** to install)](#7-fallback-only-with-explicit-approval-to-install)
- [8) One-page quick reference (copy/paste safe)](#8-one-page-quick-reference-copy-paste-safe)
- [Mission and Scope](#mission-and-scope)
- [Runtime prerequisites](#runtime-prerequisites)
- [Module architecture](#module-architecture)
- [Core capabilities](#core-capabilities)
- [Ingestion workflow](#ingestion-workflow)
- [Search API quick reference](#search-api-quick-reference)
- [Key invariants](#key-invariants)
- [Test matrix & verification](#test-matrix-verification)
- [Troubleshooting cues](#troubleshooting-cues)
- [Reference commands](#reference-commands)
- [Ownership & change management](#ownership-change-management)
- [Coding Standards & Module Organization](#coding-standards-module-organization)

# 🚨 Mandatory Pre-Read: FAISS/cuVS Stack Walkthrough

Before executing or modifying any HybridSearch code, you **must** complete the following sequence:

1. **Read [`faiss-gpu-wheel-reference.md`](./faiss-gpu-wheel-reference.md).** This is the authoritative guide to the custom `faiss-1.12.0` CUDA wheel. It covers runtime prerequisites, environment knobs (`FAISS_OPT_LEVEL`, `use_cuvs`), GPU index families, tiling heuristics, and the mathematically heavy kernels that underpin HybridSearch. Treat it as mandatory before touching ingestion, routing, or search code.
2. **Inspect the FAISS package** under `.venv/lib/python3.13/site-packages/faiss`. Review `swigfaiss.py`, `gpu_wrappers.py`, `class_wrappers.py`, and contrib utilities to understand dtype/layout expectations, stream semantics, and the no-index helpers (`knn_gpu`, `pairwise_distance_gpu`) that HybridSearch relies on.
3. **Read [`cuvs-reference.md`](./cuvs-reference.md).** The cuVS toolkit supplies GPU ANN, clustering, and distance primitives built on RAPIDS RAFT/RMM. The reference documents algorithm APIs, loader prerequisites, and how HybridSearch preloads the libraries even though FAISS currently cannot enable cuVS kernels.
4. **Read [`libcuvs-reference.md`](./libcuvs-reference.md).** This explains the shared-library loader chain (`libcuvs`, `libraft`, `librmm`, `rapids_logger`), environment switches, and HybridSearch integration hooks (`_ensure_cuvs_loader_path`, `resolve_cuvs_state`, `AdapterStats.cuvs_*`). Essential for diagnosing loader-path or RAPIDS memory-manager issues.

Completing these steps ensures you understand how FAISS, cuVS, and the RAPIDS memory/logging stack interoperate today, why cuVS is currently disabled in the custom FAISS build, and what guardrails HybridSearch enforces when operating on GPU indexes.

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
# HybridSearch quickstart (JSONL by default; add --vector-format parquet when DocParsing emitted parquet vectors)
./.venv/bin/python examples/hybrid_search_quickstart.py --query "hybrid retrieval faiss"

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
- **`IngestError: parquet vector ingestion requires pyarrow`**
  Install the DocsToKG `docparse-parquet` extra (adds `pyarrow`) or run DocParsing in JSONL mode. HybridSearch cannot ingest parquet vectors without that dependency.

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

# Agents Guide – HybridSearch

Last updated: 2025-02-15

> For full detail, read [README.md](./README.md) and the [FAISS GPU wheel reference](./faiss-gpu-wheel-reference.md). This section mirrors the high-level guidance there so agents have a consistent source of truth.

## Mission and Scope

- Mission: Provide hybrid (lexical + dense) retrieval with deterministic fusion, GPU-accelerated storage, and robust observability for DocsToKG.
- In scope: DocParsing‑driven ingestion, feature normalisation, FAISS/OpenSearch orchestration, namespace routing, fusion logic, API/service layer, metrics.
- Out of scope: Embedding model training, downstream answer generation, long-term archival policy.

## Runtime prerequisites

- Linux with CUDA‑12 capable NVIDIA GPUs plus the custom FAISS 1.12 GPU wheel. Required shared libraries: `libcudart.so.12`, `libcublas.so.12`, `libopenblas.so.0`, `libjemalloc.so.2`, `libgomp.so.1`, compatible `GLIBC_2.38`/`GLIBCXX_3.4.32`.
- Environment variables: `DOCSTOKG_DATA_ROOT` (defaults to `./Data`), optional `DOCSTOKG_HYBRID_CONFIG`, `TEMP_DIR`/`TMPDIR` for snapshot staging, `FAISS_OPT_LEVEL` / `FAISS_DISABLE_CPU_FEATURES` for loader overrides.
- Inputs: DocParsing chunk JSONL + embedding JSONL (aligned via `uuid`/`UUID`) and their manifests; ingestion trusts DocParsing for ID consistency.

## Module architecture

- `config.py` – Dataclass configs (`ChunkingConfig`, `DenseIndexConfig`, `FusionConfig`, `RetrievalConfig`, `HybridSearchConfig`) and the thread-safe `HybridSearchConfigManager`. The config layer now covers snapshot refresh throttles (`snapshot_refresh_interval_seconds` / `_writes`), persistence policies (`persist_mode`), forced ID removal fallbacks, and cuVS/FP16 toggles that map 1:1 onto FAISS GPU options (`GpuMultipleClonerOptions`, `StandardGpuResources`, tiling limits).
- `features.py` – Canonical deterministic tokeniser, sliding-window chunker, and `FeatureGenerator` used by ingestion, validation, and fixtures. `devtools.features` re-exports these symbols for backwards compatibility.
- `pipeline.py` – `ChunkIngestionPipeline`, `Observability`, and `IngestMetrics` stream DocParsing artefacts, normalise BM25/SPLADE/dense payloads into contiguous `float32` tensors, keep lexical + FAISS stores in lockstep, and surface structured telemetry.
- `store.py` – `FaissVectorStore` and `ManagedFaissAdapter` own CPU training, `index_cpu_to_gpu(_multiple)` cloning, multi-GPU replication/sharding, `StandardGpuResources` pools, cuVS negotiation (`resolve_cuvs_state`), cosine/inner-product helpers (`cosine_topk_blockwise`, `pairwise_inner_products`), snapshot utilities (`serialize_state`, `restore_state`), the `ChunkRegistry`, and rich `AdapterStats`.
- `router.py` – `FaissRouter` provisions namespace-scoped adapters, caches serialized payloads for evicted stores, rehydrates snapshots lazily, propagates ID resolvers, and reports per-namespace stats/last-used timestamps.
- `service.py` – Houses `HybridSearchValidator`, `HybridSearchService`, and `HybridSearchAPI`; validates requests, enforces pagination (`verify_pagination`), schedules concurrent lexical/dense searches, applies RRF + optional MMR diversification via `ResultShaper`, emits diagnostics, and exposes `build_stats_snapshot` for health checks.
- `types.py` & `interfaces.py` – Shared dataclasses (`ChunkPayload`, `ChunkFeatures`, `HybridSearchRequest/Response`, diagnostics) and protocols (`DenseVectorStore`, `LexicalIndex`) that keep ingestion, storage, and service layers interoperable.
- `devtools/` – Re-exports deterministic feature utilities and the in-memory `OpenSearchSimulator` (plus schema helpers) so notebooks and regression harnesses can import from a single namespace without external services.

## Core capabilities

- **Ingestion pipeline** – Validates manifests, normalises chunk payloads, keeps lexical and dense stores in sync, and surfaces metrics (latency histograms, GPU usage).
- **Vector store management** – Handles FAISS GPU lifecycle (training, replication, memory reservations, cosine/inner-product helpers) and snapshot metadata used for cold starts.
- **Namespace routing** – Maintains namespace→adapter mappings, caches snapshots when evicting idle stores, and aggregates stats for multi-tenant deployments.
- **Hybrid retrieval** – Executes dense + lexical lookups in parallel, fuses results via RRF/MMR, enforces pagination and token budgets, returns per-channel diagnostics.
- **Configuration surface** – Exposes the knobs that map YAML/JSON configs to FAISS runtime behaviour with atomic reload and legacy alias handling.
- **Observability** – Provides `Observability`, `AdapterStats`, and `service.build_stats_snapshot` for dashboards, health checks, and regression harnesses.

## Ingestion workflow

1. **Source artifacts** – Place chunk and embedding JSONL (plus manifests) under `${DOCSTOKG_DATA_ROOT}`.
2. **Load configuration** – Instantiate `HybridSearchConfigManager(Path(...))` and call `manager.get()` to obtain the current `HybridSearchConfig` (namespaces, budgets, GPU settings, snapshot directories).
3. **Initialise indexes** – Instantiate `ManagedFaissAdapter` (and optional lexical indexes) using the loaded config; ensure `ChunkRegistry` is ready.
4. **Stream ingestion** – Call `ChunkIngestionPipeline.upsert_documents(...)` to normalise features, upsert lexical payloads, add dense vectors to FAISS, and emit telemetry.
5. **Snapshot & persist** – Use `FaissRouter.serialize_all()` or per-adapter `serialize()`/`snapshot_meta()` to capture FAISS bytes + metadata; store in durable storage for fast restore.
6. **Serve queries** – Wire `HybridSearchService` / `HybridSearchAPI` into your runtime, load snapshots on start-up, then serve hybrid queries with deterministic fusion.

## Search API quick reference

- **Request** – `HybridSearchRequest` carries the query string, optional namespace, filter mapping, pagination (`page_size`, `cursor`), and toggles for `diversification`, `diagnostics`, and `recall_first`. Channel weights and top-k budgets come from `HybridSearchConfig`.
- **Response** – `HybridSearchResponse` bundles ranked chunks with `doc_id`, `chunk_id`, `vector_id`, `fused_rank`, highlights, metadata, and per-result diagnostics (`bm25`, `splade`, `dense`, optional `fusion_weights`). Top-level fields include `next_cursor`, `total_candidates`, `timings_ms`, `fusion_weights`, and `stats` (including `AdapterStats` snapshots).
- **Validation errors** – Surface as `RequestValidationError` with JSON payload; pagination guards reject out-of-budget requests.

## Key invariants

- Stable UUID→FAISS-id mapping via `_vector_uuid_to_faiss_int`; never mutate directly.
- Fusion determinism: identical inputs/config must yield identical ranking; update tests if fusion logic changes.
- Chunk registry parity: lexical and dense stores must process identical add/remove sequences.
- Token/byte budgets enforced by `ResultShaper`; adjust only with accompanying tests and config updates.

## Test matrix & verification

```bash
direnv exec . ruff check src/DocsToKG/HybridSearch tests/hybrid_search
direnv exec . mypy src/DocsToKG/HybridSearch
direnv exec . pytest tests/hybrid_search/test_suite.py -q
direnv exec . pytest tests/hybrid_search/test_suite.py::test_hybrid_scale_suite -q  # optional perf check
```

- GPU-specific scenarios rely on the custom wheel; run on hardware with CUDA 12.
- Keep fixtures under `tests/hybrid_search/` consistent with DocParsing schema expectations.

## Troubleshooting cues

| Symptom | Likely cause | Quick checks |
|---|---|---|
| Search latency spike | FAISS replica evicted or `nprobe` too high | Inspect `AdapterStats`; run `FaissRouter.stats()`; tune `DenseIndexConfig.nprobe`. |
| Missing highlights | Lexical index not updated | Compare `ChunkRegistry` entries vs lexical bulk upsert; rerun ingestion for namespace. |
| GPU OOM during ingest | Oversized IVFPQ / replication | Lower `expected_ntotal`, disable replication, use CPU persist mode temporarily. |
| Pagination duplicates | Cursor misuse or disabled verification | Ensure `verify_pagination` remains enabled and respect configured page size limits. |

## Reference commands

```bash
# Hybrid quickstart harness (writes tmp config, ingests sample data, runs search)
direnv exec . python examples/hybrid_search_quickstart.py

# Regression suite
direnv exec . pytest tests/hybrid_search/test_suite.py -q
```

## Ownership & change management

- Code owners: see repository `CODEOWNERS` entry for `src/DocsToKG/HybridSearch/`.
- Update this guide alongside `README.md` whenever you change ingestion flow, fusion behaviour, GPU requirements, or configuration fields.

## Coding Standards & Module Organization

- Follow the documentation in [CODE_ANNOTATION_STANDARDS.md](../docs/CODE_ANNOTATION_STANDARDS.md) when annotating code or updating NAVMAP headers.
- Structure modules according to [MODULE_ORGANIZATION_GUIDE.md.txt](../docs/html/_sources/MODULE_ORGANIZATION_GUIDE.md.txt) so imports, dataclasses, and public APIs remain predictable for downstream agents.
