## Environment Setup

Use the uv bootstrap to stand up the project environment:
1. Optionally run `direnv allow` once per machine to trust `.envrc`.
2. For CPU-only work, run `./scripts/bootstrap_env.sh`.
3. For GPU work (requires wheels in `.wheelhouse/`), run `./scripts/bootstrap_env.sh --gpu`.
4. Activate with `direnv exec . <command>` or `source .venv/bin/activate`.

The script installs uv if it is missing, respects `UV_PROJECT_ENVIRONMENT`, and installs DocsToKG in editable mode. After activation, use the tools in `.venv/bin/` (for example `pytest -q`, `ruff check`, or `python -m DocsToKG.<module>`).

## Code Style & Architecture References

- Review [docs/Formats and Standards/CODESTYLE.md](<../../../../docs/Formats and Standards/CODESTYLE.md>) for the Python 3.12+ baseline, lint/type tooling expectations, and NAVMAP/docstring rules.
- Architecture briefs: [DocParsing overview](<../../../../docs/architecture/120-docparsing.updated.md>) and [Level 2 implementation notes](<../../../../docs/architecture/120-docparsing.level2.updated.md>).

## Mission & Scope

- **Mission**: Convert raw documents into DocTags, chunks, and embeddings with resumable manifests and deterministic hashing so downstream search pipelines can rely on consistent outputs.
- **Scope**: DocTags conversion (PDF/HTML), chunk coalescence, embedding generation (dense/sparse/lexical), telemetry/manifests, staging utilities.
- **Out-of-scope**: Vector-store ingestion, downstream ranking orchestration, training new embedding/DocTags models.

## Prerequisites & Optional Dependencies (aligns with README)

- Python 3.13+, Linux recommended; GPU strongly suggested for PDF DocTags (vLLM) and Qwen embeddings.
- Extras (`pip install`):
  - Core pipeline: `"DocsToKG[docparse]"` (installs `filelock>=3.20.0` and `jsonlines>=4.0.0` for locking/streaming primitives)
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
- Schemas: `DocsToKG.DocParsing.formats` exports `ChunkRow`, `VectorRow`, and schema version helpers; `DocsToKG.DocParsing.schemas` is a deprecated shim scheduled for removal in DocsToKG 0.3.0.
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
