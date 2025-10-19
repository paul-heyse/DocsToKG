---
subdir_id: docstokg-docparsing
owning_team: docs-to-kg
interfaces: [cli, python]
stability: stable
versioning: semver
codeowners: "@paul-heyse"
last_updated: 2025-10-18
related_adrs: []
slos: { availability: "99.5%", latency_p50_ms: 2200 }
data_handling: restricted-pii
sbom: { path: docs/sbom/docparsing.cdx.json }
---

# DocsToKG • DocParsing

Purpose: Convert raw documents (PDF/HTML/etc.) into DocTags, topic-aware chunks, and embeddings with resumable manifests.
Scope boundary: Handles conversion, chunking, embedding, and telemetry; does not persist vectors to external stores or orchestrate downstream ingestion.

---

## Quickstart
> Bootstrap environment, run DocTags → chunking → embedding on a sample corpus.
```bash
./scripts/bootstrap_env.sh
direnv allow                                     # or source .venv/bin/activate

# Convert PDFs/HTML to DocTags
direnv exec . python -m DocsToKG.DocParsing.core.cli doctags \
  --mode pdf \
  --input Data/PDFs \
  --output Data/DocTagsFiles

# Chunk DocTags
direnv exec . python -m DocsToKG.DocParsing.core.cli chunk \
  --in-dir Data/DocTagsFiles \
  --out-dir Data/ChunkedDocTagFiles

# Generate embeddings
direnv exec . python -m DocsToKG.DocParsing.core.cli embed \
  --chunks-dir Data/ChunkedDocTagFiles \
  --out-dir Data/Embeddings
```

## Common commands
```bash
# Table stakes CLI flows
direnv exec . python -m DocsToKG.DocParsing.core.cli doctags --help
direnv exec . python -m DocsToKG.DocParsing.core.cli chunk --help
direnv exec . python -m DocsToKG.DocParsing.core.cli embed --help

# Focused reruns
direnv exec . python -m DocsToKG.DocParsing.core.cli doctags --input Data/PDFs/doc-001.pdf --force
direnv exec . python -m DocsToKG.DocParsing.core.cli chunk --resume
direnv exec . python -m DocsToKG.DocParsing.core.cli embed --validate-only
```

### Planner previews
- `--plan-only` mode reports total document counts per bucket and shows up to
  five representative IDs for each category (`process`, `skip`, `validate`,
  etc.). Larger plans keep the counts accurate while truncating previews with a
  `... (+N more)` suffix so operators can gauge scale without loading thousands
  of identifiers.

## Folder map
- `cli_errors.py` – Structured CLI exception types.
- `config.py` / `config_loaders.py` – Pydantic-style configuration loaders (YAML/JSON/TOML) for chunking/embedding.
- `core/` – Pipeline orchestration (DocTags converters, chunk coalescence, embedding runners).
- `chunking/` – Hybrid chunker implementation and heuristics.
- `embedding/` – Dense/sparse embedding wrappers (Qwen, SPLADE, BM25).
- `formats/` – Schemas and JSON codecs for DocTags/chunk/vector manifests.
- `interfaces.py` – Protocol definitions for converters, chunkers, embedder backends.
- `io.py` – Filesystem helpers, manifest writers, doc discovery.
- `logging.py` / `telemetry.py` – Structured logging and telemetry sinks.
- `schemas.py` – Pydantic models and schema introspection utilities.
- `token_profiles.py` – Tokenizer profiles/presets shared across stages.
- `tests/docparsing/` – Unit/integration tests and synthetic benchmarks (with stubs).

## System overview
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
```mermaid
sequenceDiagram
  participant U as Operator/Agent
  participant CLI as DocParsing CLI
  participant Tags as DocTagsConverter
  participant Chunk as ChunkIngestionPipeline
  participant Embed as EmbeddingRunner
  U->>CLI: doctags --mode pdf --input …
  CLI->>Tags: render_to_doctags()
  Tags-->>CLI: DocTags + manifest row
  U->>CLI: chunk --in-dir …
  CLI->>Chunk: generate_chunks()
  Chunk-->>CLI: chunk JSONL + manifest row
  U->>CLI: embed --chunks-dir …
  CLI->>Embed: build_vectors()
  Embed-->>CLI: embeddings + manifest row
  CLI-->>U: command exit / telemetry
```

## Entry points & contracts
- Entry points: `python -m DocsToKG.DocParsing.core.cli` subcommands (`doctags`, `chunk`, `embed`), Python APIs in `core` for embedding/chunking.
- Contracts/invariants:
  - DocTags → chunk → embedding outputs mirror directory hierarchy and use consistent doc IDs.
  - Manifests (`docparse.*.manifest.jsonl`) are append-only and idempotent for resume logic.
  - Chunk and embedding schemas versioned via `formats.validate_schema_version`.

## Configuration
- Config sources: environment (`DOCSTOKG_DATA_ROOT`, `DOCSTOKG_MODEL_ROOT`, optional `DOCSTOKG_SPLADE_DEVICE`, `DOCSTOKG_QWEN_DEVICE`, etc.), YAML/TOML via `config_loaders`.
- CLI flags: shared `--resume`, `--force`, `--log-level`; stage-specific `--min-tokens`, `--max-tokens`, `--shard-count/index`, `--batch-size-*`, `--tokenizer-model`, `--format`, etc. PDF DocTags runs additionally accept `--vllm-wait-timeout` (mirrored by `docparse all`) so operators can extend the readiness window for vLLM backends.
- Environment overrides:
  - DocTags: `DOCSTOKG_DOCTAGS_*` family (`_INPUT`, `_OUTPUT`, `_MODEL`, `_WORKERS`, `_VLLM_WAIT_TIMEOUT`, etc.).
  - Chunking: `DOCSTOKG_CHUNK_*` toggles for tokenizer, shard count, and validation.
  - Embedding: `DOCSTOKG_EMBED_*` flags plus `DOCSTOKG_QWEN_DIR`, `DOCSTOKG_SPLADE_DIR` for model caches.
- Validate configuration: run `python -m DocsToKG.DocParsing.core.cli chunk --validate-only` or `... embed --validate-only` before production runs.

### Content hashing defaults
- **New default (2025-10-20):** `compute_content_hash`, `compute_chunk_uuid`, and `resolve_hash_algorithm()` now use SHA-256 by
  default. SHA-1 remains available via `DOCSTOKG_HASH_ALG=sha1` or explicit function arguments.
- **Operational impact:** SHA-256 digests change manifest `input_hash` values and chunk UUIDs, so resumed runs may schedule
  reprocessing for artifacts produced with the previous default. Set `DOCSTOKG_HASH_ALG=sha1` during migration if you must resume
  older manifests before switching fleets to the stronger digest.

## Data contracts & schemas
- Schemas: `formats.CHUNK_ROW_SCHEMA`, `formats.VECTOR_ROW_SCHEMA`, DocTags manifest rows emitted via `telemetry.ManifestEntry`.
- Manifests stored under `Data/Manifests/docparse.*.manifest.jsonl` (DocTags, chunking, embeddings).
- Chunk output: JSONL lines with `ChunkPayload`, `ProvenanceMetadata`, token spans; embeddings output: JSONL with dense vector (float32 list) + sparse weights.

## Interactions with other packages
- Upstream: consumes raw documents, optional DocTags produced by external systems.
- Downstream: supplies chunked text and embeddings to `HybridSearch`, `OntologyDownload` is independent.
- Guarantees: stable doc IDs across stages; chunk/embedding outputs designed for direct ingestion by hybrid search pipeline.

## Observability
- Logs: `logging.py` provides structured logs with doc IDs, stage, durations.
- Telemetry: `telemetry.TelemetrySink` appends JSONL attempts and manifests (`docparse.*.manifest.jsonl`), guarded by advisory locks for atomic writes.
- Metrics: wrap `StageTelemetry` via `logging.telemetry_scope` to capture per-stage success/failure counters and durations for run dashboards.
- **SLIs/SLOs**: Maintain ≥99.5 % manifest success across stages and keep embedding `--validate-only` P50 ≤2.2 s per document (baseline from synthetic benchmark).
- Health check: prefer `direnv exec . docparse chunk --validate-only` / `docparse embed --validate-only` to exercise manifests without rewriting artifacts.

## Security & data handling
- ASVS level: L2 baseline for pipelines that ingest untrusted documents and invoke GPU services.
- Threats:
  - Tampering: ensure DocTags/Chunks/Embeddings stored in controlled directories; keep manifests append-only.
  - DoS: guard via `--shard-count`, `--batch-size` controls; manifest resume prevents redundant work.
  - Information disclosure: treat source documents as sensitive; data contains regulated or customer-provided content.
  - Supply chain: verify Docling/vLLM dependencies; prefer locked versions.
- Data classification: `restricted-pii`; raw corpora and derived embeddings may include sensitive personal or proprietary data—scope storage and access controls accordingly.

## Development tasks
```bash
direnv exec . ruff check src/DocsToKG/DocParsing tests/docparsing
direnv exec . mypy src/DocsToKG/DocParsing
direnv exec . pytest tests/docparsing -q
direnv exec . pytest tests/docparsing/test_synthetic_benchmark.py -q  # optional perf check
```
- Format / lint: `direnv exec . ruff format src/DocsToKG/DocParsing tests/docparsing`.
- Use dependency stubs in tests (`tests.docparsing.stubs`) to run without GPUs.

## Agent guardrails
- Do:
  - Extend chunking/embedding via interfaces and keep manifests consistent.
  - Document new schema fields and migrate manifest validators.
- Do not:
  - Break directory layout or doc ID conventions without downstream coordination.
  - Bypass manifests or resume logic (tools depend on accurate entries).
- Danger zone:
  - `rm -rf Data/DocTagsFiles` or manually editing manifests may break resume; use CLI `--force` and allow pipeline to rebuild artifacts.
  - Changing embedding formats (`--format`) requires updating `formats` validators and downstream loaders.

## FAQ
- Q: How do I resume after a failure?
  A: Use `--resume` on the affected stage; manifests mark completed docs. Combine with `--force` for targeted reruns.

- Q: How do I validate outputs without writing new files?
  A: Run `chunk` or `embed` with `--validate-only` to check existing JSONL artifacts and exit with status.

<!-- Machine-readable appendix -->
```json x-agent-map
{
  "entry_points":[
    {"type":"cli","module":"DocsToKG.DocParsing.core.cli","commands":["doctags","chunk","embed"]},
    {"type":"python","module":"DocsToKG.DocParsing.core","symbols":["ChunkIngestionPipeline","EmbeddingRunner"]},
    {"type":"python","module":"DocsToKG.DocParsing.chunking","symbols":["HybridChunker"]},
    {"type":"python","module":"DocsToKG.DocParsing.embedding","symbols":["EmbeddingRunner","SPLADEAdapter","QwenAdapter"]}
  ],
  "env":[
    {"name":"DOCSTOKG_DATA_ROOT","default":"<repo>/Data","required":false}
  ],
  "schemas":[
    {"kind":"python","path":"src/DocsToKG/DocParsing/formats.py"},
    {"kind":"pydantic","path":"src/DocsToKG/DocParsing/config.py"}
  ],
  "artifacts_out":[
    {"path":"Data/DocTagsFiles/**/*.doctags","consumed_by":["chunk stage"]},
    {"path":"Data/ChunkedDocTagFiles/**/*.chunks.jsonl","consumed_by":["embedding stage","hybrid search"]},
    {"path":"Data/Embeddings/**/*.vectors.jsonl","consumed_by":["hybrid search"]}
  ],
  "danger_zone":[
    {"command":"direnv exec . python -m DocsToKG.DocParsing.core.cli doctags --force --input Data/PDFs","effect":"Reprocesses entire corpus; can overwrite manifests"}
  ]
}
```
