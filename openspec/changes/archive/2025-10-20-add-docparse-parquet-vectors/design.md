## Context
- CLI and orchestration layers (`docparse all`, `docparse embed`) already expose `--format/--embed-format` with `parquet` listed, but the runtime raises `NotImplementedError`. Documentation still promises `*.vectors.jsonl` outputs, creating a mismatch.
- Resume and planning logic derive output paths by hard-coding `.vectors.jsonl`. HybridSearch ingestion is likewise JSONL-only, so even if the writer existed, downstream callers would fail.
- `pyarrow` is already listed in optional dependencies, so no new third-party library is required, but we must tolerate environments where it is absent (clear error and actionable guidance).

## Goals / Non-Goals
- **Goals**
  - Provide a stable Parquet vector writer with atomic semantics and schema parity with JSONL.
  - Allow operators to select the output format via CLI, config files, or environment variables, with manifests/resume respecting the chosen extension.
  - Support validation (`--validate-only`) and HybridSearch ingestion for both JSONL and Parquet datasets.
  - Update documentation and fixtures so advertised formats match behavior.
- **Non-Goals**
  - Changing the default output format (remain JSONL unless explicitly requested).
  - Introducing new schema fields or altering the logical contents of vector rows.
  - Reworking DocParsing chunk output or DocTags stages.

## Decisions
- **Parquet file layout:** Each vectors artifact will retain one logical row per chunk. The Arrow schema will mirror the JSONL structure:
  - `UUID: string`
  - `BM25: struct<terms: list<string>, weights: list<float32>, avgdl: float64, N: int64>`
  - `SPLADEv3: struct<tokens: list<string>, weights: list<float32>>`
  - `Qwen3_4B: struct<model_id: string, vector: list<float32>, dimension: int32>`
  - `model_metadata: map<string, string|float|int|bool>` (store as `map<string, json>` by serialising metadata dict to JSON text to keep schema stable)
  - `schema_version: string`
  Dense vectors will be stored as float32 to match runtime expectations; sparse weights remain float32. Files adopt the suffix `.vectors.parquet`.
- **Parquet writer implementation:** Create `ParquetVectorWriter` that batches validated rows into Arrow tables using `pyarrow.Table.from_pylist` and writes them with `pyarrow.parquet.write_table`. The writer will allocate at most one batch of rows at a time to bound memory.
- **Atomic writes:** Wrap parquet writes in the existing `atomic_write` helper. The writer will stream the table to the temporary file, flush, fsync, and promote on success. A `_crash_after_write` gating hook will raise mid-write to reuse existing durability tests.
- **Format negotiation:** Extend `ParsingContext`, manifest payloads, CLI wiring, and resume logic to carry `vector_format`. Introduce a helper in `core.discovery` that, given the logical chunk path and format, derives the correct vectors filename so every call site shares the same logic.
- **Capability detection:** Import `pyarrow` lazily when Parquet is selected. If the module is missing (or lacks required symbols), raise `EmbeddingCLIValidationError` naming the module, the required extra (`DocsToKG[docparse-parquet]`), and actionable remediation steps instead of surfacing a generic `ModuleNotFoundError`.
- **Validation and resume:** `_validate_vectors_for_chunks` will dispatch to either the existing JSONL reader or a parquet reader built on `pyarrow.parquet.ParquetFile.iter_batches`, converting each record batch back into dictionaries before schema validation. Resume comparisons will only skip work when both the manifest and requested format align; mismatched extensions trigger regeneration.
- **HybridSearch ingestion:** Detect the format either by explicit dataset metadata or file suffix. Parquet ingestion will reuse Arrow iterators to hydrate sparse/dense vectors and maintain the same `ChunkPayload` fields. Mixed formats in a single namespace will yield a descriptive error before mutable state is touched.

## Risks / Trade-offs
- **Runtime footprint:** Writing dense vectors into Parquet can temporarily materialize data structures in memory; mitigate by batching the same size as existing JSONL batches.
- **Dependency availability:** Environments lacking `pyarrow` will now hit a clearer error pathway. Document installation expectations in README/AGENTS.
- **Compatibility:** Downstream automation that assumes `.vectors.jsonl` naming may need updates. Provide migration guidance, maintain JSONL as default, and surface a guardrail when manifests reference a different format than requested.
- **Mixed corpus ambiguity:** HybridSearch ingestion must defensively detect mixed JSONL/Parquet corpora to prevent silent partial loads. Error messaging should enumerate offending files to aid remediation.

## Migration Plan
1. Land code changes behind format toggle; default remains JSONL.
2. Update documentation/runbooks to announce new capability with explicit opt-in instructions.
3. Provide sample commands to validate parquet outputs (`docparse embed --format parquet --validate-only`).
4. After rollout, monitor HybridSearch ingestion metrics to ensure parquet datasets load successfully before encouraging broader adoption.

## Open Questions
- Do we need an environment variable override to force JSONL for legacy workflows even if configs request parquet? (Assumed unnecessary for now.)
- Should we emit combined manifests listing the output format per document for analytics? (Context already records `vector_format`; verify consumers read it.)
- Are there downstream services besides HybridSearch that deserialize vectors directly (e.g., analytics notebooks) that require sample code updates? Capture during documentation review.
