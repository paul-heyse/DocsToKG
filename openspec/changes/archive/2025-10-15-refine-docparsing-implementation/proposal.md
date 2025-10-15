# Refine DocParsing Implementation: Production Hardening & Technical Debt Reduction

## Why

The DocParsing pipeline (`src/DocsToKG/DocParsing`) is functional but contains technical debt and production-readiness gaps that create maintenance burden, portability issues, and subtle correctness problems. These include:

- **Legacy script duplication**: Two deprecated direct-invocation converters remain alongside the unified CLI, confusing users and doubling maintenance surface
- **Non-atomic writes**: Crash during chunk/vector writing leaves partial JSONL files that corrupt resume logic
- **Incorrect UTC timestamps**: Logs claim UTC (`...Z`) but emit localtime, breaking cross-machine correlation
- **Hardcoded paths**: `/home/paul/hf-cache` and model directories prevent deployment to other environments
- **Memory inefficiency**: Embeddings pipeline retains full corpus text in `uuid_to_chunk` dict causing OOM on large datasets
- **CLI parsing complexity**: "Merge defaults + provided" pattern adds ~60 lines of boilerplate across scripts
- **Missing hash algorithm tagging**: Resume logic can't distinguish SHA-1 vs SHA-256 hashes, risking false matches
- **Manifest scaling**: Single monolithic `docparse.manifest.jsonl` slows resume scans on large runs

This change systematically addresses these 18 production-readiness gaps without altering the core pipeline architecture or breaking existing workflows.

## What Changes

### 1. Legacy Script Quarantine & Shims

- Move `run_docling_html_to_doctags_parallel.py` and `run_docling_parallel_with_vllm_debug.py` to `legacy/` subdirectory
- Replace their `main()` bodies with thin shims that invoke the unified CLI
- Preserve backward compatibility while steering users to the canonical entry points

### 2. Test Scaffolding Cleanup

- Remove `_promote_simple_namespace_modules()` from production chunker (lines 59-76 in `DoclingHybridChunkerPipelineWithMin.py`)
- Move helper to `tests/_stubs.py` as test-only utility
- Delete duplicate `SOFT_BARRIER_MARGIN = 64` constant (line 80)

### 3. Atomic Chunk & Vector Writes

- Replace direct `open("w")` with `atomic_write()` context manager from `_common`
- Changes: chunker line 705, embeddings line 755
- Prevents partial files on SIGKILL/OOM crashes

### 4. True UTC Timestamps in Structured Logs

- Set `JSONFormatter.converter = time.gmtime` in `_common.get_logger()`
- Ensures `%Y-%m-%dT%H:%M:%S.%fZ` format matches actual UTC

### 5. Hash Algorithm Tagging & SHA-256 Option

- Add `DOCSTOKG_HASH_ALG` environment variable (default: `sha1`, allow: `sha256`)
- Include `hash_alg` field in manifest entries alongside `input_hash`
- Update `compute_content_hash()` to respect env override
- Enables future migration to SHA-256 without breaking existing resume logic

### 6. Simplify CLI Argument Merging

- Replace "parse twice, merge defaults" pattern with direct `args = args or parser.parse_args()`
- Removes ~20 lines per script (chunker, embeddings, HTML/PDF converters)
- Eliminates subtle precedence bugs

### 7. Drop Corpus-Wide `uuid_to_chunk` Map

- Refactor embeddings Pass A to discard text after BM25 stat accumulation
- Pass B reads chunk files directly instead of consulting in-memory map
- Reduces peak memory by corpus-text-size (e.g., 10GB for 50K documents)

### 8. De-Hardcode Model & Cache Paths

- Replace `/home/paul/hf-cache` with `Path(os.getenv("HF_HOME") or Path.home()/".cache"/"huggingface")`
- Add `DOCSTOKG_MODEL_ROOT`, `DOCSTOKG_QWEN_DIR`, `DOCSTOKG_SPLADE_DIR` env vars
- Add `--qwen-model-dir` and `--splade-model-dir` CLI flags
- Makes pipeline portable across dev/CI/production environments

### 9. Manifest Sharding by Stage

- Split `docparse.manifest.jsonl` into `docparse.chunks.manifest.jsonl`, `docparse.embeddings.manifest.jsonl`, etc.
- OR rotate manifests by size (e.g., `docparse.manifest.001.jsonl`, scan latest first)
- Keeps resume scans fast on large (100K+ document) runs

### 10. vLLM Service Preflight Manifest Entry

- Persist single `doc_id="__service__"` entry at PDF pipeline startup with vLLM diagnostics
- Include: `served_models`, `vllm_version`, `port`, `/metrics` health check result
- Provides audit trail proving service was healthy before conversions started

### 11. SPLADE Sparsity Threshold Documentation

- Add `sparsity_warn_threshold_pct: 1.0` to corpus summary manifest entry
- Numeric threshold makes CI alerts unambiguous

### 12. Promote Image Flags to Top-Level Chunk Fields

- Keep `provenance.has_image_captions` etc., AND add optional top-level `ChunkRow` fields
- Enables downstream filters without digging into nested provenance

### 13. Offline / Local-Files-Only Toggles

- Add `--offline` flag setting `TRANSFORMERS_OFFLINE=1` for SPLADE/Qwen
- Add `--qwen-model-dir` existence check before vLLM instantiation
- Deterministic, air-gapped runs with clear early failures

### 14. Document SPLADE Attention Backend Fallback

- Update `--splade-attn=auto` help text to mention SDPA→eager→FA2 selection order
- Reduces operator confusion about which backend is actually used

### 15. Schema Version Enforcement at Readers

- Add one-liner check in readers: `validate_schema_version(row["schema_version"], COMPATIBLE_*)`
- Quick fail on mixed outputs (e.g., stale shard with old schema in same folder)

### 16. Resume Parity & Manifest Consistency

- Verify chunker/embeddings both use same `load_manifest_index()` + `input_hash` comparison logic
- Add `chunk_count` to success entries (already present, verify)

### 17. CLI Consolidation Documentation

- Ensure `cli/chunk_and_coalesce.py`, `cli/embed_vectors.py`, `cli/doctags_convert.py` are thin wrappers
- Update README to reference unified CLIs as primary entry points

### 18. PR Breakdown for Reviewability

- Phase 1: Legacy shims + test cleanup + constant dedupe (#1–3)
- Phase 2: Atomic writes + UTC logging + hash tagging (#4–6)
- Phase 3: CLI simplification across 3 entrypoints (#7)
- Phase 4: Embeddings memory refactor (drop uuid_to_chunk) (#8)
- Phase 5: Paths/env & offline flags + CLI overrides (#9, #13)
- Phase 6: Manifest sharding + vLLM preflight (#10–11)
- Phase 7: SPLADE threshold + image flags top-level (#12–13)
- Phase 8: DocTags CLI parity verification (#17–18)

## Impact

### Affected Specs

- **doc-parsing** (existing capability): MODIFIED requirements for atomic writes, schema enforcement, resume logic

### Affected Code

- `src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py` (atomic writes, test cleanup)
- `src/DocsToKG/DocParsing/EmbeddingV2.py` (memory refactor, atomic writes, paths)
- `src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py` (shim, CLI simplification)
- `src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py` (shim, CLI simplification, preflight)
- `src/DocsToKG/DocParsing/_common.py` (UTC fix, hash algorithm support)
- `src/DocsToKG/DocParsing/schemas.py` (optional top-level image fields)
- `src/DocsToKG/DocParsing/cli/` (thin wrapper verification)
- **NEW**: `src/DocsToKG/DocParsing/legacy/` (quarantine directory)

### Breaking Changes

**NONE** - All changes are backward-compatible additions or internal refactorings. Existing JSONL files, CLI invocations, and manifest entries continue to work.

### Migration Path

No migration required. Users can:

- Continue using direct script invocation (deprecated warnings guide to unified CLI)
- Mix old (SHA-1) and new (SHA-256) hashes in manifest (algorithm field distinguishes)
- Read old chunks/vectors without schema_version field (defaults applied)

### Dependencies

- No new dependencies required
- Existing: `pydantic`, `docling`, `vllm`, `sentence-transformers`, `transformers`, `tqdm`, `requests`
