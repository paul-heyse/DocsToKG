# DocParsing Capability Specification Deltas

## MODIFIED Requirements

### Requirement: Atomic File Writes

The chunk and vector writing stages SHALL use atomic write operations to prevent partial file creation on process interruption.

**Modified from**: Direct file writes using `open("w")`

**Changed to**: All output JSONL files (chunks, vectors) MUST be written atomically via temporary file + rename pattern to guarantee either complete or absent output.

#### Scenario: Process interrupted during chunk write

- **WHEN** chunking process receives SIGKILL while writing output JSONL
- **THEN** either complete `.chunks.jsonl` exists OR no file exists (no partial `.chunks.jsonl`)
- **AND** resume logic correctly handles absence by retrying the document

#### Scenario: Resume after crash

- **WHEN** pipeline restarts with `--resume` after interrupted write
- **THEN** incomplete outputs are detected and document is reprocessed
- **AND** no corrupted JSONL rows are present in final output

### Requirement: UTC Timestamp Correctness

Structured JSON logs SHALL emit genuine UTC timestamps, not localtime formatted with Z suffix.

**Modified from**: `formatTime()` using local timezone despite `%Z` format specifier

**Changed to**: JSONFormatter MUST set `converter = time.gmtime` to ensure timestamps truly represent UTC.

#### Scenario: Cross-timezone log correlation

- **WHEN** pipeline runs on host in PST timezone
- **THEN** logged timestamps match `datetime.utcnow()` to within 1 second
- **AND** timestamp format is ISO 8601 with Z suffix (e.g., `2025-10-15T10:30:45.123456Z`)

### Requirement: Content Hash Algorithm Tagging

Manifest entries SHALL record which hash algorithm was used for input content hashing to support algorithm migration.

**Modified from**: `input_hash` field only, algorithm implicit (always SHA-1)

**Changed to**: Manifest entries MUST include both `input_hash` and `hash_alg` fields, with `hash_alg` defaulting to `"sha1"` but respecting `DOCSTOKG_HASH_ALG` environment variable.

#### Scenario: SHA-256 migration

- **WHEN** operator sets `DOCSTOKG_HASH_ALG=sha256` and processes new documents
- **THEN** manifest entries contain `"hash_alg": "sha256"`
- **AND** resume logic correctly distinguishes SHA-1 vs SHA-256 hashes (no false matches)

#### Scenario: Mixed algorithm corpus

- **WHEN** corpus contains documents processed with SHA-1 and SHA-256
- **THEN** resume logic compares hashes only within same algorithm
- **AND** algorithm upgrade does not invalidate existing SHA-1 manifests

### Requirement: CLI Argument Parsing Simplicity

Command-line interfaces SHALL use straightforward argument parsing without "merge defaults + provided" boilerplate.

**Modified from**: ~20-line pattern parsing defaults, parsing provided, merging via setattr loop

**Changed to**: CLI main functions MUST use pattern `args = args if args is not None else parser.parse_args()` for direct parsing or programmatic invocation.

#### Scenario: Programmatic invocation from tests

- **WHEN** test code calls `main(args=Namespace(...))` with pre-constructed args
- **THEN** provided args are used without reparsing or merging
- **AND** all specified values are respected

### Requirement: Embedding Memory Efficiency

Embedding pipeline SHALL process large corpora without retaining full corpus text in memory.

**Modified from**: Pass A builds `uuid_to_chunk` dict with full text for entire corpus

**Changed to**: Pass A MUST discard chunk text immediately after BM25 statistics accumulation. Pass B MUST read chunk text from disk per-file as needed.

#### Scenario: 50K document corpus processing

- **WHEN** embedding pipeline processes 50,000 documents (10GB corpus text)
- **THEN** peak memory usage is < 16GB (bounded by batch size, not corpus size)
- **AND** Pass B completes successfully without OOM errors

#### Scenario: Resume efficiency

- **WHEN** embeddings resume after processing 30K of 50K documents
- **THEN** Pass A re-scans all chunks for BM25 stats (fast, no memory retention)
- **AND** Pass B skips completed files efficiently without loading their text

### Requirement: Portable Model Paths

Model and cache directory paths SHALL be configurable via environment variables and CLI flags, not hardcoded.

**Modified from**: Hardcoded `/home/paul/hf-cache` and model subdirectories

**Changed to**: Paths MUST respect `HF_HOME`, `DOCSTOKG_MODEL_ROOT`, `DOCSTOKG_QWEN_DIR`, `DOCSTOKG_SPLADE_DIR` environment variables with sensible defaults (e.g., `~/.cache/huggingface`). CLI MUST provide `--qwen-model-dir` and `--splade-model-dir` overrides.

#### Scenario: CI environment deployment

- **WHEN** pipeline runs in CI with `HF_HOME=/ci-cache`
- **THEN** all model loading uses `/ci-cache` without code modification
- **AND** no attempts are made to access `/home/paul/`

#### Scenario: Custom model location

- **WHEN** operator runs embeddings with `--qwen-model-dir /models/custom-qwen`
- **THEN** Qwen encoder loads from specified directory
- **AND** default HF cache is ignored for Qwen (but still used for SPLADE if not overridden)

### Requirement: Manifest Scalability via Sharding

Processing manifests SHALL be sharded by pipeline stage to maintain fast resume scan performance on large datasets.

**Modified from**: Single monolithic `docparse.manifest.jsonl` for all stages

**Changed to**: Manifest appends MUST write to stage-specific files (e.g., `docparse.chunks.manifest.jsonl`, `docparse.embeddings.manifest.jsonl`). Resume index loading MUST read only the relevant shard.

#### Scenario: 100K document resume scan

- **WHEN** chunking resumes on 100K-document corpus
- **THEN** `load_manifest_index("chunks")` completes in < 5 seconds
- **AND** only `docparse.chunks.manifest.jsonl` is read (not PDF/HTML conversion entries)

#### Scenario: Backward compatibility

- **WHEN** manifest shard for stage does not exist
- **THEN** loader falls back to reading monolithic `docparse.manifest.jsonl` and filtering by stage
- **AND** old manifests continue working without migration

## ADDED Requirements

### Requirement: Legacy Script Quarantine

Deprecated direct-invocation scripts SHALL be moved to `legacy/` subdirectory with thin shims for backward compatibility.

#### Scenario: Backward-compatible invocation

- **WHEN** user runs `python run_docling_html_to_doctags_parallel.py --help`
- **THEN** deprecation warning is displayed
- **AND** unified CLI is invoked successfully
- **AND** all flags work as expected

#### Scenario: Clear migration path

- **WHEN** user sees deprecation warning
- **THEN** warning message includes exact unified CLI command to use instead
- **AND** documentation links are provided

### Requirement: vLLM Service Preflight Telemetry

PDF conversion SHALL record vLLM service diagnostics to manifest before processing any documents.

#### Scenario: Service readiness audit trail

- **WHEN** PDF pipeline starts and vLLM server is healthy
- **THEN** manifest entry with `doc_id="__service__"` is written
- **AND** entry includes: `served_models`, `vllm_version`, `port`, `metrics_healthy`

#### Scenario: Failure attribution

- **WHEN** conversions fail and service preflight shows vLLM was unhealthy at startup
- **THEN** operator can distinguish service issues from document issues
- **AND** troubleshooting begins with infrastructure, not documents

### Requirement: Offline Operation Support

Embedding and conversion pipelines SHALL support offline operation for air-gapped or network-restricted environments.

#### Scenario: Air-gapped deployment

- **WHEN** embeddings run with `--offline` flag and no network access
- **THEN** transformers library does not attempt network calls
- **AND** models load from local cache or fail fast with clear error

#### Scenario: Model availability check

- **WHEN** Qwen model directory does not exist
- **THEN** pipeline fails immediately with `FileNotFoundError` before GPU allocation
- **AND** error message shows expected model path

### Requirement: Image Metadata Promotion

Chunk rows SHALL include top-level image metadata fields for efficient downstream filtering.

#### Scenario: Quick image-bearing chunk filter

- **WHEN** downstream consumer queries chunks with images
- **THEN** `has_image_captions`, `has_image_classification`, `num_images` are accessible at top level
- **AND** no nested provenance parsing is required
- **AND** provenance metadata is still available for detailed inspection

### Requirement: SPLADE Sparsity Threshold Documentation

Corpus-level SPLADE statistics SHALL include explicit sparsity warning threshold for unambiguous CI alerting.

#### Scenario: Automated quality gate

- **WHEN** embedding pipeline completes and logs corpus summary
- **THEN** manifest includes `sparsity_warn_threshold_pct: 1.0`
- **AND** CI can compare `splade_zero_pct` against threshold programmatically

### Requirement: Schema Version Enforcement

JSONL readers SHALL validate schema version compatibility and fail fast on unsupported versions.

#### Scenario: Mixed schema detection

- **WHEN** downstream consumer reads chunks from directory with mixed schema versions
- **THEN** incompatible schema triggers `ValueError` with version details
- **AND** error message indicates which schema versions are supported

## REMOVED Requirements

*None* - This change only adds capabilities and strengthens existing requirements. No functionality is removed.

## RENAMED Requirements

*None* - Requirement names remain unchanged.
