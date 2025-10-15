# Design Document: DocParsing Pipeline Refactoring

## Context

The DocParsing module implements a three-stage pipeline for converting scientific documents into searchable hybrid vectors:

1. **Stage 1: DocTags Conversion** - HTML/PDF → DocTags format (via Docling)
2. **Stage 2: Chunking** - DocTags → token-normalized chunks with minimum size coalescence
3. **Stage 3: Embedding** - Chunks → hybrid vectors (BM25 + SPLADEv3 + Qwen3-4B dense)

Current implementation has four standalone scripts with duplicated utilities, hard-coded paths, and limited error boundaries. The system works well for small-to-medium datasets but exhibits brittleness (CUDA crashes, OOM failures) and maintenance burden (scattered configuration, duplicated code).

**Stakeholders**: Data engineers running conversions, researchers needing reproducible embeddings, downstream indexing systems consuming JSONL outputs.

**Constraints**:

- Must preserve existing stage boundaries (DocTags ↔ Chunks ↔ Vectors)
- Cannot couple to upstream downloaders or downstream indexers
- Must maintain backward compatibility with existing JSONL consumers
- Local-first architecture: HuggingFace models cached at `/home/paul/hf-cache/`

## Goals / Non-Goals

### Goals

1. **Reduce code duplication** - Consolidate path handling, I/O, logging, and utilities into shared module
2. **Harden robustness** - Add validation, error boundaries, and invariant checks to prevent silent failures
3. **Enable scale** - Stream embeddings to handle datasets exceeding GPU memory
4. **Improve observability** - Centralized structured logging and processing manifest
5. **Unify configuration** - Single source of truth for paths, models, and parameters
6. **Maintain independence** - Keep clean stage boundaries without upstream/downstream coupling

### Non-Goals

- **Not** redesigning the chunking algorithm itself (HybridChunker from docling-core)
- **Not** replacing the embedding models (SPLADE, Qwen remain as-is)
- **Not** integrating with download orchestration or search indexing
- **Not** adding distributed processing (remains single-machine, multi-GPU capable)
- **Not** changing the JSONL output format (only adding optional fields)

## Decisions

### Decision 1: Shared Utilities Module (`_common.py`)

**Choice**: Create single `_common.py` with all cross-cutting functions.

**Alternatives Considered**:

- **Option A**: Leave utilities duplicated across scripts (status quo)
  - ❌ High maintenance burden, drift risk
- **Option B**: Create multiple utility modules by concern (paths, io, logging)
  - ❌ Over-engineering for current scale; hard to discover functions
- **Option C**: Single `_common.py` with clear function grouping ✅
  - ✅ Easy import, single place to look, minimal structure

**Rationale**: Four scripts share 8+ utility functions. Single module reduces from ~120 duplicated LOC to ~60 shared LOC while improving discoverability.

### Decision 2: Two-Pass Streaming for Embeddings

**Choice**: Pass A = UUID + BM25 stats (no text), Pass B = batch encode → shard write.

**Alternatives Considered**:

- **Option A**: Load all chunks into memory (status quo)
  - ❌ OOM on datasets >10K documents
- **Option B**: Fully streaming single-pass with on-disk BM25 index
  - ❌ Complexity; BM25 requires global statistics
- **Option C**: Two-pass streaming: first pass collects stats, second encodes in batches ✅
  - ✅ Bounded memory; simple implementation; enables resume

**Rationale**: BM25 IDF calculation requires knowing document frequency across full corpus. Two passes separate global statistics (fast, memory-light) from encoding (GPU-heavy, streamable).

**Implementation**:

```python
# Pass A: Statistics
for chunk_file in chunk_files:
    rows = load_rows(chunk_file)
    for row in rows:
        ensure_uuid(row)
        accumulate_bm25_stats(row['text'])  # no text retention
    save_rows(chunk_file, rows)

# Pass B: Encoding (per-file batches)
for chunk_file in chunk_files:
    rows = load_rows(chunk_file)
    texts = [r['text'] for r in rows]

    # Batch encode
    for batch in Batcher(texts, batch_size):
        splade_vecs = encode_splade(batch)
        qwen_vecs = encode_qwen(batch)
        write_shard(batch, splade_vecs, qwen_vecs)
```

### Decision 3: Pydantic Schema Validation

**Choice**: Define `ChunkRow` and `VectorRow` Pydantic models; validate on write.

**Alternatives Considered**:

- **Option A**: No validation (status quo)
  - ❌ Silent schema drift, hard-to-debug errors downstream
- **Option B**: JSON Schema files with separate validator
  - ❌ Requires external tooling; not Python-native
- **Option C**: Pydantic models with inline validation ✅
  - ✅ Type-safe, auto-documentation, runtime validation

**Rationale**: Pydantic is already in ecosystem (used by vLLM), provides excellent error messages, enables IDE autocomplete, and serves as living documentation.

**Schema Versioning**: Include `schema_version: str` field (e.g., `"docparse/1.1.0"`) to track evolution. Downstream consumers check version for compatibility.

### Decision 4: Unified CLI with Mode Selection

**Choice**: Single `cli/doctags_convert.py` with `--mode pdf|html|auto`.

**Alternatives Considered**:

- **Option A**: Separate scripts (status quo)
  - ❌ Duplicated arg parsing, documentation, maintenance
- **Option B**: Delete HTML script, make PDF script support both
  - ❌ Mixes concerns; PDF script complex enough
- **Option C**: New unified CLI delegates to backend functions ✅
  - ✅ Single entry point, shared flags, backends remain testable

**Rationale**: 80% of CLI logic is shared (workers, overwrite, paths, logging). Backend functions remain importable for testing; CLI layer is thin dispatcher.

### Decision 5: Spawn Multiprocessing for CUDA Safety

**Choice**: Enforce `multiprocessing.set_start_method('spawn')` in scripts using GPU workers.

**Alternatives Considered**:

- **Option A**: Use 'fork' (Linux default)
  - ❌ CUDA context cannot be forked; crashes workers
- **Option B**: Use 'forkserver'
  - ❌ Adds complexity; not needed
- **Option C**: Use 'spawn' ✅
  - ✅ Safe for CUDA; works cross-platform; slight startup overhead acceptable

**Rationale**: CUDA explicitly forbids forked processes. 'spawn' is the safe default for GPU workloads. Performance impact negligible (startup cost amortized over long-running conversions).

### Decision 6: Topic-Aware Coalescence with Soft Boundaries

**Choice**: Detect structural boundaries (headings, captions); apply relaxed merge constraint.

**Alternatives Considered**:

- **Option A**: Strict length-based merging (status quo)
  - ❌ Breaks topicality; merges unrelated sections
- **Option B**: Never merge across any boundary
  - ❌ Too conservative; leaves many small chunks
- **Option C**: Soft boundary: allow merge if fits within `(max_tokens - 64)` ✅
  - ✅ Balances topicality with minimum size goals

**Rationale**: Embeddings benefit from coherent chunks. 64-token buffer ensures merged chunk doesn't become oversized. Heuristic detection (markdown headers, "Figure caption:", "Table:") is fast and effective.

**Implementation**:

```python
def is_structural_boundary(rec: Rec) -> bool:
    text = rec.text.lstrip()
    if text.startswith('#'):  # Markdown heading
        return True
    if any(text.startswith(kw) for kw in ['Figure caption:', 'Table:']):
        return True
    return False

# In coalesce_small_runs:
if is_structural_boundary(records[k]) and (g.n_tok + records[k].n_tok) > (max_tokens - 64):
    break  # Don't merge across boundary
```

### Decision 7: Manifest-Based Progress Tracking

**Choice**: Append JSON lines to `Data/Manifests/docparse.manifest.jsonl` at key milestones.

**Alternatives Considered**:

- **Option A**: No progress tracking (status quo)
  - ❌ Hard to audit, resume, or debug failures
- **Option B**: SQLite database
  - ❌ Adds dependency; requires schema migrations
- **Option C**: JSONL manifest ✅
  - ✅ Simple, appendable, human-readable, no schema lock-in

**Rationale**: JSONL aligns with existing data formats, tools like `jq` make querying easy, and append-only writes are safe for parallel processing.

**Schema**:

```json
{
  "timestamp": "2025-10-15T10:30:45Z",
  "stage": "doctags",
  "doc_id": "paper_123",
  "status": "success",
  "duration_s": 4.2,
  "warnings": [],
  "schema_version": "docparse/1.1.0",
  "metadata": {
    "parse_engine": "docling-vlm",
    "num_pages": 12
  }
}
```

## Risks / Trade-offs

### Risk 1: Pydantic Dependency Added

**Mitigation**: Pydantic is stable, widely-used, and already a transitive dependency via vLLM. Lock version in `pyproject.toml`.

### Risk 2: Two-Pass Streaming Doubles I/O

**Impact**: Chunk files read twice (Pass A: stats, Pass B: encode).
**Mitigation**: Files are small (<10MB typical); SSD I/O negligible vs GPU encoding time. Benefit (bounded memory) outweighs cost.

### Risk 3: Breaking Changes to JSONL Format

**Mitigation**: Only *add* optional fields (`schema_version`, `provenance`). Existing consumers ignore unknown fields. Document schema evolution in manifest.

### Risk 4: Unified CLI Complexity

**Mitigation**: Keep CLI thin (arg parsing + dispatch); backend functions remain unchanged. Add integration tests for mode switching.

### Risk 5: Refactoring Introduces Regressions

**Mitigation**:

- Preserve original scripts as `*_legacy.py` during transition
- Add golden-path fixtures to lock expected outputs
- Run before/after hash comparison on sample dataset

## Migration Plan

### Phase 1: Infrastructure (Week 1)

1. Create `_common.py`, `schemas.py`, `serializers.py`
2. Add Pydantic to `pyproject.toml`
3. Implement and test utility functions in isolation
4. Create `Data/Manifests/` directory structure

### Phase 2: Refactor Existing Scripts (Week 2)

1. Update chunking script: shared paths, serializers, topic-aware coalescence
2. Update HTML converter: shared utilities, logging
3. Update PDF converter: spawn mode, model validation, enhanced flags
4. Preserve original scripts as `*_legacy.py` with deprecation warnings

### Phase 3: Streaming Embeddings (Week 3)

1. Implement two-pass architecture in `EmbeddingV2.py`
2. Add batch-size flags and shard logic
3. Add invariant checks and validation
4. Test on full-scale dataset

### Phase 4: Unified CLI & Cleanup (Week 4)

1. Implement `cli/doctags_convert.py` with mode dispatch
2. Implement `cli/chunk_and_coalesce.py` wrapper
3. Implement `cli/embed_vectors.py` wrapper
4. Update documentation and README
5. Delete `*_legacy.py` scripts
6. Run full integration test suite

### Rollback Strategy

- Keep original scripts available for 1 release cycle
- Manifest includes schema version for backward compatibility detection
- New features behind flags (`--use-legacy-chunking` escape hatch)

## Open Questions

### Q1: Should we add distributed processing support?

**Decision**: No, not in this change. Single-machine multi-GPU sufficient for current scale. Revisit if datasets exceed 100K documents.

### Q2: Should Qwen embedding dimension be configurable (MRL)?

**Current**: Hard-coded 2560-d output.
**Proposal**: Add `--embedding-dimension` flag for MRL models (e.g., 256, 512, 1024).
**Resolution**: Defer to separate change. Current default works; configurability adds complexity without proven need.

### Q3: Should we support other embedding models (e.g., BGE, E5)?

**Decision**: Not in this change. Keep model selection orthogonal. Current architecture allows swapping models via `--qwen-model-path` flag; validation logic remains generic (dimension check).

### Q4: How to handle schema version mismatches in downstream consumers?

**Proposal**: Consumers should:

1. Parse `schema_version` field
2. Implement version-specific deserialization logic
3. Fail fast on unknown versions with clear error message

**Example**:

```python
def load_chunks(path: Path) -> List[ChunkRow]:
    rows = jsonl_load(path)
    for row in rows:
        version = row.get('schema_version', 'docparse/1.0.0')  # default for old files
        if version.startswith('docparse/1.'):
            yield ChunkRow.parse_obj(row)
        else:
            raise ValueError(f"Unsupported schema version: {version}")
```

### Q5: Should manifest be queryable via CLI tool?

**Proposal**: Add `scripts/query_manifest.py` with filters (stage, status, date range).
**Resolution**: Defer to separate change. Users can use `jq` for now; dedicated tool adds value but not blocking.

## Success Metrics

1. **Code Reduction**: ≥100 lines removed via consolidation
2. **Memory Usage**: Embeddings stage uses <80% GPU memory at all times (down from OOM failures)
3. **Robustness**: Zero CUDA re-initialization crashes in CI
4. **Observability**: 100% of processing tracked in manifest
5. **Validation Coverage**: 100% of JSONL rows pass Pydantic validation
6. **Test Coverage**: ≥85% coverage for new utility modules
7. **Performance**: End-to-end runtime within 5% of baseline (slight I/O overhead acceptable)

## References

- Docling documentation: <https://github.com/DS4SD/docling>
- vLLM multiprocessing guide: <https://docs.vllm.ai/en/latest/>
- Pydantic validation: <https://docs.pydantic.dev/>
- BM25 algorithm: Robertson & Zaragoza (2009)
- SPLADE-v3: <https://huggingface.co/naver/splade-v3>
- Qwen3-Embedding: <https://huggingface.co/Qwen/Qwen3-Embedding-4B>

## Error Handling Strategy

### Error Classification

Errors are categorized into four severity levels with distinct handling strategies:

1. **Fatal Errors** - Stop processing immediately, rollback partial writes
   - Examples: CUDA re-initialization, invalid configuration, missing dependencies
   - Action: Exit with non-zero code, log full stack trace, clean up resources

2. **Document-Level Errors** - Skip document, continue pipeline
   - Examples: Malformed DocTags, empty documents, validation failures
   - Action: Log warning, write manifest entry with status="failure", continue

3. **Recoverable Errors** - Retry with backoff, then skip
   - Examples: Network timeouts, temporary file I/O errors
   - Action: Retry up to 3 times with exponential backoff (1s, 2s, 4s)

4. **Warnings** - Log but don't fail
   - Examples: Oversized chunks, missing optional fields, deprecated usage
   - Action: Append to warnings list in manifest

### Error Code Conventions

Exit codes follow standard Unix conventions plus custom codes:

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | All documents processed |
| 1 | General error | Invalid arguments |
| 2 | Configuration error | Invalid parameter combination |
| 137 | OOM (128+9) | CUDA out of memory |
| 143 | SIGTERM (128+15) | Graceful shutdown requested |

### Exception Hierarchy

```python
class DocParsingError(Exception):
    """Base exception for DocParsing pipeline."""
    pass

class ConfigurationError(DocParsingError):
    """Invalid configuration or arguments."""
    pass

class ValidationError(DocParsingError):
    """Schema validation failed."""
    pass

class ProcessingError(DocParsingError):
    """Document processing failed."""
    pass
```

## Performance Targets and Benchmarks

### Throughput Targets

Based on empirical testing with representative documents:

| Stage | Target Throughput | Hardware Requirement |
|-------|-------------------|---------------------|
| HTML → DocTags | 30-50 docs/min | 12-core CPU |
| PDF → DocTags (vLLM) | 5-10 docs/min | 1x A100 GPU |
| Chunking | 10-20 docs/min | 8-core CPU |
| Embeddings (streaming) | 5-8 docs/min | 1x A100 GPU |

### Memory Budgets

| Component | Max Memory | Notes |
|-----------|------------|-------|
| Chunking | <4GB RAM | Per-document processing |
| SPLADE encoding | <6GB VRAM | Batch size 32 |
| Qwen encoding | <12GB VRAM | Batch size 64 |
| BM25 statistics (Pass A) | <8GB RAM | Scales with corpus size |
| Pass B streaming | <16GB RAM | Bounded by batch size |

### Performance Monitoring Points

Insert timing instrumentation at:
- Script entry/exit (total runtime)
- Per-document start/end (individual throughput)
- Model loading (initialization overhead)
- Batch encoding (GPU utilization)
- File I/O operations (disk bottlenecks)

### Optimization Opportunities

If performance targets not met:

1. **Chunking bottleneck**: Parallelize document-level processing (already implemented)
2. **Embedding bottleneck**: Increase batch size up to memory limit
3. **I/O bottleneck**: Use SSD storage, increase buffer sizes
4. **vLLM bottleneck**: Increase `--gpu-memory-utilization` or add tensor parallelism

## Operational Considerations

### Deployment Patterns

**Single-Machine Setup** (recommended baseline):
- Input: Local `Data/` directory
- Processing: Sequential stages with resume capability
- Output: Local JSONL files
- Monitoring: Tail manifest file

**Distributed Setup** (future):
- Input: Shared filesystem (NFS, S3)
- Processing: Multiple workers via job queue
- Output: Shared storage with advisory locking
- Monitoring: Centralized manifest aggregation

### Capacity Planning

To process N documents:

1. **Estimate total time**:
   - DocTags: N / (throughput * worker_count)
   - Chunking: N / throughput
   - Embeddings: N / (throughput * gpu_count)

2. **Estimate storage**:
   - DocTags: ~200KB per document
   - Chunks: ~100KB per document
   - Vectors: ~500KB per document (BM25 + SPLADE + Qwen)
   - Manifest: ~1KB per document

3. **Resource requirements**:
   - CPU: 16 cores recommended
   - RAM: 32GB minimum
   - GPU: 1x A100 (40GB) or 2x RTX 4090
   - Disk: 1GB per 1000 documents

### Monitoring and Alerting

**Key Metrics to Track**:
- Documents processed per stage (gauge)
- Processing rate (docs/min, rate)
- Failure rate by stage (percentage)
- Average duration per document (histogram)
- Peak memory usage (gauge)
- GPU utilization (percentage)

**Alert Conditions**:
- Failure rate >5% in last hour → Warning
- Failure rate >20% in last hour → Critical
- Processing stalled (no progress in 10 min) → Warning
- Disk usage >90% → Warning
- GPU memory usage >95% → Critical

**Manifest Query Examples**:

```bash
# Failure rate in last hour
jq -r 'select(.timestamp > "'$(date -u -d '1 hour ago' -Iseconds)'") | select(.status=="failure")' manifest.jsonl | wc -l

# Average duration by stage
jq -s 'group_by(.stage) | map({stage: .[0].stage, avg_duration: (map(.duration_s) | add / length)})' manifest.jsonl

# Top error types
jq -r 'select(.status=="failure") | .error' manifest.jsonl | sort | uniq -c | sort -rn | head -10
```

### Maintenance Procedures

**Weekly Tasks**:
- Review manifest for failure patterns
- Check disk usage, archive old files if needed
- Validate random sample of outputs

**Monthly Tasks**:
- Run full golden-path test suite
- Update performance baselines
- Review and optimize slow documents

**Quarterly Tasks**:
- Update dependencies (Docling, vLLM, transformers)
- Re-run tokenizer calibration on corpus
- Archive completed manifest entries

### Disaster Recovery

**Data Loss Prevention**:
- All writes are atomic (tmp → rename)
- Manifest is append-only (no corruption risk)
- Lock files prevent concurrent writes
- Resume capability enables restart from failure

**Recovery Procedures**:

1. **Partial file detected**: Remove `.tmp` files, re-run with `--resume`
2. **Corrupted manifest**: Split by valid JSON lines, rebuild
3. **Stale lock files**: Check PID, remove if process dead
4. **Missing outputs**: Query manifest for successful entries, compare with filesystem

## Testing Strategy

### Test Pyramid

1. **Unit Tests** (fast, isolated):
   - Utility functions (_common.py): 90% coverage
   - Schema validation (schemas.py): 95% coverage
   - Serializers: 85% coverage
   - Target runtime: <1 second total

2. **Integration Tests** (moderate, end-to-end stage):
   - Chunking with topic-aware coalescence
   - Streaming embeddings (small corpus)
   - vLLM lifecycle management
   - Target runtime: <30 seconds total

3. **System Tests** (slow, full pipeline):
   - Complete HTML → Vectors flow
   - Complete PDF → Vectors flow
   - Resume functionality
   - Target runtime: <5 minutes total

### Golden Fixtures

Requirements for golden test data:
- **sample.doctags**: 5-page document with headings, figures, tables, captions
- **sample.chunks.jsonl**: Expected chunks (deterministic tokenization)
- **sample.vectors.jsonl**: Expected vectors (deterministic with fixed random seed)
- All committed to git for reproducibility

### Property-Based Testing

Use Hypothesis library to test invariants:
- **Chunking**: `min_tokens <= chunk.num_tokens <= max_tokens` (except last)
- **Coalescence**: No adjacent small runs remain
- **BM25**: `len(terms) == len(weights)`
- **SPLADE**: `all(w >= 0 for w in weights)`

### CI/CD Integration

GitHub Actions workflow:
```yaml
name: DocParsing Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e . && pip install pytest pytest-cov hypothesis
      - name: Run tests
        run: pytest tests/test_docparsing_*.py -v --cov=src/DocsToKG/DocParsing --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Appendix: Detailed Function Signatures

### _common.py

```python
def detect_data_root(start: Optional[Path] = None) -> Path:
    """Locate DocsToKG Data directory via env var or ancestor scan."""
    ...

def data_doctags(root: Optional[Path] = None) -> Path:
    """Return DocTagsFiles directory, creating if needed."""
    ...

def data_chunks(root: Optional[Path] = None) -> Path:
    """Return ChunkedDocTagFiles directory, creating if needed."""
    ...

def data_vectors(root: Optional[Path] = None) -> Path:
    """Return Vectors directory, creating if needed."""
    ...

def data_manifests(root: Optional[Path] = None) -> Path:
    """Return Manifests directory, creating if needed."""
    ...

def find_free_port(start: int = 8000, span: int = 32) -> int:
    """Find available TCP port on localhost."""
    ...

@contextlib.contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Context manager for atomic file writes."""
    ...

def iter_doctags(directory: Path) -> Iterator[Path]:
    """Yield all .doctags files in directory tree."""
    ...

def iter_chunks(directory: Path) -> Iterator[Path]:
    """Yield all .chunks.jsonl files in directory."""
    ...

def jsonl_load(path: Path, skip_invalid: bool = False, max_errors: int = 10) -> List[dict]:
    """Load JSONL file with optional error tolerance."""
    ...

def jsonl_save(path: Path, rows: List[dict], validate: Optional[Callable[[dict], None]] = None) -> None:
    """Save JSONL file atomically with optional validation."""
    ...

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get configured logger with JSON formatting."""
    ...

class Batcher(Generic[T]):
    """Yield fixed-size batches from iterable."""
    def __init__(self, iterable: Iterable[T], batch_size: int): ...
    def __iter__(self) -> Iterator[List[T]]: ...

def manifest_append(stage: str, doc_id: str, status: str, 
                   duration_s: float = 0.0, warnings: Optional[List[str]] = None,
                   error: Optional[str] = None, schema_version: str = "",
                   **metadata) -> None:
    """Append processing record to manifest."""
    ...

def compute_content_hash(path: Path, algorithm: str = "sha1") -> str:
    """Compute file content hash for change detection."""
    ...

@contextlib.contextmanager
def acquire_lock(path: Path, timeout: float = 60.0) -> Iterator[bool]:
    """Acquire advisory lock on file."""
    ...
```

### schemas.py

```python
CHUNK_SCHEMA_VERSION = "docparse/1.1.0"
VECTOR_SCHEMA_VERSION = "embeddings/1.0.0"
COMPATIBLE_CHUNK_VERSIONS = ["docparse/1.0.0", "docparse/1.1.0"]
COMPATIBLE_VECTOR_VERSIONS = ["embeddings/1.0.0"]

class ProvenanceMetadata(BaseModel):
    parse_engine: str  # "docling-html" or "docling-vlm"
    docling_version: str
    has_image_captions: bool = False
    has_image_classification: bool = False
    num_images: int = 0

class ChunkRow(BaseModel):
    doc_id: str
    source_path: str
    chunk_id: int
    source_chunk_idxs: List[int]
    num_tokens: int
    text: str
    doc_items_refs: List[str] = []
    page_nos: List[int] = []
    schema_version: str = CHUNK_SCHEMA_VERSION
    provenance: Optional[ProvenanceMetadata] = None
    uuid: Optional[str] = None

class BM25Vector(BaseModel):
    terms: List[str]
    weights: List[float]
    k1: float = 1.5
    b: float = 0.75
    avgdl: float
    N: int

class SPLADEVector(BaseModel):
    model_id: str = "naver/splade-v3"
    tokens: List[str]
    weights: List[float]

class DenseVector(BaseModel):
    model_id: str
    vector: List[float]
    dimension: Optional[int] = None

class VectorRow(BaseModel):
    UUID: str
    BM25: BM25Vector
    SPLADEv3: SPLADEVector
    Qwen3_4B: DenseVector = Field(alias="Qwen3-4B")
    model_metadata: Optional[Dict[str, Any]] = {}
    schema_version: str = VECTOR_SCHEMA_VERSION

def validate_chunk_row(row: dict) -> ChunkRow:
    """Validate and parse chunk JSONL row."""
    ...

def validate_vector_row(row: dict) -> VectorRow:
    """Validate and parse vector JSONL row."""
    ...

def get_docling_version() -> str:
    """Detect installed docling package version."""
    ...

def validate_schema_version(version: str, compatible_versions: List[str]) -> bool:
    """Check if schema version is compatible."""
    ...
```

These function signatures serve as API contracts for implementation.
