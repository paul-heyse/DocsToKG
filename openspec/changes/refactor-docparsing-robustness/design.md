# Design Document: DocParsing Robustness Refactoring

## Context

The DocsToKG document parsing system processes raw document inputs (HTML, PDF) through a multi-stage pipeline: conversion to DocTags format, chunking with token-aware coalescence, and embedding generation via BM25, SPLADE, and Qwen models. Each stage produces intermediate artifacts and records processing outcomes in manifest files for observability and resume capability.

Current implementation exhibits several architectural inconsistencies accumulated through iterative development. Document identifiers use file basenames which risk collision when subdirectory structures contain duplicate names. Output paths flatten hierarchical inputs causing overwrites. Schema definitions contain inadvertent duplicate fields. Model initialization repeats expensively. Configuration embeds hard-coded paths. Logging mixes structured JSON with unstructured print statements.

These issues undermine key system properties: auditability requires stable identifiers across stages; resumability depends on reliable change detection; observability needs consistent log formats; portability demands configuration externalization.

**Stakeholders**: Pipeline operators, embedding infrastructure maintainers, downstream search indexers, continuous integration systems.

**Constraints**:

- Must maintain backward compatibility for existing chunk and vector schemas
- Cannot introduce new external dependencies
- Must preserve multiprocessing CUDA safety guarantees
- Should minimize performance regression while optimizing where practical

## Goals

**Primary Goals**:

1. Establish canonical document identifiers using relative paths eliminating collision ambiguity
2. Prevent output path collisions through directory structure mirroring
3. Remove schema validation ambiguities from duplicate field declarations
4. Improve embedding performance through LLM instance caching
5. Standardize logging to enable uniform machine parsing
6. Externalize configuration to support portable deployment

**Non-Goals**:

- Changing chunk or vector schema field names or structure
- Modifying stage boundary interfaces or adding new pipeline stages
- Implementing distributed processing or horizontal scaling
- Altering tokenization algorithms or chunking strategies
- Adding new embedding models or vector representations

## Decisions

### Decision 1: Relative Path as Canonical Identifier

**Choice**: Use `file_path.relative_to(input_dir).as_posix()` as canonical doc_id throughout pipeline

**Rationale**:

- Relative paths guarantee uniqueness across subdirectory hierarchies
- POSIX format ensures cross-platform consistency
- Aligns with existing manifest practice reducing impedance
- Enables reliable join operations between stages for auditing

**Alternatives Considered**:

- **UUID generation**: Would require persistent mapping infrastructure and complicate human debugging
- **Content-based hashing**: Creates chicken-egg problem for resume logic and obscures source relationships
- **Maintain basename with collision detection**: Adds runtime complexity without solving root cause

**Trade-offs**:

- Longer identifiers in manifests and logs (acceptable given modern storage)
- Requires relative path computation overhead (negligible compared to I/O)
- Migration complexity for systems expecting short identifiers (mitigated by gradual rollout)

### Decision 2: Directory Mirroring for Output Paths

**Choice**: Mirror input directory structure in output locations via `(output_dir / relative_path).with_suffix()`

**Rationale**:

- Prevents filename collisions by construction
- Matches existing HTML pipeline behavior reducing cognitive load
- Preserves semantic relationships between source organization and outputs
- Enables intuitive output navigation mirroring source layout

**Alternatives Considered**:

- **Flat output with name mangling**: Creates opaque filenames and complicates debugging
- **Hash-based subdirectory sharding**: Loses semantic structure and requires lookup tables
- **Database-backed path registry**: Introduces external dependency and failure mode

**Trade-offs**:

- Deeper directory trees may approach path length limits on some filesystems (unlikely in practice)
- Requires parent directory creation logic in workers (simple implementation)
- Changes output layout requiring documentation update (one-time migration)

### Decision 3: Module-Level LLM Caching

**Choice**: Cache LLM instances in module-global dictionary keyed by configuration tuple

**Rationale**:

- Eliminates redundant model loading during single-process Pass B execution
- Simple implementation with no external dependencies
- Automatically cleans up on process exit via garbage collection
- Configuration-keyed cache safely handles multiple model variants

**Alternatives Considered**:

- **No caching**: Wastes 10-30 seconds per file on model initialization
- **Process pool with pre-warmed workers**: Complicates resource management and adds failure modes
- **Shared memory LLM sharing**: Requires complex synchronization and vLLM doesn't support it cleanly

**Trade-offs**:

- Increases process memory footprint by one LLM instance (acceptable given existing embedding memory requirements)
- Cache persists for process lifetime (desired behavior for batch processing)
- Not safe for hypothetical future multi-process Pass B (document for future work)

### Decision 4: Environment-Driven Model Resolution

**Choice**: Resolve model paths via precedence chain: CLI flag > specific env var > general cache env var > conventional default

**Rationale**:

- Eliminates hard-coded workstation paths enabling portable deployment
- Follows HuggingFace ecosystem conventions reducing learning curve
- Provides granular override at multiple levels for different use cases
- Makes resolution explicit through logging for troubleshooting

**Alternatives Considered**:

- **Configuration file**: Adds deployment complexity and file format decisions
- **Command-line only**: Forces every invocation to specify paths reducing usability
- **Auto-discovery scan**: Fragile with unpredictable behavior and performance implications

**Trade-offs**:

- Operators must understand precedence rules (mitigated by clear documentation)
- Environment variables persist across sessions requiring careful management (standard practice)
- Offline deployments require explicit configuration (desirable behavior)

### Decision 5: Deprecation Path for Legacy Modules

**Choice**: Emit DeprecationWarning on legacy imports, remove shims in next major release

**Rationale**:

- Provides migration window for external consumers
- Aligns with Python community deprecation practices
- Enables measurement of legacy usage through warning counts
- Reduces long-term maintenance burden by converging on single code path

**Alternatives Considered**:

- **Immediate removal**: Breaks existing code without warning
- **Permanent compatibility shims**: Perpetuates technical debt indefinitely
- **Capability detection**: Overly complex for simple module forwarding

**Trade-offs**:

- Requires two-phase rollout (warnings then removal)
- Test suites must tolerate deprecation warnings temporarily
- Documentation must explain both current and legacy patterns during transition

## Architecture Patterns

### Identifier Canonicalization Pattern

All stages shall compute document identifier once during file discovery as:

```
relative_path = source_path.relative_to(input_directory)
doc_id = relative_path.as_posix()
```

This identifier flows through:

1. Task construction (PDF/HTML pipeline task objects)
2. Manifest recording (all manifest_append calls)
3. Data model fields (ChunkRow.doc_id)
4. Cache key construction (resume logic)

Stem-based names remain for local purposes:

- Output filename generation (avoid directory nesting in filenames)
- Progress bar display (reduce visual clutter)
- Log message context (improve human readability)

### Output Path Construction Pattern

All conversion workers shall construct outputs as:

```
relative_path = source_path.relative_to(input_directory)
output_path = output_directory / relative_path
output_path = output_path.with_suffix(target_extension)
output_path.parent.mkdir(parents=True, exist_ok=True)
```

Applied uniformly across:

- PDF conversion tasks in pdf_main
- HTML conversion tasks in html_main
- Any future conversion pipeline additions

### Caching Pattern

Module-level caches for expensive objects shall follow:

```
_CACHE: Dict[KeyTuple, ExpensiveObject] = {}

def get_or_create(config):
    key = make_key(config)
    if key not in _CACHE:
        _CACHE[key] = create(config)
    return _CACHE[key]
```

Applied to:

- Qwen LLM instances (new)
- SPLADE encoders (existing pattern)
- Future expensive initialization targets

Cache keys must include all configuration affecting object behavior. Cache cleanup occurs automatically via process exit.

### Configuration Resolution Pattern

Model path resolution shall follow precedence chain:

```
1. Explicit CLI argument (--model) - highest precedence
2. Model-specific environment variable (DOCLING_PDF_MODEL)
3. General model root (DOCSTOKG_MODEL_ROOT) + conventional subpath
4. HuggingFace cache (HF_HOME) + conventional subpath
5. User home default ~/.cache/huggingface + conventional subpath
```

Resolved paths logged at INFO level for operational visibility.

## Migration Plan

### Phase 1: Core Correctness (Single PR)

**Changes**:

- Document identifier canonicalization in chunker
- PDF output path mirroring
- Schema duplicate field removal
- Variable deduplication in EmbeddingV2

**Validation**:

- Integration tests confirm no output schema changes
- Manifest entries use relative path identifiers
- PDF subdirectory outputs function correctly
- Chunk schema validation passes without warnings

**Rollback**: If critical issues surface, revert single PR. Existing data remains valid.

### Phase 2: Performance and Portability (Single PR)

**Changes**:

- LLM caching in EmbeddingV2
- Model path environment resolution
- Multiprocessing helper consolidation

**Validation**:

- Performance benchmarks show initialization time reduction
- Pipeline starts successfully with various environment configurations
- Offline mode works with local caches

**Rollback**: Independent from Phase 1, safe to revert independently.

### Phase 3: Cleanup (Single PR)

**Changes**:

- Structured logging replacements
- Directory naming alignment
- Legacy module deprecation warnings

**Validation**:

- All log output parses as JSON
- Documentation matches code directory names
- Legacy imports trigger deprecation warnings

**Rollback**: Cosmetic changes, low risk to revert.

### Phase 4: Removal (Future Release)

**Changes**:

- Delete legacy module shim code
- Remove pdf_pipeline.py test helper

**Validation**:

- No internal code references removed modules
- External users had deprecation warning period

**Rollback**: Not applicable, planned removal after deprecation period.

### Data Migration

**Chunk Files**: No migration required. New runs produce relative path doc_ids; old files with stem-based identifiers remain valid for embedding stage which reads identifiers from row data.

**PDF Outputs**: First run after Phase 1 deployment creates mirrored subdirectories. Existing flat outputs remain accessible. Optional cleanup script can reorganize historical outputs to mirror structure if desired.

**Vector Files**: No migration required. Vector rows reference chunk UUIDs not doc_ids, maintaining independence from identifier scheme changes.

**Manifests**: New entries use relative path identifiers. Legacy entries with stem-based identifiers remain readable. Manifest loading logic already treats doc_id as opaque string enabling gradual transition.

## Risks and Mitigations

### Risk: Identifier Length Impacts Database Systems

**Likelihood**: Low
**Impact**: Medium
**Mitigation**: Modern databases handle VARCHAR fields efficiently. Relative paths rarely exceed 200 characters. Monitor manifest index query performance post-deployment.

### Risk: Directory Mirroring Hits Path Length Limits

**Likelihood**: Very Low
**Impact**: Medium
**Mitigation**: POSIX PATH_MAX is 4096 bytes. Practical document hierarchies rarely exceed 10 levels. Add validation warning if path approaches limits.

### Risk: Cache Memory Pressure with Multiple Model Variants

**Likelihood**: Low
**Impact**: Low
**Mitigation**: Typical deployments use single model configuration. Cache stores one instance per unique config. Document memory implications for multi-variant scenarios.

### Risk: Environment Variable Precedence Confusion

**Likelihood**: Medium
**Impact**: Low
**Mitigation**: Comprehensive documentation with decision tree diagram. Log resolved paths prominently. Provide diagnostic command to show resolution without execution.

### Risk: Legacy Import Deprecation Breaks External Tools

**Likelihood**: Medium
**Impact**: Medium
**Mitigation**: DeprecationWarning provides advance notice. Monitor warning metrics. Communicate deprecation in release notes. Provide migration examples.

### Risk: Resume Logic Breaks with Identifier Change

**Likelihood**: Medium
**Impact**: Medium
**Mitigation**: Manifest comparison uses input_hash as primary signal. doc_id serves as index key. Hash-based resume works regardless of identifier format. Test resume scenarios explicitly.

## Open Questions

1. **Directory naming**: Should canonical name be "Vectors" (shorter) or "Embeddings" (more descriptive)?
   - **Resolution approach**: Survey stakeholders for preference. Check existing deployment scripts. Choose based on least-surprise principle.

2. **Migration script for PDF outputs**: Should we provide automated reorganization tool for historical flat outputs?
   - **Resolution approach**: Assess pain of manual vs automated migration. If corpus is small, document manual process. If large, provide optional script.

3. **Cache eviction policy**: Should LLM cache support explicit eviction for long-running services?
   - **Resolution approach**: Current batch processing model makes this low priority. Document as future enhancement if service-style deployment emerges.

4. **Parallel processing**: Should Phase 2 include readiness for multi-process Pass B?
   - **Resolution approach**: Not in scope. Current single-process Pass B performs adequately. Document cache safety constraints for future work.

## Success Metrics

**Correctness**:

- Zero basename collision incidents in production logs
- 100% manifest doc_id to chunk row doc_id alignment
- Zero Pydantic validation warnings from schema definitions

**Performance**:

- 15-30 second reduction in per-file embedding time from LLM caching
- No regression in chunking or conversion throughput

**Operational**:

- Pipeline starts successfully on fresh CI runners without hard-coded path failures
- 100% of log output parseable as JSON
- Zero questions from operators about output directory naming

**Adoption**:

- All internal code migrated from legacy imports within one sprint
- Deprecation warning count approaches zero after notification period
