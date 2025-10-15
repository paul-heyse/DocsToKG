# DocParsing Pipeline Refactoring and Robustness Enhancements

## Why

The current DocParsing pipeline (`src/DocsToKG/DocParsing`) implements a solid three-stage architecture (DocTags → Chunking → Embeddings) but suffers from scattered utilities, hard-coded paths, CUDA initialization risks in multiprocessing, potential OOM failures at scale, and insufficient validation. These issues create maintenance burden, reduce robustness, and make the system brittle under load or when configuration changes.

This refactoring consolidates duplicated code, hardens error boundaries with explicit validation, enables streaming to handle large-scale datasets, and centralizes configuration—all while preserving the existing stage boundaries and avoiding coupling to upstream downloaders or downstream indexers.

## What Changes

### Code Organization & Reduction

- **NEW**: Create `_common.py` utility module with shared functions (path resolution, atomic writes, JSONL I/O, logging, batching, port detection)
- **NEW**: Move serializers (`CaptionPlusAnnotationPictureSerializer`, `RichSerializerProvider`) to shared module
- **REFACTOR**: Unify path handling across all scripts via `detect_data_root()` with `DOCSTOKG_DATA_ROOT` env var support
- **REFACTOR**: Replace hard-coded absolute paths with dynamic resolution in `DoclingHybridChunkerPipelineWithMin.py`
- **REFACTOR**: Consolidate HTML and PDF conversion CLIs into single `cli/doctags_convert.py` with `--mode` flag

### Robustness & Safety

- **FIX**: Force multiprocessing start method to `spawn` in PDF conversion script to prevent CUDA re-initialization errors
- **NEW**: Add Pydantic schema models for `ChunkRow` and `VectorRow` with validation
- **NEW**: Implement schema versioning (e.g., `docparse/1.1.0`) in all JSONL outputs
- **NEW**: Add invariant checks for embeddings (Qwen dimension assertions, SPLADE nnz validation, BM25 statistics)
- **NEW**: Implement idempotent outputs with `.lock` sentinels to prevent race conditions
- **NEW**: Add content-based hashing (`content_sha1`) for reproducibility verification

### Scalability & Memory Management

- **REFACTOR**: Convert `EmbeddingV2.py` to two-pass streaming architecture:
  - Pass A: UUID assignment + BM25 statistics (no text retention)
  - Pass B: Batch-wise SPLADE/Qwen encoding with shard outputs
- **NEW**: Add `--batch-size-splade` and `--batch-size-qwen` configuration flags
- **NEW**: Implement atomic shard merging for vector outputs
- **NEW**: Add resume capability to skip already-processed files

### Tokenization & Alignment

- **NEW**: Add `--tokenizer-model` flag to chunking script
- **NEW**: Align chunk token counting with dense embedding model tokenizer (Qwen3-4B)
- **NEW**: Provide calibration guidance for BERT vs Qwen tokenizer discrepancies

### Chunking Enhancements

- **ENHANCE**: Implement topic-aware boundary detection in `coalesce_small_runs()` to avoid merging across headings/captions
- **NEW**: Add soft barrier rule: require merged size ≤ (max_tokens - 64) for structural boundaries

### Logging & Observability

- **NEW**: Adopt structured logging with JSON formatter across all scripts
- **NEW**: Create unified progress ledger at `Data/Manifests/docparse.manifest.jsonl`
- **NEW**: Record per-document metadata (stage, duration, warnings, schema_version)
- **NEW**: Add per-row provenance enrichment (parse_engine, docling_version, image flags)

### vLLM Server Management

- **ENHANCE**: Add `--model`, `--served-model-name`, and `--gpu-memory-utilization` flags to PDF converter
- **NEW**: Implement model validation before processing (verify served model in `/v1/models`)
- **ENHANCE**: Strengthen health check to require non-empty models list for reuse
- **NEW**: Add explicit lifecycle guards and error messages for server failures

### Testing & Quality

- **NEW**: Add golden-path fixtures for deterministic chunk count validation
- **NEW**: Create trip-wire CI tests for coalescer invariants, empty document handling, and embedding shape validation
- **NEW**: Implement corpus summary reporting (N chunks, avgdl, median token length, SPLADE zero percentage)

## Impact

### Affected Specs

- **NEW**: `doc-parsing` (new capability specification)

### Affected Code

- `src/DocsToKG/DocParsing/DoclingHybridChunkerPipelineWithMin.py` (path handling, tokenizer alignment, topic-aware coalescence)
- `src/DocsToKG/DocParsing/EmbeddingV2.py` (streaming architecture, validation, batching)
- `src/DocsToKG/DocParsing/run_docling_parallel_with_vllm_debug.py` (spawn safety, model validation, enhanced flags)
- `src/DocsToKG/DocParsing/run_docling_html_to_doctags_parallel.py` (unified path handling, shared utilities)
- **NEW**: `src/DocsToKG/DocParsing/_common.py` (shared utilities module)
- **NEW**: `src/DocsToKG/DocParsing/cli/doctags_convert.py` (unified CLI entry point)
- **NEW**: `src/DocsToKG/DocParsing/cli/chunk_and_coalesce.py` (refactored chunking CLI)
- **NEW**: `src/DocsToKG/DocParsing/cli/embed_vectors.py` (refactored embedding CLI)
- **NEW**: `src/DocsToKG/DocParsing/serializers.py` (extracted serializers)

### Data Files

- **NEW**: `Data/Manifests/docparse.manifest.jsonl` (processing ledger)
- **MODIFIED**: All chunk JSONL files (add schema_version, provenance fields)
- **MODIFIED**: All vector JSONL files (add schema_version, model metadata)

### Breaking Changes

- **NONE** - All changes are backward-compatible; existing JSONL files remain readable

### Dependencies

- **NEW**: `pydantic` (for schema validation)
- Existing: `docling`, `docling-core`, `vllm`, `sentence-transformers`, `transformers`, `tqdm`, `requests`

## Dependencies

### New Dependencies
- **pydantic** (>=2.0, <3.0): Required for schema validation
  - Rationale: Industry-standard validation library, already transitive dependency via vLLM
  - Risk: None - stable and well-maintained
  - License: MIT

### Updated Dependencies
- **docling**: Keep current version (>=1.0)
- **docling-core**: Keep current version (>=1.0)
- **vllm**: Keep current version (>=0.3.0)
- **sentence-transformers**: Keep current version
- **transformers**: Keep current version

### System Dependencies
- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- CUDA 11.8+ for GPU operations
- 32GB RAM minimum (64GB recommended for large corpora)
- 1x A100 GPU (40GB) or equivalent

## Security Considerations

### Data Privacy
- **No external API calls**: All processing is local-first
- **No credential storage**: No authentication required for pipeline
- **Manifest contains no PII**: Only document IDs, not content

### Input Validation
- **Schema validation**: Pydantic models enforce strict types
- **Path traversal protection**: All file operations validate paths are within data root
- **JSONL injection**: Line-by-line parsing prevents multiline injection
- **Size limits**: Chunk token count capped at 100K (sanity check)

### Process Isolation
- **Multiprocessing spawn mode**: Workers cannot access parent memory
- **Lock files include PID**: Prevents accidental concurrent writes
- **Atomic writes**: Tmp files prevent partial outputs

### Logging Security
- **No sensitive data in logs**: Model paths and file paths only
- **Redaction for credentials**: If future auth added, credentials will be redacted
- **Structured logs**: JSON format prevents log injection attacks

## Testing Strategy

### Unit Tests
- **Coverage Target**: ≥90% for _common.py, ≥95% for schemas.py
- **Frameworks**: pytest, pytest-cov, hypothesis (property-based testing)
- **Fixtures**: Golden DocTags, chunks, vectors committed to git
- **Assertions**: Deterministic outputs, schema compliance, invariant checks

### Integration Tests
- **Scope**: Single-stage pipelines (DocTags→Chunks, Chunks→Vectors)
- **Test Data**: Representative sample documents (HTML, PDF, multi-page)
- **Validations**: Output file format, row counts, processing time
- **Error Scenarios**: Malformed inputs, missing files, invalid configurations

### System Tests
- **Scope**: Full end-to-end pipeline (PDF→DocTags→Chunks→Vectors)
- **Test Data**: Small corpus (10 documents)
- **Validations**: Manifest completeness, schema compliance, resource usage
- **Performance**: Throughput meets minimum targets

### Backward Compatibility Tests
- **Old chunk files**: Verify reading without schema_version field
- **Old vector files**: Verify reading without provenance metadata
- **Schema migration**: Verify adding schema_version to old files preserves data

### Regression Tests
- **Golden fixtures**: Compare current outputs to committed expected outputs
- **Performance baselines**: Ensure <10% runtime regression, <20% memory regression
- **Hash comparison**: Verify deterministic outputs produce identical hashes

## Rollback Plan

### Phase-by-Phase Rollback

**If Issues in Phase 1 (Infrastructure)**:
- Delete new modules (_common.py, schemas.py, serializers.py)
- Revert to commit before refactoring started
- No impact on production - infrastructure not yet integrated

**If Issues in Phase 2 (Script Refactoring)**:
- Restore *_legacy.py scripts as primary
- Update documentation to point to legacy scripts
- New modules remain but unused
- Production continues with original scripts

**If Issues in Phase 3 (Streaming Embeddings)**:
- Revert EmbeddingV2.py to pre-refactor version
- Keep other refactorings (shared utilities, schemas)
- Partial benefit retained (code reduction, validation)

**If Issues in Phase 4 (Unified CLI)**:
- Remove cli/ directory
- Restore deprecation warnings in original scripts
- All backend functionality remains working

### Rollback Triggers

Trigger rollback if:
- **Critical Bug**: Data corruption, silent failures, incorrect embeddings
- **Performance Regression**: >20% slower than baseline
- **Resource Regression**: >50% more memory usage
- **Reliability Regression**: Failure rate >5% for previously working documents

### Rollback Verification

After rollback:
- Run full test suite and verify all tests pass
- Process sample corpus and validate outputs match pre-refactor baselines
- Review manifest for failure patterns
- Update CHANGELOG with rollback details

## Success Criteria

### Functional Criteria
- ✅ All 338 scenarios in spec.md pass validation
- ✅ Zero regressions in existing functionality
- ✅ All 184 implementation tasks completed
- ✅ OpenSpec validation passes with --strict flag

### Quality Criteria
- ✅ Test coverage: ≥90% for new modules
- ✅ Documentation: All public APIs documented with examples
- ✅ Code review: Approved by at least one reviewer
- ✅ Linting: Passes flake8, mypy, black formatting

### Performance Criteria
- ✅ Streaming embeddings use <80% GPU memory
- ✅ Zero CUDA re-initialization crashes in CI
- ✅ Processing time within 10% of baseline
- ✅ Chunking: ≥10 docs/min, Embeddings: ≥5 docs/min

### Operational Criteria
- ✅ 100% of processing tracked in manifest
- ✅ Manifest queries support monitoring workflows
- ✅ Troubleshooting guide covers common error scenarios
- ✅ CLI help text includes examples for all modes

### Adoption Criteria
- ✅ Migration guide published and reviewed
- ✅ Deprecation warnings in place for 1 release cycle
- ✅ Zero user-reported issues with unified CLI
- ✅ README updated with new architecture overview

## Timeline Estimates

### Phase 1: Infrastructure (5 days)
- Day 1-2: _common.py, schemas.py implementation
- Day 3: serializers.py, unit tests
- Day 4-5: Integration testing, bug fixes

### Phase 2: Script Refactoring (5 days)
- Day 1: Path handling updates, HTML script
- Day 2: PDF script, CUDA safety
- Day 3: Chunking script, topic-aware coalescence
- Day 4-5: Testing, legacy script wrappers

### Phase 3: Streaming Embeddings (7 days)
- Day 1-2: Two-pass architecture implementation
- Day 3: Validation and invariant checks
- Day 4: Batch size configuration
- Day 5-6: Testing with large corpus
- Day 7: Performance tuning, documentation

### Phase 4: Unified CLI & Cleanup (3 days)
- Day 1: CLI implementation and testing
- Day 2: Documentation updates, examples
- Day 3: Final integration test, CHANGELOG

**Total: 20 working days (4 weeks)**

## Post-Deployment Plan

### Week 1 After Deploy
- Monitor manifest for failure patterns
- Review performance metrics daily
- Address any user-reported issues immediately
- Collect feedback on unified CLI

### Week 2-4 After Deploy
- Optimize batch sizes based on real workloads
- Update troubleshooting guide with new scenarios
- Run full corpus through pipeline, validate results
- Prepare legacy script removal PR

### Month 2 After Deploy
- Remove *_legacy.py scripts
- Archive change proposal to openspec/changes/archive/
- Update specs/ with final capability documentation
- Document lessons learned

## Communication Plan

### Stakeholder Updates

**Before Implementation**:
- Share proposal with team for review
- Present architecture decisions in team meeting
- Get approval for breaking changes (if any)

**During Implementation**:
- Daily standup updates on progress
- Flag blockers immediately
- Demo new features as completed

**After Deployment**:
- Send migration guide to users
- Announce unified CLI in release notes
- Offer training session for new workflows

### Documentation Updates

**User-Facing**:
- README.md: Architecture overview, quick start
- CLI help text: Examples and troubleshooting
- Migration guide: Old→New command mapping

**Developer-Facing**:
- CONTRIBUTING.md: How to add new processing stages
- Architecture diagrams: Stage boundaries, data flow
- API documentation: Function signatures, examples

## Risk Register

| Risk | Probability | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Pydantic version conflict | Low | Medium | Pin version, test in CI | Dev |
| Performance regression | Medium | High | Benchmark before/after | Dev |
| Silent data corruption | Low | Critical | Golden fixtures, hash checks | QA |
| vLLM API changes | Low | Medium | Version pin, compatibility tests | Dev |
| User adoption resistance | Medium | Low | Clear migration guide, training | PM |
| Incomplete testing | Medium | High | 90% coverage target, CI enforcement | QA |
| Documentation drift | High | Low | Automated doc generation, reviews | Dev |
| Manifest schema changes | Low | Medium | Version field, backward compat | Dev |

## Approvals Required

- [ ] Technical Lead: Architecture design approval
- [ ] Data Team: Schema changes and backward compatibility
- [ ] Operations: Monitoring and alerting integration
- [ ] Product: Unified CLI UX and migration timeline

## Questions for Stakeholders

1. **Timeline**: Is 4-week timeline acceptable or should it be compressed/extended?
2. **Testing**: Should we run production traffic against refactored pipeline in parallel before cutover?
3. **Migration**: Preference for hard cutover vs gradual rollout?
4. **Monitoring**: Existing monitoring infrastructure or should we use manifest-only approach?
5. **Documentation**: Preferred format for API docs (Sphinx, MkDocs, README-only)?
