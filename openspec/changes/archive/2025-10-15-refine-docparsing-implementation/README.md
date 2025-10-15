# Refine DocParsing Implementation

**Change ID**: `refine-docparsing-implementation`
**Status**: Proposed
**Created**: 2025-10-15

## Quick Summary

This change proposal addresses 18 production-readiness gaps in the DocParsing pipeline through targeted refinements that improve correctness, portability, memory efficiency, and maintainability **without breaking existing workflows**.

## Problem Statement

The DocParsing pipeline works but contains technical debt that creates operational pain:

- Non-atomic writes risk partial files on crashes
- Hardcoded paths prevent multi-environment deployment
- Memory inefficiency causes OOM on large corpora
- Incorrect UTC timestamps break log correlation
- Legacy script duplication confuses users

## Solution Overview

Systematic refinements across 8 categories:

1. **Crash Safety**: Atomic writes, UTC timestamps, hash algorithm tagging
2. **Memory Efficiency**: Drop corpus-wide text retention in embeddings
3. **Portability**: Environment-driven model paths, offline mode
4. **Scalability**: Manifest sharding by stage
5. **Observability**: vLLM preflight telemetry, SPLADE threshold docs
6. **Code Quality**: CLI simplification, test cleanup, legacy quarantine
7. **Data Integrity**: Schema enforcement, top-level image flags
8. **User Experience**: Clear deprecation warnings, better help text

## Key Benefits

- **Zero Breaking Changes**: All refinements are backward-compatible
- **Immediate Value**: Each change solves a specific operational pain point
- **Phased Implementation**: 8 reviewable PRs, each < 200 LOC
- **Proven Patterns**: Atomic writes, env-driven config, manifest sharding are industry-standard

## Files

- **[proposal.md](./proposal.md)**: Full problem statement, change details, and impact analysis
- **[tasks.md](./tasks.md)**: Detailed implementation guide with 8 phases and ~30 tasks
- **[design.md](./design.md)**: Architectural decisions and trade-offs
- **[implementation-patterns.md](./implementation-patterns.md)**: Reusable code patterns, scripts, and testing recipes
- **[specs/doc-parsing/spec.md](./specs/doc-parsing/spec.md)**: Specification deltas (MODIFIED/ADDED requirements)

## How to Use This Proposal

### For AI Programming Agents

1. **Start with tasks.md**: Read the prerequisites and task dependencies matrix
2. **Follow phase sequence**: Phases 1-2 have detailed step-by-step instructions
3. **Reference patterns**: Phases 3-8 reference reusable patterns in implementation-patterns.md
4. **Apply patterns**: Each pattern in implementation-patterns.md includes:
   - Complete code examples
   - Automated application scripts
   - Validation commands
   - Test templates

### For Human Reviewers

1. **proposal.md**: Understand the "why" and high-level "what"
2. **design.md**: Review architectural decisions and trade-offs
3. **specs/doc-parsing/spec.md**: Validate specification changes
4. **tasks.md**: Review implementation approach
5. **implementation-patterns.md**: Examine code quality and best practices

### Implementation Workflow

```bash
# Phase 1-2: Follow detailed instructions
cd /home/paul/DocsToKG
direnv allow  # Activate environment
git checkout -b refine-docparsing-phase-1

# Follow tasks.md Phase 1 line by line
# Each task has validation commands

# Phase 3-8: Apply patterns
# Example: Phase 3 (Atomic Writes)
# 1. Read pattern in implementation-patterns.md → "Atomic Write Pattern"
# 2. Apply pattern using provided script
# 3. Run validation from pattern
# 4. Commit phase changes

git add -A
git commit -m "Phase 1: Legacy script quarantine"
```

## Implementation Plan

8 phases, each merge-ready independently:

| Phase | Focus | Files | LOC | Pattern Reference |
|-------|-------|-------|-----|-------------------|
| 1 | Legacy quarantine + test cleanup | 6 | 200 | File Move, Shim, Test Suite |
| 2 | Atomic writes + UTC + hash tagging | 4 | 150 | Atomic Write, UTC Logger |
| 3 | CLI simplification | 4 | -60 | CLI Simplification |
| 4 | Embeddings memory refactor | 1 | 100 | Streaming Architecture |
| 5 | Portable paths + offline mode | 3 | 120 | Env Config, Offline Mode |
| 6 | Manifest sharding + vLLM preflight | 3 | 100 | Manifest Sharding, Service Preflight |
| 7 | Image flags + SPLADE docs | 3 | 60 | Schema Field Addition, CLI Help |
| 8 | Final polish | 5 | 80 | Multiple patterns |

**Total**: ~660 LOC changed, ~60 LOC deleted net

## Testing Strategy

Each change includes:

- **Unit tests**: Verify isolated behavior (e.g., atomic write on SIGKILL)
- **Integration tests**: End-to-end pipeline on small corpus
- **Smoke tests**: CLI invocations with `--help`, small runs
- **Backward compatibility**: Existing JSONL files and manifests still work

Deferred to follow-up (documented in tasks.md):

- GPU benchmarking (requires live vLLM)
- 100K-document performance regression testing

## Affected Components

**Modified**:

- `DoclingHybridChunkerPipelineWithMin.py` (atomic writes, test cleanup)
- `EmbeddingV2.py` (memory refactor, portable paths)
- `run_docling_html_to_doctags_parallel.py` (shim)
- `run_docling_parallel_with_vllm_debug.py` (shim, preflight)
- `_common.py` (UTC fix, hash support, sharding)
- `schemas.py` (top-level image fields)

**Added**:

- `legacy/` subdirectory (quarantine for deprecated scripts)

**Dependencies**: None (uses existing `pydantic`, `docling`, `vllm`, `transformers`)

## Migration Path

**No migration required**. All changes are:

- Backward-compatible additions (new env vars, CLI flags)
- Internal refactorings (memory efficiency, UTC fix)
- Optional enhancements (sharding auto-enabled, old manifests still work)

Users can continue using existing CLIs, manifests, and JSONL files without modification.

## Related Changes

This proposal complements but is **separate from** the existing `refactor-docparsing-pipeline` change, which focused on architectural modularization. This change focuses on production hardening of the **existing** architecture.

## Next Steps

1. Review proposal.md for completeness
2. Validate tasks.md against actual codebase
3. Implement Phase 1 (legacy quarantine + test cleanup)
4. Iterate through phases 2-8 based on feedback

## Questions or Feedback?

See [proposal.md](./proposal.md) for detailed rationale, [design.md](./design.md) for architectural decisions, or [tasks.md](./tasks.md) for implementation specifics.
