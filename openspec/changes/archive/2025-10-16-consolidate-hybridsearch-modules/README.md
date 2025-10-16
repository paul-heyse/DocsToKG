# Consolidate HybridSearch Module Structure

## Overview

This change proposal consolidates the HybridSearch module architecture from 12 modules plus a Tools directory down to 9 core modules, eliminating adapter modules, clarifying GPU resource ownership, and reducing cross-module coupling while maintaining complete backward compatibility through deprecation shims.

## Problem Statement

The current HybridSearch implementation suffers from:

- **Excessive module fragmentation**: 12 modules with thin adapters that exist solely for indirection
- **GPU resource management opacity**: Device and resource handles passed through multiple boundaries
- **Cognitive overhead**: Simple operations require navigating many module boundaries
- **CLI duplication**: Tools directory replicates validation functionality

## Solution Summary

**Consolidation targets:**

1. Merge `results.py` → `ranking.py` (result shaping with ranking)
2. Integrate `similarity.py` → `vectorstore.py` (GPU ops near FAISS)
3. Distribute `operations.py` → `service.py` + `vectorstore.py` (by responsibility)
4. Merge `schema.py` → `storage.py` (if separate, unify lexical storage)
5. Eliminate `retrieval.py` → `service.py` (remove thin shim)
6. Retire `Tools/` directory → `validation.py` (single CLI entry point)

**Result:** 9 focused modules with clear ownership boundaries and unidirectional dependencies.

## Document Structure

### [proposal.md](./proposal.md)

- **Why**: Explains architectural issues with current structure
- **What Changes**: Lists specific consolidation actions
- **Impact**: Details affected code, specs, and migration paths

### [tasks.md](./tasks.md)

Comprehensive 12-phase implementation checklist covering:

1. Preparation and analysis
2. ResultShaper migration to ranking.py
3. Similarity function integration into vectorstore.py
4. Operations distribution by responsibility
5. Schema merger with storage.py
6. Retrieval elimination into service.py
7. Tools directory retirement
8. Public interface updates and backward compatibility
9. Test suite import path updates
10. Documentation consistency updates
11. Validation and quality assurance
12. Deployment and communication

**Total:** ~80+ granular subtasks with specific file paths and verification steps.

### [design.md](./design.md)

Detailed technical design including:

- **Context**: Current architecture analysis and challenges
- **Goals**: Primary objectives and explicit non-goals
- **Decisions**: 7 key architectural decisions with rationales and alternatives
- **Architecture**: Post-consolidation structure with dependency graph
- **Risks**: 4 major risks with detailed mitigation strategies
- **Trade-offs**: Explicit engineering choices and their justifications
- **Migration Plan**: 3-phase rollout (implementation, deprecation, removal)

### [specs/hybrid-search/spec.md](./specs/hybrid-search/spec.md)

Requirements specification establishing behavioral contracts:

- 10 major requirements covering module organization, GPU lifecycle, backward compatibility, functional cohesion, unidirectional dependencies, CLI consolidation, observability, configuration isolation, test updates, documentation, and behavioral equivalence
- 31 scenarios with explicit WHEN/THEN assertions for testable validation

## Key Design Principles

### 1. Functional Cohesion

Related operations grouped in single modules:

- Ranking + result shaping (both operate on fused scores with GPU deduplication)
- FAISS + GPU similarity (all GPU ops centralized)
- Service operations (pagination, stats) separate from state management

### 2. GPU Resource Ownership

Single source of truth model:

- `FaissIndexManager` creates and owns `StandardGpuResources`
- Explicit parameter passing: `device`, `resources` threaded to consumers
- No module-level caching or hidden GPU state

### 3. Backward Compatibility

Deprecation shim pattern:

```python
# results.py (shim)
from .ranking import ResultShaper  # noqa: F401
import warnings
warnings.warn(
    "DocsToKG.HybridSearch.results is deprecated; "
    "import from DocsToKG.HybridSearch.ranking instead",
    DeprecationWarning
)
```

### 4. Unidirectional Dependencies

Dependency DAG prevents circular imports:

```
types ← (all modules)
config ← service, vectorstore, ranking, ingest
vectorstore ← ingest, service, ranking, validation
storage ← ingest, service, ranking
ranking ← service
features ← service
service ← (validation only)
```

## Migration Timeline

### Phase 1: Implementation (Current Change)

- Execute consolidation tasks
- Add deprecation shims
- Update tests and CI
- Validate via comprehensive test suite

### Phase 2: Deprecation Period (Next Release, e.g., v0.5.0)

- Ship with deprecation warnings active
- Communicate migration paths to users
- Provide 1 full release cycle for adaptation

### Phase 3: Removal (Subsequent Release, e.g., v0.6.0)

- Delete shim files
- Remove sys.modules aliasing
- Update docs to remove deprecated references

## Validation Checklist

✓ OpenSpec validation passes (`openspec validate consolidate-hybridsearch-modules --strict`)
✓ All requirements have at least one scenario
✓ All scenarios use proper `#### Scenario:` format with WHEN/THEN structure
✓ Design document includes context, goals, decisions, risks, architecture
✓ Tasks broken down into atomic, verifiable steps
✓ Proposal clearly states why, what, and impact

## Review Guidance

**For architectural review:**

- Focus on `design.md` decisions section
- Review dependency graph and invariants
- Assess risk mitigation strategies

**For implementation planning:**

- Use `tasks.md` as execution roadmap
- Each task includes specific files and verification steps
- Phases can be executed incrementally

**For validation:**

- Reference `specs/hybrid-search/spec.md` requirements
- Each scenario provides testable assertions
- Behavioral equivalence requirement ensures no semantic changes

## Next Steps

1. **Review & Approval**: Obtain stakeholder approval on architectural decisions
2. **Implementation**: Execute tasks sequentially, validating each phase
3. **Testing**: Run full test suite + GPU validation harness after each consolidation
4. **Documentation**: Update user-facing docs with new import paths
5. **Communication**: Announce deprecation timeline and provide migration guide
6. **Monitoring**: Track adoption during deprecation period, assist users with migration

---

**Change ID:** `consolidate-hybridsearch-modules`
**Type:** Architectural refactoring (non-breaking during deprecation period)
**Complexity:** High (touches all modules, requires careful testing)
**Impact:** Reduces module count 25% (12→9), improves maintainability, clarifies GPU ownership
**Risk Level:** Medium (mitigated via comprehensive testing and deprecation cycle)
