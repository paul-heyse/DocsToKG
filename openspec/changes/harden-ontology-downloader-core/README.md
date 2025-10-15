# Harden Ontology Downloader Core - Change Summary

This change proposal comprehensively addresses technical debt, correctness bugs, and scalability limitations in the ontology downloader while maintaining backward compatibility for public APIs.

## Quick Reference

- **Change ID:** `harden-ontology-downloader-core`
- **Status:** Pending approval (0/26 tasks completed)
- **Type:** Refactoring + Bug Fixes + Performance Improvements
- **Breaking Changes:** Yes (legacy import paths only)
- **Validation:** ✅ Passed strict validation

## What's Included

### 1. `proposal.md`

High-level overview explaining:

- **Why:** Technical debt, correctness bugs, scalability gaps
- **What Changes:** Code consolidation, bug fixes, streaming normalization, concurrency
- **Impact:** Affected files, breaking changes, migration path

### 2. `tasks.md`

Detailed 26-step implementation checklist organized into 6 phases:

1. **Legacy Cleanup & Consolidation** (7 tasks)
   - Remove import aliases, unify utilities, eliminate duplication
2. **Correctness & Robustness** (3 tasks)
   - Fix URL validation bug, add version locking, concurrent validators
3. **Performance & Scale** (2 tasks)
   - True streaming normalization, threshold configuration
4. **Extensibility** (3 tasks)
   - Plugin infrastructure, manifest migration
5. **Testing & Documentation** (5 tasks)
   - Security tests, concurrency tests, streaming tests, CLI tests, docs
6. **Final Verification** (6 tasks)
   - Full test suite, linting, smoke tests, release prep

### 3. `design.md`

Comprehensive technical design covering:

- **Context & Constraints:** Boundary preservation, determinism requirements
- **Goals/Non-Goals:** Clear scope boundaries
- **6 Key Decisions:**
  1. Legacy import removal (clean break)
  2. Concurrent validators (ThreadPoolExecutor with bounds)
  3. True streaming normalization (external sort + incremental hash)
  4. Inter-process version locking (fcntl/msvcrt)
  5. Plugin infrastructure (entry points with fail-soft)
  6. Manifest schema migration (forward-compatible reads)
- **Risks & Trade-offs:** SHA-256 stability, concurrent safety, plugin security
- **Migration Plan:** 4-phase rollout with rollback strategy

### 4. `specs/ontology-download/spec.md`

18 formal requirements with concrete scenarios:

- **Code Consolidation** (7 requirements)
- **Correctness** (2 requirements)
- **Streaming & Performance** (2 requirements)
- **Extensibility** (3 requirements)
- **Testing** (3 requirements)
- **HTTP Robustness** (1 requirement)

Each requirement includes 1-4 scenarios written in WHEN/THEN/AND format.

## Key Technical Highlights

### Critical Bug Fix

- **Issue:** `_populate_plan_metadata` passes list instead of `DownloadConfiguration` to `validate_url_security`
- **Fix:** Pass `config.defaults.http` (full object) for proper allowlist/IDN checks
- **Impact:** Prevents crashes, enables security enforcement

### True Streaming Normalization

- **Current:** Claims "streaming" but loads all triples into memory via `list(graph)`
- **New:** External sort (platform `sort` or Python merge-sort) + incremental SHA-256
- **Benefit:** Handle ontologies larger than RAM with deterministic output

### Concurrent Validators

- **Current:** Sequential execution increases latency
- **New:** `ThreadPoolExecutor` with bounded concurrency (default 2, max 8)
- **Benefit:** 2-3x latency reduction for multi-validator runs

### Inter-Process Locking

- **Current:** Concurrent writes can corrupt version directories
- **New:** File locks (`fcntl.flock`/`msvcrt.locking`) per version
- **Benefit:** Safe parallel downloads, no corruption

### Plugin Infrastructure

- **Current:** Extending requires forking
- **New:** `importlib.metadata.entry_points` for resolvers/validators
- **Benefit:** Ecosystem extensibility with fail-soft loading

## Validation Results

```bash
$ openspec validate harden-ontology-downloader-core --strict
Change 'harden-ontology-downloader-core' is valid
```

All requirements have proper scenario formatting with `#### Scenario:` headers and WHEN/THEN/AND structures.

## Next Steps

1. **Review & Approval:** Stakeholders review proposal, design, and requirements
2. **Implementation:** Follow `tasks.md` checklist sequentially (6 phases)
3. **Testing:** Comprehensive unit, integration, and property-based tests
4. **Documentation:** Update CHANGELOG, MIGRATION.md, API docs
5. **Release:** Minor version bump (breaking change for imports only)

## Breaking Changes

**Only affects import paths:**

- ❌ `from DocsToKG.OntologyDownload.core import ...`
- ❌ `from DocsToKG.OntologyDownload.config import ...`
- ✅ `from DocsToKG.OntologyDownload import fetch_all, plan_all, ...`

**Public API remains stable:**

- `fetch_all()`, `plan_all()`, `fetch_one()`, `plan_one()`
- `FetchSpec`, `FetchResult`, `PlannedFetch`, `ResolvedConfig`
- Manifest schema version "1.0" (no changes)

## Estimated Scope

- **Lines changed:** ~1,500
- **Lines added (tests):** ~500
- **Lines removed (duplication):** ~200
- **Net change:** +1,800 (consolidation + tests)
- **Implementation time:** 2-3 weeks (26 tasks × 2-4 hours avg)

## References

- **AGENTS.md:** OpenSpec workflow guide
- **proposal.md:** High-level change description
- **design.md:** Technical design decisions
- **tasks.md:** Step-by-step implementation checklist
- **spec.md:** Formal requirements with scenarios

## Contact

For questions or clarifications about this change proposal, consult:

1. The detailed `design.md` for architectural rationale
2. The `tasks.md` checklist for implementation sequencing
3. The `spec.md` requirements for acceptance criteria
