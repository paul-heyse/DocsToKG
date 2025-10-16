# Change Proposal Summary: Refactor ContentDownload Robustness

## Overview
This comprehensive change proposal consolidates the ContentDownload module by eliminating duplicate code, removing legacy compatibility layers, standardizing HTTP behavior, and improving resource management across 4 Python modules totaling ~3,800 lines of code.

## Structure

### 📋 proposal.md
- **Why**: Details technical debt including duplicate context blocks, resource leaks, and 30% dead code
- **What**: 6 priority-grouped change categories spanning correctness, legacy removal, standardization, abstraction, robustness, and observability
- **Impact**: Identifies 4 breaking changes with migration paths and preserves backward compatibility for manifests and configuration

### ✅ tasks.md (186 tasks across 8 major sections)
1. **High-Impact Correctness Fixes** (14 tasks) - CLI context, CSV leak, HEAD precheck unification
2. **Legacy Code Removal** (24 tasks) - __getattr__ shims, proxy functions, session-less branches
3. **HTTP Behavior Standardization** (18 tasks) - Unified retries, timeouts, headers
4. **Code Reduction Through Abstraction** (26 tasks) - ApiResolverBase, HTML helpers, sink protocol
5. **Robustness Enhancements** (18 tasks) - Pre-validation, jittered throttling, hardened detection
6. **Observability Improvements** (24 tasks) - Staging mode, manifest index, last-attempt CSV
7. **Testing and Validation** (26 tasks) - Unit, integration, and regression tests
8. **Documentation and Migration** (10 tasks) - CHANGELOG, migration guide, code cleanup

### 🏗️ design.md
- **Context**: Stakeholders, constraints, background on 4-module architecture
- **Goals**: Primary (eliminate duplication, remove legacy, standardize) and secondary (observability, robustness)
- **Non-Goals**: No new resolvers, no schema changes, no parsing coupling, no distributed execution
- **6 Key Decisions**:
  1. Unified HEAD precheck in network layer
  2. ApiResolverBase for JSON API pattern (100-200 line reduction)
  3. AttemptSink protocol with MultiSink composition
  4. Remove legacy exports and session-less branches
  5. Conditional request pre-validation
  6. Staging directory mode (opt-in)
- **Risks & Mitigations**: Breaking changes, HTTP behavior changes, API base class fit, staging path logic
- **6-Phase Migration Plan**: Correctness → Legacy Removal → HTTP → Code Reduction → Observability → Robustness
- **Timeline**: 12-20 days estimated

### 📜 specs/content-download/spec.md (21 ADDED requirements)
Comprehensive specification defining:
- CLI context management and resource lifecycle
- Idempotent cleanup for all closeable resources
- Unified HEAD preflight with 405/501 degradation
- Network layer import unification
- Standardized resolver HTTP behavior (retries, timeouts, headers)
- Session-less branch elimination
- ApiResolverBase class with _request_json() helper
- HTML scraping helper extraction
- Protocol-based logging architecture (AttemptSink, MultiSink)
- Conditional request pre-validation
- Jittered domain throttling
- Hardened PDF detection (octet-stream handling)
- Enhanced corruption detection (HEAD-validated tiny PDFs)
- Staging directory isolation
- Manifest index generation
- Last-attempt CSV generation
- Organized CLI help grouping
- Resolver toggle default centralization
- Module independence preservation
- Backward manifest compatibility
- Configuration file compatibility

Each requirement includes 1-5 detailed scenarios using WHEN/THEN/AND format.

## Key Metrics
- **Code Reduction**: Target 400-600 lines removed
- **Requirements**: 21 comprehensive requirements with 60+ scenarios
- **Tasks**: 186 implementation tasks
- **Testing**: 26 dedicated testing tasks
- **Files Impacted**: 3 primary (CLI, network, resolvers), 1 untouched (utils)
- **Breaking Changes**: 4 documented with migration paths
- **Estimated Effort**: 12-20 days

## Validation Status
✅ Passed `openspec validate refactor-content-download-robustness --strict`

## Next Steps
1. Review proposal with stakeholders
2. Approve design decisions
3. Begin Phase 1 (Correctness Fixes) implementation
4. Proceed through 6-phase migration plan
5. Archive change after deployment

## Related Changes
- consolidate-hybridsearch-modules (0/87 tasks) - orthogonal, different module
- harden-ontology-downloader-core (0/26 tasks) - orthogonal, different module
- refactor-docparsing-robustness (Complete) - same pattern, parsing domain

## Usage
```bash
# View proposal
openspec show refactor-content-download-robustness

# View detailed JSON
openspec show refactor-content-download-robustness --json --deltas-only

# Validate
openspec validate refactor-content-download-robustness --strict

# After implementation and deployment
openspec archive refactor-content-download-robustness
```
