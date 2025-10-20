# Phase 4 Pre-Implementation Audit: Planning Infrastructure

**Date**: October 20, 2025
**Purpose**: Identify legacy code and incomplete work in planning infrastructure before implementing Phase 4

---

## Executive Summary

Before implementing Phase 4 (Plan Caching & Comparison), we should audit and potentially clean up the planning infrastructure to avoid building on legacy code. This document identifies findings and recommendations.

---

## Current State Analysis

### ✅ What Already Exists

**1. Plan Diff Feature (COMPLETE & ACTIVE)**

- **Location**: `src/DocsToKG/OntologyDownload/cli.py` lines 239-301 (parser)
- **Location**: `src/DocsToKG/OntologyDownload/cli.py` lines 905-971 (_handle_plan_diff)
- **Location**: `src/DocsToKG/OntologyDownload/manifests.py` lines 441-488 (compute_plan_diff, format_plan_diff)
- **Status**: Production-ready
- **Functionality**:
  - Compares baseline vs current plans
  - Supports file-based baseline or manifest-based baseline
  - Can update baseline snapshots
  - Generates diff report (added/removed/modified)
  - Outputs JSON or formatted text

**2. Plan Conversion Utility (ACTIVE)**

- **Location**: `src/DocsToKG/OntologyDownload/manifests.py` lines 81-140 (plan_to_dict)
- **Status**: Production-ready
- **Functionality**: Converts PlannedFetch objects to dictionaries for comparison

**3. Plan All Function (ACTIVE)**

- **Location**: `src/DocsToKG/OntologyDownload/planning.py` lines 2322-2739 (plan_all)
- **Status**: Production-ready
- **Functionality**: Main planning orchestration with concurrency support

---

## Legacy Code Assessment

### 🔍 Identified Legacy Patterns

**1. File-Based Plan Baseline Storage**

**Status**: Working but not optimal

- **File**: `DEFAULT_PLAN_BASELINE` in cli.py
- **Location**: Uses JSON files for baseline storage
- **Problem**: Not integrated with database, difficult to version control, no transaction semantics
- **Recommendation**: KEEP (for now) - file-based baseline is useful for CLI users; database storage should be optional enhancement in Phase 4

**2. Manifest-Based Comparison**

**Status**: Working but could be optimized

- **Location**: `src/DocsToKG/OntologyDownload/cli.py` lines 937-960
- **Problem**: Reads individual manifest files for each ontology (could be O(n) I/O operations)
- **Recommendation**: In Phase 4, could batch these reads via database

**3. Plan Index Lock System**

**Status**: Appears to be historical/unused

- **Location**: `src/DocsToKG/OntologyDownload/planning.py` lines 1518, 1558
- **Pattern**: References to "ontology index lock" in debug logs
- **Analysis**: Only appears in logging, no actual lock implementation visible
- **Recommendation**: Review if this is dead code or if locking exists elsewhere

**4. Planner Probe System**

**Status**: Active but potentially over-engineered for current use

- **Location**: `src/DocsToKG/OntologyDownload/planning.py` lines 906-1110 (planner_http_probe)
- **Status**: Production-ready
- **Note**: HEAD probes for resolver validation; works as designed
- **Recommendation**: KEEP - useful for planning accuracy

---

## Code Quality Observations

### ✅ Well-Designed Components

1. **PlannedFetch dataclass**: Clear structure, comprehensive metadata
2. **compute_plan_diff algorithm**: Clean set-based comparison
3. **plan_all orchestration**: Good concurrency patterns with futures
4. **Error handling**: Robust exception handling throughout

### ⚠️ Potential Technical Debt

1. **Manifest reading in plan_diff**
   - **Issue**: Sequential file I/O for each ontology
   - **Impact**: Low for small datasets, could be slow for 1000+ ontologies
   - **Mitigation**: Phase 4 database integration would solve this

2. **No caching of plan results**
   - **Issue**: Re-running `plan_all` with same specs re-probes resolvers
   - **Impact**: Slower re-runs, higher network load
   - **Mitigation**: Phase 4 would cache plans in database

3. **Lock reference inconsistency**
   - **Issue**: "ontology index lock" referenced but no visible implementation
   - **Impact**: Debug logs may be misleading
   - **Recommendation**: Investigate and clarify

---

## Recommendations

### ✅ Phase 4 Strategy

**Recommended Approach**: Build Phase 4 ON TOP of existing functionality, not replacing it

1. **Keep existing plan-diff working**: File-based baselines are useful
2. **Add optional database caching**: New tables for plan history
3. **Enable database-based comparison**: New query methods for faster diffs
4. **Maintain backward compatibility**: Don't break existing CLI

### 🔧 Before Phase 4: Cleanup Items

| Item | Priority | Action | Effort |
|------|----------|--------|--------|
| Clarify "ontology index lock" references | LOW | Investigation + documentation | 30 min |
| Add comments about manifest I/O pattern | LOW | Code annotation | 15 min |
| Document planner probe design | LOW | Architecture doc | 30 min |

### ⏱️ Phase 4 Implementation Order

1. **Add plan caching tables to database schema** (0004_events → 0005_plans)
2. **Implement plan storage facade** (upsert_plan, get_plan, list_plans)
3. **Integrate caching into plan_all** (optional write-through cache)
4. **Add plan-diff database queries** (compare stored vs current)
5. **Enhance CLI** (--use-cache flag for plan-diff)
6. **Test and document**

---

## Conclusion

**The planning infrastructure is well-designed and production-ready.** There's minimal legacy code to clean up. The focus for Phase 4 should be:

1. ✅ Build database caching ON TOP of existing system
2. ✅ Keep file-based baselines working
3. ✅ Add optional database acceleration
4. 🔧 Do lightweight cleanup (documentation, clarification)

**Recommendation**: PROCEED with Phase 4 implementation using the proposed strategy. No blocking issues identified.

---

## Files Involved

- `src/DocsToKG/OntologyDownload/planning.py` (2739 LOC - well-organized, minimal legacy)
- `src/DocsToKG/OntologyDownload/cli.py` (2493 LOC - plan-diff integrated cleanly)
- `src/DocsToKG/OntologyDownload/manifests.py` (489 LOC - plan comparison utilities)

---

## Next Steps

1. Address low-priority cleanup items (documentation)
2. Design Phase 4 schema additions
3. Implement plan caching layer
4. Update documentation with caching options
