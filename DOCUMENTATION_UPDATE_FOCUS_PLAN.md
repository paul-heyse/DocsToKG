# Documentation Update Focus Plan - Targeted Approach

**Date:** October 21, 2025  
**Scope:** Focused docstring, NAVMAP, and top-level description updates  
**Strategy:** Lightweight maintenance-focused updates, not comprehensive overhaul

---

## Overview

Given the active evolution of the codebase, we'll take a **pragmatic, targeted approach** focusing on:

1. **Module Docstrings** - Ensure accuracy of module-level documentation
2. **NAVMAPs** - Keep navigation maps current and accurate
3. **Top-Level Descriptions** - Clear, concise module purpose statements
4. **AGENTS.md Files** - Keep agent guides current and relevant
5. **README.md Files** - Ensure main documentation is accurate

This approach ensures documentation stays current **without blocking development** and enables future comprehensive documentation when the codebase stabilizes.

---

## Phase 1: ContentDownload Updates (1-2 hours)

### Key Modules to Update

**Core Modules:**
- [ ] `src/DocsToKG/ContentDownload/__init__.py` - Package root docstring
- [ ] `src/DocsToKG/ContentDownload/args.py` - CLI args documentation
- [ ] `src/DocsToKG/ContentDownload/runner.py` - DownloadRun class documentation
- [ ] `src/DocsToKG/ContentDownload/pipeline.py` - Pipeline orchestration
- [ ] `src/DocsToKG/ContentDownload/core.py` - Core types documentation

**Networking Modules:**
- [ ] `src/DocsToKG/ContentDownload/httpx_transport.py` - HTTP transport layer
- [ ] `src/DocsToKG/ContentDownload/networking.py` - Network retry logic
- [ ] `src/DocsToKG/ContentDownload/ratelimit/manager.py` - Rate limiting

**Advanced Features:**
- [ ] `src/DocsToKG/ContentDownload/breakers.py` - Circuit breaker system
- [ ] `src/DocsToKG/ContentDownload/fallback/orchestrator.py` - Fallback strategy
- [ ] `src/DocsToKG/ContentDownload/telemetry.py` - Telemetry schemas

**Documentation:**
- [ ] `src/DocsToKG/ContentDownload/AGENTS.md` - Agent guide (if exists)
- [ ] `src/DocsToKG/ContentDownload/README.md` - Main documentation

### Update Strategy

For each module:
1. ✅ Read current module docstring
2. ✅ Verify accuracy against current implementation
3. ✅ Check NAVMAP relevance and accuracy
4. ✅ Update if outdated or inaccurate
5. ✅ Ensure top-level description is clear

---

## Phase 2: OntologyDownload Updates (1-2 hours)

### Key Modules to Update

**Core Modules:**
- [ ] `src/DocsToKG/OntologyDownload/__init__.py` - Package root docstring
- [ ] `src/DocsToKG/OntologyDownload/api.py` - Public API documentation
- [ ] `src/DocsToKG/OntologyDownload/planning.py` - Planning orchestration
- [ ] `src/DocsToKG/OntologyDownload/cli.py` - CLI interface

**Catalog & Storage:**
- [ ] `src/DocsToKG/OntologyDownload/catalog/__init__.py` - Catalog interface
- [ ] `src/DocsToKG/OntologyDownload/catalog/repo.py` - Repository queries
- [ ] `src/DocsToKG/OntologyDownload/io/filesystem.py` - Filesystem ops

**Advanced Features:**
- [ ] `src/DocsToKG/OntologyDownload/policy/gates.py` - Security gates
- [ ] `src/DocsToKG/OntologyDownload/validation.py` - Validator orchestration
- [ ] `src/DocsToKG/OntologyDownload/analytics/pipelines.py` - Analytics

**Documentation:**
- [ ] `src/DocsToKG/OntologyDownload/AGENTS.md` - Agent guide (if exists)
- [ ] `src/DocsToKG/OntologyDownload/README.md` - Main documentation

### Update Strategy

Same as Phase 1 - focused, lightweight updates

---

## Phase 3: Root-Level Documentation (30 minutes)

### Files to Update

- [ ] `src/DocsToKG/ContentDownload/AGENTS.md` - Update examples and references
- [ ] `src/DocsToKG/OntologyDownload/AGENTS.md` - Update examples and references
- [ ] `README.md` (root) - Update if references these modules
- [ ] Any top-level guides that reference these packages

---

## What NOT to Do

❌ **DON'T:**
- Rewrite entire module structures
- Change architecture documentation extensively
- Add new examples or detailed use cases (keep it focused)
- Update every utility function
- Create new documentation files

✅ **DO:**
- Fix inaccurate docstrings
- Update NAVMAPs to reflect current structure
- Clarify purpose statements
- Update AGENTS.md with current reality
- Ensure cross-references are correct

---

## Priority Order

1. **High Priority** - Public APIs and main entry points
   - `__init__.py` files
   - `api.py` / `args.py`
   - `AGENTS.md` guides

2. **Medium Priority** - Core orchestration
   - `runner.py`, `planning.py`, `pipeline.py`
   - `cli.py` files
   - Main README files

3. **Lower Priority** - Advanced features (only if needed)
   - Circuit breakers, rate limiting
   - Advanced orchestration
   - Utility modules

---

## Quality Checklist

For each module updated:
- [ ] Docstring is accurate to current implementation
- [ ] NAVMAP (if present) reflects current structure
- [ ] Top-level description is clear and concise
- [ ] No references to removed/deprecated features
- [ ] Examples (if any) show current usage
- [ ] Cross-references are correct
- [ ] Passes basic linting (no syntax errors)

---

## Git Workflow

Each phase will be committed with:
- Clear commit message describing what was updated
- Specific modules listed
- No code changes, only documentation

Example:
```
docs: Update ContentDownload module docstrings and NAVMAPs

Updated docstrings, NAVMAPs, and top-level descriptions for:
- args.py - CLI argument parsing
- runner.py - DownloadRun orchestration
- httpx_transport.py - HTTP caching and transport
- AGENTS.md - Agent guide

Verified accuracy against current implementations.
No code changes, documentation only.
```

---

## Estimated Timeline

- Phase 1 (ContentDownload): 1-2 hours
- Phase 2 (OntologyDownload): 1-2 hours
- Phase 3 (Root docs): 30 minutes

**Total: 2.5-4.5 hours** (much more manageable than comprehensive approach)

---

## Post-Update

Once complete:
- ✅ Documentation will be current with latest code
- ✅ New contributors can rely on accurate module docs
- ✅ NAVMAPs provide current navigation
- ✅ AGENTS.md guides are relevant
- ✅ Ready for future comprehensive docs when code stabilizes

---

## Notes

This targeted approach ensures:
- **Minimal disruption** to active development
- **Maximum accuracy** of current documentation
- **Clear foundation** for future comprehensive work
- **Developer experience** stays high even during rapid changes

The comprehensive documentation effort can happen later when the codebase stabilizes.

