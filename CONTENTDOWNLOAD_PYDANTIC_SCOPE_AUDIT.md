# ContentDownload Pydantic v2 Config Scope Audit

**Date:** October 21, 2025
**Status:** ⚠️ **INCOMPLETE - IMPLEMENTATION PLANNED BUT NOT DEPLOYED**
**Scope Source:** `DocParsing - Pydantic implemetation.md` (P2 Objectives)

---

## Executive Summary

The **Pydantic v2 Config refactoring** for ContentDownload was planned in detail but **has NOT been implemented**. The current codebase still uses:
- **Dataclasses** (not Pydantic models) for configuration
- **Ad-hoc resolver loading** (not a registry pattern)
- **Scattered CLI parsing** (not unified Typer app with config introspection)
- **No formal API surface** (helpers use dict/dataclass hybrids)

This is a **significant scope gap** requiring deliberate action to complete.

---

## Planned vs. Actual: Detailed Comparison

### 1. Pydantic v2 Config Models

#### ✅ Planned
- **Location:** `config/models.py`
- **Contents:**
  - RetryPolicy (v2 model)
  - BackoffPolicy (v2 model)
  - RateLimitPolicy (v2 model)
  - RobotsPolicy (v2 model)
  - DownloadPolicy (v2 model)
  - HttpClientConfig (v2 model)
  - TelemetryConfig (v2 model)
  - ResolverCommonConfig (v2 model)
  - ContentDownloadConfig (v2 model, top-level)
- **Features:**
  - `extra="forbid"` (strict validation)
  - `validate_assignment=True` (runtime mutation prevention)
  - Type hints (all fields)
  - Nested model hierarchy
  - Environment variable support

#### ❌ Actual
**Status: NOT IMPLEMENTED**

- `config/` directory exists but contains only YAML files:
  - `ratelimits.yaml`
  - `cache.yaml`
  - `breakers.yaml`
  - `fallback.yaml`

- No `models.py` file exists
- No Pydantic v2 models in ContentDownload
- Configuration still uses **dataclasses**:
  - `DownloadConfig` in `download.py` (dataclass)
  - `ResolvedConfig` in `args.py` (dataclass)
  - `DownloadContext` in `core.py` (dataclass)

### 2. Config Loader (file/env/CLI precedence)

#### ✅ Planned
- **Location:** `config/loader.py`
- **Features:**
  - Load from YAML/JSON files
  - Environment variable overlay (double-underscore → dot notation)
  - CLI override composition
  - Precedence: file < env < CLI
  - Example: `DTKG_HTTP__USER_AGENT="..."` → `http.user_agent` field

#### ❌ Actual
**Status: NOT IMPLEMENTED**

- No `loader.py` file
- Config loading scattered:
  - `args.py` handles CLI argument parsing
  - `cache_loader.py` handles cache YAML manually
  - `ratelimits_loader.py` handles ratelimits YAML manually
  - No unified composition strategy
- No environment variable → nested field mapping
- No dynamic config merging

### 3. Resolver Registry Pattern

#### ✅ Planned
- **Location:** `resolvers/__init__.py`
- **Features:**
  - `@register(name)` decorator
  - `_REGISTRY: Dict[str, Type[Resolver]]`
  - `get_registry()` function
  - `build_resolvers(order, config)` function
  - Dynamic resolver instantiation by name
  - Per-resolver config overrides via `from_config(rcfg, root_cfg)`

- **Each resolver module:**
  - `@register("unpaywall")` decorator
  - Implements `resolve(...) -> ResolverResult`
  - Optional `from_config()` classmethod
  - Tight coupling to config model

#### ❌ Actual
**Status: NOT IMPLEMENTED**

- No registry system
- Resolvers loaded manually in `args.py`:
  - Hard-coded import statements
  - Manual instantiation in functions
  - No decorator pattern
  - No dynamic ordering support

- Resolver modules exist but **not integrated with registry:**
  - `resolvers/unpaywall.py`
  - `resolvers/crossref.py`
  - `resolvers/arxiv.py`
  - `resolvers/europe_pmc.py`
  - `resolvers/pmc.py`
  - `resolvers/core.py`
  - `resolvers/doaj.py`
  - `resolvers/semantic_scholar.py` (replaces s2)
  - `resolvers/landing_page.py`
  - `resolvers/wayback.py`
  - Plus many others not in original scope

### 4. API Surface: Unified Types

#### ✅ Planned
- **Location:** `api/types.py`
- **Types:**
  ```python
  @dataclass(frozen=True)
  class DownloadPlan:
      url: str
      resolver_name: str
      referer: Optional[str]
      expected_mime: Optional[str]

  @dataclass(frozen=True)
  class DownloadStreamResult:
      path_tmp: str
      bytes_written: int
      http_status: int
      content_type: Optional[str]

  @dataclass(frozen=True)
  class DownloadOutcome:
      ok: bool
      path: Optional[str]
      classification: str  # "success" | "skip" | "error"
      reason: Optional[str]
      meta: Dict[str, Any]

  @dataclass(frozen=True)
  class ResolverResult:
      plans: List[DownloadPlan]
      notes: Dict[str, Any]
  ```

#### ❌ Actual
**Status: PARTIALLY EXISTS, DIFFERENT LOCATION**

- No `api/` directory
- No `api/types.py` file
- Related types scattered:
  - `WorkArtifact` in `core.py`
  - `DownloadContext` in `core.py`
  - Various helpers return dicts or custom classes
  - No unified contract across pipeline/helpers

### 5. CLI Polish (Typer Commands)

#### ✅ Planned
- **Location:** `cli/app.py`
- **Commands:**
  - `run` — main execution with overrides
  - `print-config` — show merged config
  - `validate-config` — fail on errors
  - `explain` — show resolver order & status
- **Options:**
  - `--config` / `-c` (file path)
  - `--resolver-order` (comma list)
  - `--no-robots`
  - `--no-atomic-write`
  - `--chunk-size`

#### ❌ Actual
**Status: NOT IMPLEMENTED**

- No `cli/app.py` file
- CLI handling in `cli.py` (legacy argparse, not Typer)
- No `print-config` command
- No `validate-config` command
- No `explain` command
- Config introspection missing

---

## Current Architecture vs. Planned

### Current (Dataclass-based)

```
CLI args (argparse)
  ↓
args.py (parse)
  ↓
DownloadConfig (dataclass)
  ↓
to_context() method
  ↓
DownloadContext (dataclass)
  ↓
Pipeline/Helpers (use dict access)
```

**Problems:**
- No strict validation
- No environment variable support
- No schema generation
- Resolvers hard-coded, not pluggable
- Config scattered across files
- No config introspection

### Planned (Pydantic v2)

```
YAML/JSON file
  ↓
load_config(path, env_prefix, cli_overrides)
  ↓
ContentDownloadConfig (Pydantic v2)
  ↓
build_resolvers(order, config)
  ↓
[Resolver1, Resolver2, ...] (dynamically instantiated)
  ↓
Pipeline uses:
  - DownloadPlan
  - DownloadStreamResult
  - DownloadOutcome
  - AttemptRecord
```

**Benefits:**
- Strict validation (`extra="forbid"`)
- Environment variable support
- CLI override composition
- JSON schema export for docs
- Registry pattern (plug new resolvers)
- Config as single source of truth
- Reproducible runs (config_hash)

---

## Missing Components Checklist

### Pydantic Models
- [ ] `config/models.py` — All config classes
  - [ ] RetryPolicy
  - [ ] BackoffPolicy
  - [ ] RateLimitPolicy
  - [ ] RobotsPolicy
  - [ ] DownloadPolicy
  - [ ] HttpClientConfig
  - [ ] TelemetryConfig
  - [ ] ResolverCommonConfig (+ resolver-specific configs)
  - [ ] ResolversConfig
  - [ ] ContentDownloadConfig (top-level)

### Config Infrastructure
- [ ] `config/loader.py` — File/env/CLI composition
- [ ] `config/schema.py` — JSON schema export (optional)
- [ ] Update `config/__init__.py` — Export public API

### Resolver Registry
- [ ] Add `@register()` decorator to `resolvers/__init__.py`
- [ ] Add `_REGISTRY` dict
- [ ] Add `get_registry()` function
- [ ] Add `build_resolvers()` function
- [ ] Decorate all existing resolvers with `@register(name)`
- [ ] Update `base.py` with Protocol definition

### API Surface
- [ ] `api/__init__.py` — Package init
- [ ] `api/types.py` — Unified types:
  - [ ] DownloadPlan
  - [ ] DownloadStreamResult
  - [ ] DownloadOutcome
  - [ ] ResolverResult
  - [ ] AttemptRecord

### CLI Modernization
- [ ] `cli/` package creation
- [ ] `cli/__init__.py`
- [ ] `cli/app.py` — Typer commands:
  - [ ] run (with --config, --resolver-order, etc.)
  - [ ] print-config
  - [ ] validate-config
  - [ ] explain

### Pipeline/Helper Refactoring
- [ ] Update helpers to use `api/types.py`
- [ ] Wire config into pipeline via dependency injection
- [ ] Remove ad-hoc config passing
- [ ] Update manifest recording to use unified types

---

## Risk Assessment

### Why This Wasn't Completed

1. **Scope Complexity** — Pydantic models + registry + CLI refactor is non-trivial
2. **Breaking Changes** — Old code depends on dataclass shape; migration requires careful planning
3. **Distributed Work** — Helpers/pipeline spread across many files; coordination overhead
4. **Team Priorities** — Other features (fallback, idempotency, telemetry) took precedence

### Why It Matters Now

1. **Technical Debt** — Config handling is scattered and hard to maintain
2. **Operator UX** — No config introspection; environment variables don't work cleanly
3. **Reproducibility** — No config_hash for run attribution
4. **Extensibility** — Adding new resolvers requires code changes, not config
5. **Testability** — Hard to mock/override configs in tests

---

## Implementation Path (Low-Risk)

### Phase 1: Foundation (No breaking changes yet)
1. Create `config/models.py` with Pydantic v2 models (parallel to existing dataclasses)
2. Create `config/loader.py` with file/env/CLI composition
3. Create `api/types.py` with unified types
4. Create `resolvers/__init__.py` registry (empty initially)
5. **No code changes to pipeline/helpers yet**

### Phase 2: Registry & Resolvers
1. Add `@register()` decorator to all resolvers
2. Create `build_resolvers()` function
3. Update resolver imports (from manual to registry)
4. **Test: resolver order & enable/disable work**

### Phase 3: Migration (Gradual cutover)
1. Create adapter layer: `ContentDownloadConfig` ↔ `DownloadConfig`
2. Update pipeline to accept both config types
3. Update helpers to use `api/types.py` incrementally
4. Run parallel: old dataclass + new Pydantic models
5. **Gradual migration, no hard cutover**

### Phase 4: CLI Modernization
1. Create `cli/app.py` with Typer
2. Keep existing `cli.py` for backward compatibility
3. Add new commands: print-config, validate-config, explain
4. **Operators can use new CLI without disruption**

### Phase 5: Cleanup
1. Remove old dataclass config classes
2. Remove legacy CLI code
3. Update docs/AGENTS.md

---

## Definition of Done (if we decide to implement)

- [ ] All Pydantic models created with strict validation (`extra="forbid"`)
- [ ] Config loader supports file/env/CLI precedence
- [ ] Resolver registry fully functional; `build_resolvers()` works
- [ ] API types unified (`DownloadPlan`, `DownloadOutcome`, etc.)
- [ ] CLI commands: run, print-config, validate-config, explain
- [ ] 100% test coverage for new config/registry/API modules
- [ ] Backward compatibility maintained (old dataclass code still works)
- [ ] Zero breaking changes in pipeline/helpers
- [ ] Documentation updated (ARCHITECTURE.md, AGENTS.md)
- [ ] No regression in existing tests

---

## Recommendation

**DO NOT implement immediately.** This scope is significant (~3-4 days of focused work) and not critical to current operations. However, it **should be scheduled** as a follow-up initiative because:

1. ✅ **Well-defined scope** (from attached documents)
2. ✅ **Low-risk path** (gradual migration possible)
3. ✅ **High-value payoff** (config management, extensibility, operator UX)
4. ⏱️ **Can be done incrementally** (no hard deadline)

**Suggested timeline:** Plan for Pillar 9 or 10 (after current observability/safety work completes).

---

## Summary Table

| Component | Planned | Current | Gap | Priority |
|-----------|---------|---------|-----|----------|
| Pydantic Models | models.py (10 classes) | None | MISSING | HIGH |
| Config Loader | loader.py | Manual parsing | MISSING | HIGH |
| Resolver Registry | @register decorator | Manual imports | MISSING | HIGH |
| API Types | api/types.py | Scattered | PARTIAL | MEDIUM |
| CLI Typer | cli/app.py (4 cmds) | argparse (1 cmd) | MAJOR | MEDIUM |
| Integration | ContentDownloadConfig → Pipeline | Dataclass → Pipeline | BREAKING | HIGH |

**Overall Status: 0% IMPLEMENTED, 100% SCOPED, READY TO START**

---

**AUDIT COMPLETE: SCOPE GAP IDENTIFIED, IMPLEMENTATION PATH DEFINED**
