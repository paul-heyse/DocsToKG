# Design: Harden Ontology Downloader Core

## Context

The ontology downloader serves as the foundational layer for acquiring, validating, and normalizing ontologies from heterogeneous sources (OBO, OLS, BioPortal, LOV, SKOS, XBRL). Its output—schema-validated manifests, deterministic normalized Turtle, and per-validator JSON artifacts—enables downstream systems to trust provenance and detect changes without re-parsing.

Current implementation has strong bones: secure downloads with SSRF protection, resolver fallback chains, normalization with blank node canonicalization, and subprocess-isolated validators. However, accumulated technical debt (duplicate logic, legacy imports, sequential validation) and scalability gaps (in-memory normalization for "streaming" mode, no inter-process locking) undermine operator confidence and increase failure rates for large ontologies.

**Constraints:**

- Must maintain backward compatibility for public API surface (`fetch_all`, `plan_all`, manifest schema)
- Cannot introduce new runtime dependencies beyond standard library where possible
- Must preserve deterministic SHA-256 fingerprints (changing normalization algorithm would invalidate existing manifests)
- Must keep "download & validate only" boundary—no downstream coupling

**Stakeholders:**

- Pipeline operators needing reliable, auditable downloads
- Downstream consumers depending on manifest schema stability
- Future maintainers requiring clear, consolidated code

## Goals / Non-Goals

### Goals

1. **Reduce code volume and complexity** by eliminating duplication between CLI and core, removing legacy import shims, and consolidating utilities
2. **Fix correctness bugs** that risk crashes or inconsistent behavior (URL validation parameter type, sequential validators)
3. **Scale to large ontologies** by implementing true streaming normalization with external sort and incremental hashing
4. **Prevent concurrent write corruption** via inter-process file locking for version directories
5. **Enable safe extensibility** through plugin infrastructure for resolvers and validators with fail-soft loading
6. **Future-proof manifest schema** with migration shim for forward-compatible reads

### Non-Goals

- Changing the manifest schema version (remains "1.0")
- Altering the normalization algorithm's output (must preserve SHA-256 determinism)
- Adding new resolver implementations (infrastructure only, not new resolvers)
- Changing the public API function signatures (`fetch_all`, `plan_all`, etc.)
- Optimizing resolver API latency (keep existing retry/backoff/rate-limit logic)

## Decisions

### 1. Legacy Import Removal

**Decision:** Remove all legacy module import aliases immediately with a breaking change notice rather than deprecation period.

**Rationale:**

- The `_LEGACY_MODULE_MAP` shims add ~15 entries that obscure the true API surface
- Deprecation warnings would pollute logs without forcing action
- Clean break enables immediate simplification and reduces cognitive load
- Modern imports are already in use throughout the test suite

**Alternatives considered:**

- Emit `DeprecationWarning` for one release cycle: Adds complexity for limited value; operators unlikely to notice warnings in production logs
- Keep shims indefinitely: Perpetuates confusion and makes dead code detection impossible

**Trade-offs:**

- Risk: Downstream consumers using legacy imports will break immediately
- Mitigation: Grep-based detection is simple (`rg "from DocsToKG.OntologyDownload\.(core|config|...)"`) and fixes are mechanical
- Migration cost is one-time and low; ongoing maintenance cost of shims is perpetual

### 2. Concurrent Validator Execution

**Decision:** Use `ThreadPoolExecutor` with bounded concurrency (default 2, max 8) for validator parallelism.

**Rationale:**

- Pronto and Owlready2 validators already execute in **subprocesses** (via `_run_validator_subprocess`), so thread safety is guaranteed
- Thread pool overhead is negligible compared to subprocess spawn and RDF parsing time
- Bounded concurrency prevents CPU/memory thrashing on resource-constrained hosts
- Preserves existing result structure and logging behavior

**Alternatives considered:**

- `ProcessPoolExecutor`: Adds unnecessary overhead since validators are already subprocesses; thread pool is simpler
- `asyncio`: RDFLib and validator libraries are synchronous; thread pool avoids async ceremony
- Keep sequential execution: Lowest risk but leaves 2-3x latency on the table for multi-validator runs

**Trade-offs:**

- Thread pool initialization adds ~10ms overhead per `run_validators` call (negligible)
- Configuration burden (new `max_concurrent_validators` field) is minimal and provides operator control

### 3. True Streaming Normalization

**Decision:** Rewrite `normalize_streaming` to use external sort (platform `sort` or Python merge-sort fallback) with incremental SHA-256.

**Rationale:**

- Current implementation materializes all triples in memory via `list(graph)` before sorting, defeating the "streaming" claim
- RDFLib's N-Triples serialization is line-oriented and streamable; sorting is the bottleneck
- Platform `sort` is highly optimized for disk-based sorting; Python `heapq.merge` provides portable fallback
- Incremental hashing during write avoids re-reading normalized output

**Algorithm:**

1. Parse source ontology into RDFLib Graph (no change)
2. Serialize Graph to unsorted N-Triples tempfile (streaming, no Python list)
3. External sort N-Triples file:
   - Try `subprocess.run(["sort", "-o", dest, source])` with timeout
   - Fallback to Python merge-sort using `heapq.merge` with memory-bounded chunks
4. Stream-read sorted triples, apply deterministic blank node renumbering (regex substitution with memoization dict)
5. Write sorted `@prefix` lines (pre-collected from namespace manager), then blank line, then canonicalized triples
6. Compute SHA-256 incrementally as lines are written; never store full content in memory

**Alternatives considered:**

- Use RDFLib's native `serialize()` with streaming: RDFLib sorts in-memory; no benefit
- Skip sorting, hash unsorted triples: Breaks determinism; manifests become order-dependent
- Implement custom RDF parser: High risk, high complexity; leverage RDFLib's battle-tested parsing

**Trade-offs:**

- Platform `sort` dependency is acceptable (universally available on Unix/Linux; Python fallback for Windows/unusual platforms)
- Disk I/O for sorting trades memory for latency; acceptable for large ontologies (minutes to parse > seconds to sort)
- Regex blank node renumbering is O(triples × patterns) but patterns are few (typically 1-2 per ontology)

### 4. Inter-Process Version Locking

**Decision:** Use platform file locking (`fcntl.flock` on Unix, `msvcrt.locking` on Windows) with lock files in `CACHE_DIR/locks/`.

**Rationale:**

- Prevents silent corruption when multiple processes (e.g., parallel CI jobs, operator retries) attempt to write the same `ontology_id/version` directory
- File locks are OS-managed, process-scoped, and automatically released on crash
- Lock granularity is per-version (not per-ontology) to allow concurrent downloads of different versions

**Lock protocol:**

1. Before `download_stream`, acquire lock on `{safe_id}__{safe_version}.lock` in blocking mode
2. Perform download, extraction, normalization, validation
3. Release lock in `finally` block (guaranteed cleanup)
4. Lock file persists (empty, small); prune with cache management

**Alternatives considered:**

- Redis/database locks: Adds external dependency; overkill for local filesystem problem
- PID file with stale detection: Fragile (race conditions, PID reuse); standard file locks are safer
- Advisory locks via `flock` flag files: Equivalent to chosen approach but less portable

**Trade-offs:**

- Lock files accumulate in `CACHE_DIR/locks/` (one per unique `id/version` pair); negligible storage cost
- Blocking lock may delay concurrent runs; acceptable given writes are infrequent and short-lived

### 5. Plugin Infrastructure

**Decision:** Use `importlib.metadata.entry_points` with private entry point groups and fail-soft loading.

**Rationale:**

- Enables downstream projects to register custom resolvers/validators without forking
- `entry_points` is standard packaging infrastructure (PEP 517/518); no new dependencies
- Private group names (`docstokg.ontofetch.resolver`, `.validator`) avoid namespace collisions
- Fail-soft loading (catch all exceptions, log warning, continue) preserves robustness

**Entry point groups:**

- `docstokg.ontofetch.resolver`: Plugins must implement `plan(spec, config, logger) -> FetchPlan` and optionally provide `NAME` attribute
- `docstokg.ontofetch.validator`: Plugins must be callables matching `(ValidationRequest, logging.Logger) -> ValidationResult`

**Loading sequence:**

1. After module-level `RESOLVERS`/`VALIDATORS` dicts are populated with built-in implementations
2. Query entry points for the relevant group
3. For each entry point, attempt `ep.load()` and instantiate/verify interface
4. Register in global dict with `NAME` or `ep.name` as key
5. Log info-level success or warning-level failure (with exception details)

**Alternatives considered:**

- Config-file-based plugin discovery: Requires inventing DSL; entry points are standard and tool-supported
- Dynamic import of modules by name: Fragile (namespace collisions); entry points provide isolation
- No plugin system: Forces forks for custom resolvers; prevents ecosystem growth

**Trade-offs:**

- Plugin authors must package with `pyproject.toml` entry points; acceptable burden for extensibility benefit
- Bad plugins can crash module import; fail-soft loading mitigates to warning-level impact

### 6. Manifest Schema Migration

**Decision:** Implement in-place migration shim in `_read_manifest` keyed by `schema_version` field.

**Rationale:**

- Allows schema evolution (future field additions, renames, type changes) without breaking older on-disk manifests
- Migrations run only on read (not write), so operator can defer upgrades
- Idempotent migrations (check version, apply transform, set new version) are easy to test
- Private implementation (not exposed in public API) keeps surface area small

**Migration pattern:**

```python
def _migrate_manifest_inplace(payload: dict) -> None:
    version = str(payload.get("schema_version", ""))
    if version in {"1.0", ""}:
        payload.setdefault("schema_version", "1.0")  # Backfill if missing
        return
    if version == "0.9":  # Example future migration
        payload["schema_version"] = "1.0"
        payload.setdefault("resolver_attempts", [])  # Add new field
        payload.pop("deprecated_field", None)  # Remove old field
        return
    # Unknown version: log warning, attempt validation anyway
```

**Alternatives considered:**

- Write migration scripts for operators to run manually: High friction, error-prone
- Maintain multiple schema validators: Complexity scales with schema history
- Break compatibility on schema changes: Unacceptable for mature systems with archived data

**Trade-offs:**

- Migration code accumulates over time; acceptable given clear version progression and rare schema changes
- Forward-compatibility only (newer code reads older manifests); older code cannot read newer manifests (fails validation)

## Risks / Trade-offs

### Risk: Breaking Change Impact on Downstream Consumers

**Probability:** Medium (some projects likely import legacy paths)
**Impact:** High (import errors block entire pipeline)

**Mitigation:**

- Provide grep command in migration guide for exhaustive detection
- Bumps minor version to signal breaking change per semantic versioning
- Update CHANGELOG with prominent notice and actionable steps

### Risk: Streaming Normalization Alters SHA-256

**Probability:** Low (algorithm preserves determinism by design)
**Impact:** Critical (would invalidate all existing manifests)

**Mitigation:**

- Comprehensive unit tests verify byte-for-byte output equivalence between old and new implementation
- Property-based tests (hypothesis) with random graphs verify determinism across runs
- Manual verification with reference ontologies (pizza.owl, go.owl) before release

### Risk: Concurrent Validators Introduce Race Conditions

**Probability:** Low (thread-safe by design via subprocess isolation)
**Impact:** High (corrupted validation outputs)

**Mitigation:**

- Each validator writes to unique JSON file (no shared state)
- Logging uses thread-safe standard library logger
- Integration tests exercise concurrent validator execution and verify result integrity

### Risk: Plugin System Enables Supply Chain Attacks

**Probability:** Low (requires malicious package installation)
**Impact:** High (arbitrary code execution during import)

**Mitigation:**

- Private entry point groups (`docstokg.*`) prevent namespace squatting
- Fail-soft loading limits blast radius to warning log
- Documentation emphasizes vetting third-party plugins

### Risk: File Locks Cause Deadlocks

**Probability:** Low (locks are version-scoped, short-held)
**Impact:** Medium (hung processes requiring manual intervention)

**Mitigation:**

- Use blocking locks (no polling busy-loops)
- Lock released in `finally` block (guaranteed cleanup even on exception)
- Lock granularity is per-version (not per-ontology), reducing contention

## Migration Plan

### Phase 1: Code Consolidation (Low Risk)

1. Remove legacy import aliases
2. Unify duplicate utilities (rate-limit, directory size, datetime parsing)
3. Collapse repeated exception blocks
4. Remove CLI metadata probing duplication

**Success criteria:** Test suite passes, no functionality changes

### Phase 2: Correctness Fixes (Medium Risk)

1. Fix URL validation bug in `_populate_plan_metadata`
2. Add inter-process version locking
3. Make validators concurrent

**Success criteria:** New unit tests pass, integration tests stable

### Phase 3: Streaming Normalization (High Risk)

1. Implement external sort with fallback
2. Rewrite `normalize_streaming` with incremental hashing
3. Verify SHA-256 determinism with reference ontologies

**Success criteria:** Byte-for-byte output match with old implementation, property tests pass

### Phase 4: Extensibility & Polish (Low Risk)

1. Add plugin infrastructure
2. Add manifest migration shim
3. Update documentation and examples

**Success criteria:** Example plugins load successfully, migration shim tested

### Rollback Strategy

**Pre-deployment checkpoints:**

- Full test suite green (unit, integration, property-based)
- Manual smoke tests with small and large ontologies
- Reference ontology SHA-256 verification

**If normalization SHA-256 mismatch detected:**

1. Revert streaming normalization changes
2. Keep all other improvements (consolidation, correctness fixes, concurrency)
3. Investigate determinism failure in isolated environment

**If breaking changes cause widespread failures:**

1. Tag rollback release with legacy imports restored temporarily
2. Communicate migration timeline extension
3. Provide automated migration script for common patterns

## Open Questions

1. **Should we add a `--validate-only` mode to CLI for re-running validators without re-downloading?**
   - Current `ontofetch validate` command already supports this
   - Decision: Document existing capability, no new code needed

2. **Should streaming normalization threshold be per-format (TTL vs OWL) or global?**
   - Decision: Keep global threshold (simpler); operators can override per-ontology via config if needed
   - Rationale: File size is format-agnostic; RDFLib parsing behavior is similar across formats

3. **Should plugin infrastructure support unloading/reloading for development workflows?**
   - Decision: No (out of scope)
   - Rationale: Module-level registration is simple and sufficient; dynamic reloading adds complexity and footguns

4. **Should we expose `_version_lock` as a public context manager for downstream use?**
   - Decision: No (keep internal for now)
   - Rationale: Locking strategy may evolve; avoid locking API to specific implementation

## Summary

This design consolidates technical debt, fixes critical correctness bugs, and scales the ontology downloader to handle large ontologies without memory constraints. Key decisions:

- **Break legacy imports cleanly** rather than deprecation period (one-time migration cost, perpetual maintenance savings)
- **Parallelize validators with bounded concurrency** (2-3x latency reduction with safety guardrails)
- **True streaming normalization via external sort** (enables ontologies larger than RAM with deterministic output)
- **Inter-process locking** (prevents concurrent write corruption)
- **Plugin infrastructure with fail-soft loading** (extensibility without fragility)
- **Manifest migration shim** (forward-compatible schema evolution)

The implementation preserves backward compatibility for public APIs and manifest schemas while modernizing internal architecture. Total estimated scope: ~1500 lines changed, ~500 lines new tests, ~200 lines removed (net -300 lines).
