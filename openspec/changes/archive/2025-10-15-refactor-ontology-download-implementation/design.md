# Design Document: Ontology Download Refactoring

## Context

The ontology download subsystem is a critical component of the DocsToKG pipeline responsible for acquiring, validating, and caching ontology resources from diverse external sources. The system must operate reliably under production conditions while maintaining compatibility with multiple resolver services, handling ontologies ranging from small vocabularies to multi-gigabyte knowledge bases, and providing operators with sufficient control and observability.

Current implementation has evolved organically and carries technical debt that impacts maintainability, reliability, and operational flexibility. This refactoring addresses accumulated issues systematically while preserving existing functionality and maintaining backward compatibility with deployed configurations.

### Stakeholders

- Pipeline operators requiring reliable ontology acquisition with minimal manual intervention
- Downstream knowledge graph construction processes depending on validated ontology artifacts
- Maintainers requiring clear module boundaries and testable components
- Resolver service providers expecting polite API behavior and rate limit compliance

### Constraints

- Must maintain backward compatibility with existing manifest format and configuration schema
- Cannot introduce new mandatory external dependencies to preserve deployment simplicity
- Must operate within memory constraints for environments processing large ontologies
- Must respect published rate limits for resolver services to maintain good citizenship
- Changes must be incrementally deployable without coordinated schema migrations

## Goals and Non-Goals

### Goals

- **Eliminate import cycles** enabling isolated unit testing and cleaner package structure
- **Reduce code duplication** through centralized archive handling and retry logic
- **Improve memory efficiency** for large ontology normalization avoiding out-of-memory failures
- **Increase download success rate** through automatic resolver fallback on transient failures
- **Enhance operational control** via CLI overrides for concurrency and allowlists
- **Strengthen system observability** through comprehensive diagnostics and planning introspection

### Non-Goals

- **Changing manifest schema version** - existing manifests remain valid without migration
- **Altering configuration file structure** - existing configuration files continue working
- **Adding GraphQL or REST API layer** - changes remain internal to Python package
- **Implementing distributed caching** - storage remains local or single fsspec backend
- **Creating web UI or dashboard** - CLI remains primary operator interface

## Architectural Decisions

### AD-1: Logging Configuration as Pure Function

**Decision**: Make `setup_logging()` accept all configuration as explicit parameters without importing from other package modules.

**Rationale**: Import cycles between logging configuration and orchestration modules create fragile initialization order dependencies. Tests cannot initialize logging in isolation. Pure function enables explicit dependency injection and simplifies testing.

**Alternatives Considered**:

- **Singleton pattern with lazy initialization**: Complexity of singleton lifecycle management outweighs benefits. Hidden dependencies make testing difficult.
- **Separate configuration module for logging**: Adds module count without solving fundamental issue of circular dependencies through configuration objects.

**Trade-offs**: Calling code must explicitly pass logging parameters. Accepted because it makes dependencies visible and enables better testing.

### AD-2: Streaming Normalization for Large Ontologies

**Decision**: Implement dual-path normalization using in-memory fast path for small ontologies and streaming path for large ontologies with external sort.

**Rationale**: In-memory normalization loads entire ontology, sorts triples in memory, and serializes. This consumes memory proportional to ontology size causing failures on multi-gigabyte ontologies. Streaming approach uses disk for sorting ensuring bounded memory usage.

**Alternatives Considered**:

- **Pure in-memory with increased memory limits**: Requires environment-specific memory configuration. Does not scale to arbitrarily large ontologies.
- **Always use streaming path**: Slower for small ontologies where in-memory path completes in milliseconds. Performance regression for common case.
- **Chunked sorting in Python**: Platform sort is optimized and battle-tested. Pure Python implementation adds complexity without clear benefit.

**Trade-offs**: Requires platform sort command availability. Fallback to Python merge sort provides pure-Python escape hatch. Threshold configuration allows operators to tune based on environment.

**Implementation Notes**:

- Serialize to N-Triples format for line-oriented sorting simplicity
- External sort provides deterministic ordering across platforms
- Stream sorted output computing SHA-256 incrementally avoiding second pass
- Delete temporaries in finally block ensuring cleanup on exception

### AD-3: Unified Retry Helper with Exponential Backoff

**Decision**: Create single `retry_with_backoff()` helper function consolidating retry logic across resolvers and downloaders.

**Rationale**: Retry implementations scattered across modules use inconsistent backoff strategies and error classification. Centralized helper ensures uniform behavior and simplifies testing of retry logic.

**Alternatives Considered**:

- **Decorator-based retry**: Less flexible for varying retry predicates and parameter configuration per call site.
- **Class-based retry policy**: Overengineered for current needs. Simple function sufficient for current use cases.

**Trade-offs**: Additional indirection through function call. Accepted because consistency and testability benefits outweigh minimal performance impact.

**Implementation Notes**:

- Accept predicate function for error classification enabling different retryable criteria per use case
- Add jitter to prevent thundering herd when multiple processes retry simultaneously
- Optional callback for logging enables structured telemetry without coupling retry logic to logging

### AD-4: Download-Time Resolver Fallback

**Decision**: Extend planning to return candidate list and implement download-time fallback iterating candidates on retryable failures.

**Rationale**: Planning succeeds but download fails due to transient service issues, rate limiting, or authentication glitches. Resolver fallback increases download success rate without manual intervention.

**Alternatives Considered**:

- **Replan on download failure**: Incurs additional API latency requerying resolver services. May hit same transient failure.
- **Fail immediately and require manual intervention**: Poor operator experience. Increases operational burden unnecessarily.

**Trade-offs**: Planning produces larger candidate set consuming additional memory. Accepted because typical ontology count in hundreds not thousands.

**Implementation Notes**:

- Populate candidates during initial planning when multiple resolvers configured
- Record attempt chain in manifest for audit trail and debugging
- Preserve polite headers across fallback attempts
- Stop on non-retryable errors like authentication failure to avoid wasting attempts

### AD-5: Parallel Resolver Planning with Per-Service Limits

**Decision**: Use thread pool for concurrent API calls during planning with per-service token buckets limiting concurrent requests per resolver service.

**Rationale**: Resolver API calls are network-bound with high latency. Sequential planning incurs wall-clock penalty proportional to ontology count. Parallel planning reduces total time while per-service limits respect provider rate limits.

**Alternatives Considered**:

- **Async/await with asyncio**: Requires async resolver API clients. Most resolver libraries use synchronous requests. Threading simpler given existing synchronous code.
- **Process pool for true parallelism**: Network I/O dominates CPU usage making GIL impact negligible. Threads sufficient and lighter weight than processes.
- **Global concurrency limit only**: Does not prevent overwhelming single service with too many concurrent requests. Per-service limits provide finer-grained control.

**Trade-offs**: Thread pool overhead and context switching. Accepted because network latency dominates and benefits outweigh costs.

**Implementation Notes**:

- Default max workers to eight based on typical resolver count and desired parallelism
- Token bucket per service identified by resolver name prevents exceeding published rate limits
- Use `as_completed()` for result streaming avoiding waiting for slowest service
- Catch exceptions per-future allowing partial success when some resolvers fail

### AD-6: Archive Extraction Centralization

**Decision**: Create single `extract_archive_safe()` function dispatching on file extension with uniform security checks.

**Rationale**: Duplicate extraction code across validators creates inconsistent security checks and maintenance burden. Single implementation ensures uniform path traversal prevention and compression bomb detection.

**Alternatives Considered**:

- **Extraction library wrapper**: No mature archive abstraction library providing both security hardening and format coverage. Building wrapper overkill for current needs.
- **Validator-specific extraction**: Preserves flexibility but maintains duplication and inconsistency.

**Trade-offs**: Centralized function requires supporting multiple archive formats increasing implementation complexity. Accepted because security and consistency benefits critical.

**Implementation Notes**:

- Detect format from double extensions like `.tar.gz` before single extension
- Validate each member path before extraction rejecting absolute paths and traversal attempts
- Calculate compression ratio before extraction rejecting suspicious archives
- Support ZIP, TAR, TGZ, TXZ formats covering observed ontology distribution formats
- Return list of extracted paths for validator processing

### AD-7: Module-Based Worker Execution

**Decision**: Modify worker scripts to support module execution pattern and invoke using `-m` flag from subprocess calls.

**Rationale**: File path imports and `sys.path` manipulation breaks in packaged installations and confuses IDE tooling. Module execution uses standard Python import machinery.

**Alternatives Considered**:

- **Continue using file path imports**: Maintains current approach but perpetuates packaging issues and IDE confusion.
- **Install workers as separate entry points**: Overcomplicates installation for internal helper scripts.

**Trade-offs**: Workers must define proper `__main__` guard and argument parsing. Accepted because this follows Python best practices.

**Implementation Notes**:

- Add `if __name__ == "__main__":` guard to worker module
- Parse command-line arguments for validator selection within guard block
- Build subprocess command as list with `sys.executable`, `-m`, full module name, arguments
- Remove all `sys.path.insert()` calls from worker code
- Test worker invocation from clean environment without source tree

### AD-8: CLI Operational Enhancements

**Decision**: Add CLI flags for concurrency control, allowlist override, version pruning, and planning introspection.

**Rationale**: Operators frequently need temporary configuration overrides for ad-hoc operations. Requiring configuration file edits for one-off operations creates friction and risk of forgotten changes.

**Alternatives Considered**:

- **Environment variable overrides**: Less discoverable than CLI flags. No help text or validation.
- **Separate configuration files**: Overhead of managing multiple configurations. Copy-paste errors.

**Trade-offs**: Increased CLI argument surface area. Accepted because operational flexibility justifies additional arguments.

**Implementation Notes**:

- Merge CLI arguments with loaded configuration before orchestration
- CLI values take precedence over configuration file values
- Validate argument values at parse time with descriptive errors
- Document flags in help text with operational examples

## Risks and Mitigations

### Risk: Streaming Normalization Determinism Across Platforms

**Impact**: High - non-deterministic hashes break cache validation and manifest verification.

**Likelihood**: Medium - platform sort implementations vary subtly.

**Mitigation**:

- Extensive determinism testing across Linux and macOS
- Golden hash fixtures validated in CI on multiple platforms
- Fallback to pure Python sort if platform sort produces inconsistent results
- Document sort collation requirements in normalization specification

### Risk: Thread Pool Deadlock Under Error Conditions

**Impact**: High - deadlock blocks pipeline indefinitely requiring manual intervention.

**Likelihood**: Low - concurrent.futures implementation mature and well-tested.

**Mitigation**:

- Use context manager for thread pool ensuring cleanup on exception
- Set timeout on future results preventing indefinite blocking
- Comprehensive exception handling per future avoiding exception leaking across threads
- Test with simulated failures and timeouts

### Risk: Resolver Fallback Amplifying Traffic to Failing Services

**Impact**: Medium - fallback attempts contribute to overload on degraded services.

**Likelihood**: Medium - transient failures often indicate service overload.

**Mitigation**:

- Respect retry delays and exponential backoff before fallback attempt
- Classify authentication failures as non-retryable avoiding wasted attempts
- Monitor and log fallback frequency enabling operator visibility into service health
- Consider circuit breaker pattern if fallback frequency becomes problematic

### Risk: Breaking Changes in External Dependencies

**Impact**: Medium - new versions of rdflib or resolver clients may alter behavior.

**Likelihood**: Low - dependencies maintained by stable organizations with semantic versioning.

**Mitigation**:

- Pin dependency versions in requirements file
- Test with dependency version matrix in CI covering older and newer versions
- Document known compatible version ranges
- Monitor dependency changelogs for relevant changes

## Migration Plan

### Phase 1: Code Quality Foundation (PR-1)

**Scope**: Logging purity, configuration deprecations, module structure cleanup.

**Validation**: Existing tests pass, new deprecation warnings appear as expected, logging works in isolation.

**Rollback**: Revert PR. No manifest or configuration changes.

### Phase 2: Archive and Worker Execution (PR-2)

**Scope**: Unified archive extraction, module-based worker execution, optional dependency stubs.

**Validation**: Existing validators produce identical outputs, archive security tests pass, workers run from clean environment.

**Rollback**: Revert PR. No changes to stored artifacts or manifests.

### Phase 3: CLI Operational Features (PR-3)

**Scope**: Concurrency flags, allowlist override, enhanced doctor command.

**Validation**: CLI help text updated, flags override configuration correctly, doctor command provides useful diagnostics.

**Rollback**: Revert PR. No breaking changes to existing CLI invocations.

### Phase 4: Reliability Core (PR-4)

**Scope**: Unified retry, resolver fallback, streaming normalization, parallel planning.

**Validation**: Download success rate increases, large ontologies normalize successfully, planning wall-clock time decreases.

**Rollback**: More complex due to manifest fingerprint changes. Consider feature flag for streaming normalization enabling gradual rollout.

### Phase 5: Version Management (PR-5)

**Scope**: Prune command, planning introspection, manifest schema version.

**Validation**: Prune deletes correct versions, plan diff identifies changes accurately, schema version appears in new manifests.

**Rollback**: Revert PR. Existing manifests without schema version continue working.

## Open Questions

1. **Should streaming normalization threshold be per-ontology configurable?**
   - Current design uses global threshold. Some ontologies may benefit from lower threshold based on complexity.
   - Resolution: Start with global, add per-ontology override in extras if needed emerges.

2. **What is appropriate default for per-service rate limits?**
   - Current design defaults to per-host limit. May need specific defaults for known services.
   - Resolution: Research published rate limits, document in configuration example, allow override.

3. **Should resolver fallback candidates be configurable or automatic?**
   - Current design populates all resolvers. May want explicit fallback chains.
   - Resolution: Start automatic based on configuration order, add explicit chains if use case emerges.

4. **How to handle schema version migration for existing manifests?**
   - Current design adds field to new manifests, tolerates absence in old manifests.
   - Resolution: Add migration command in future if batch rewriting needed.

## References

- OpenSpec process: `openspec/AGENTS.md`
- Existing ontology download code: `src/DocsToKG/OntologyDownload/`
- Related change: `enhance-ontology-downloader-robustness` (completed)
- RDFLib documentation: <https://rdflib.readthedocs.io/>
- OLS API documentation: <https://www.ebi.ac.uk/ols/docs/api>
- BioPortal API documentation: <https://data.bioontology.org/documentation>
