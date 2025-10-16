# Harden Ontology Downloader Core

## Why

The ontology downloader has accumulated technical debt and operational fragility that undermines its core mission: **reliably fetch ontologies from diverse sources, validate their correctness, and produce deterministic, auditable artifacts**. Current issues include:

1. **Legacy code sprawl**: Duplicate utility functions between CLI and core pipeline, unused module import aliases that hide dead code, and repeated exception handling blocks create maintenance burden and drift risk.

2. **Correctness bugs**: A critical bug in URL security validation during planning passes the wrong parameter type, risking crashes rather than clean error handling. Sequential validator execution increases end-to-end latency unnecessarily.

3. **Scalability gaps**: Large ontology normalization claims to be "streaming" but still materializes all triples in memory before sorting, defeating the purpose for ontologies exceeding available RAM. No inter-process locking allows concurrent runs to corrupt version directories.

4. **Boundary violations**: The CLI duplicates metadata enrichment logic that the library already performs, violating single-source-of-truth principles and increasing test surface area.

These defects compound operator friction, increase failure rates for large ontologies, and make the system harder to reason about and maintain.

## What Changes

This change consolidates, hardens, and clarifies the ontology downloader scope through targeted refactoring:

### Code Reduction & Consolidation

- **Remove legacy module import aliases** from `__init__.py` that map old module paths to consolidated implementations
- **Unify duplicate utilities**: rate-limit parsing, directory sizing, datetime parsing, plan metadata enrichment, latest symlink management between CLI and core into single implementations
- **Collapse repeated exception handling blocks** in validator functions
- **Eliminate CLI metadata probing** that duplicates library-level enrichment

### Correctness & Robustness

- **Fix URL security validation bug** in `_populate_plan_metadata` to pass `DownloadConfiguration` instead of bare host list
- **Make validators concurrent** with configurable bounded parallelism (default 2 workers, max 8) while preserving result structure and per-validator JSON artifacts
- **Add inter-process version locking** using platform-appropriate file locks to prevent concurrent writes to the same ontology version directory

### Performance & Scale

- **Implement true streaming normalization** using external sort (platform `sort` command or Python merge-sort fallback) with incremental SHA-256 computation to handle ontologies larger than available memory
- **Add streaming threshold configuration** `streaming_normalization_threshold_mb` (default 200MB) with logging of normalization mode used

### Operator Experience

- **Storage backend consistency**: Update `prune` command to call `STORAGE.set_latest_version` instead of duplicating symlink logic
- **Validation improvements**: Add unit tests for allowlist enforcement, IDN checks, HEAD 405 fallback, ETag cache hits, resume logic, and archive safety

### Extensibility (Future-Proofing)

- **Plugin infrastructure** for resolvers and validators via `importlib.metadata.entry_points` with fail-soft loading and dedicated private entry point groups (`docstokg.ontofetch.resolver`, `docstokg.ontofetch.validator`)
- **Manifest schema migration shim** in `_read_manifest` to forward-compatibly handle older on-disk manifests during schema evolution

## Impact

**Affected specs:** `ontology-download` (new spec)

**Affected code:**

- `src/DocsToKG/OntologyDownload/__init__.py` — Remove `_LEGACY_MODULE_MAP` and alias installation
- `src/DocsToKG/OntologyDownload/ontology_download.py` — Fix URL validation bug, add concurrency config, implement true streaming normalization, add version locking, unify utilities, add migration shim
- `src/DocsToKG/OntologyDownload/cli.py` — Remove duplicate metadata probing, use shared utilities, update prune to call storage backend
- `src/DocsToKG/OntologyDownload/resolvers.py` — Add plugin loader for resolver extensibility
- `tests/` — Add comprehensive unit tests for security, concurrency, streaming, and locking

**Breaking changes:**

- **BREAKING**: Legacy module import paths removed (e.g., `DocsToKG.OntologyDownload.core`, `.config`, `.validators`, etc.). Imports must use the public API from `DocsToKG.OntologyDownload` directly or the consolidated `.ontology_download` module.

**Migration path:**

1. Update all imports to use symbols exported from `DocsToKG.OntologyDownload.__all__`
2. Grep codebase for legacy imports: `rg "from DocsToKG.OntologyDownload\.(core|config|validators|download|storage|optdeps|utils|logging_config|validator_workers|foundation|infrastructure|network|pipeline|settings|validation|cli_utils)" --files-with-matches`
3. Test suite must pass with no legacy imports remaining

**Deployment strategy:** This change requires a minor version bump and coordinated update of any downstream consumers. The change is backward-incompatible for import paths only; all public API functions remain stable.
