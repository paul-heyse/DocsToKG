# Implementation Tasks

## 1. Legacy Cleanup & Consolidation

- [ ] 1.1 **Remove legacy module import aliases**
  - [ ] 1.1.1 Delete the `_LEGACY_MODULE_MAP` dictionary and its associated loop in `src/DocsToKG/OntologyDownload/__init__.py`
  - [ ] 1.1.2 Scan the entire codebase for imports using legacy paths: `rg "from DocsToKG.OntologyDownload\.(core|config|validators|download|storage|optdeps|utils|logging_config|validator_workers|foundation|infrastructure|network|pipeline|settings|validation|cli_utils)" --files-with-matches`
  - [ ] 1.1.3 Update all found imports to use the public API symbols from `DocsToKG.OntologyDownload.__all__` or `.ontology_download`/`.cli` directly
  - [ ] 1.1.4 Run full test suite to verify no import errors remain
  - [ ] 1.1.5 Update CHANGELOG.md with breaking change notice and migration instructions

- [ ] 1.2 **Unify rate-limit pattern utilities**
  - [ ] 1.2.1 Keep the `_RATE_LIMIT_PATTERN` regex in `ontology_download.py` as the single source of truth
  - [ ] 1.2.2 Remove `_RATE_LIMIT_RE` from `cli.py`
  - [ ] 1.2.3 Extract a small helper function `parse_rate_limit_to_rps(limit_str: str) -> Optional[float]` in the core module
  - [ ] 1.2.4 Update `cli.py` to import and use this shared helper in `_rate_limit_to_rps`
  - [ ] 1.2.5 Add unit tests verifying rate limit parsing for various formats

- [ ] 1.3 **Unify directory size utilities**
  - [ ] 1.3.1 Keep the `_directory_size` function in `ontology_download.py` as canonical
  - [ ] 1.3.2 Remove `_directory_size` and `_directory_size_bytes` duplicates from `cli.py`
  - [ ] 1.3.3 Update all callsites in `cli.py` to import and use the core implementation
  - [ ] 1.3.4 Add unit tests with temporary directories to verify consistent behavior

- [ ] 1.4 **Unify datetime parsing utilities**
  - [ ] 1.4.1 Consolidate `_parse_iso_datetime`, `_parse_http_datetime`, `_parse_version_timestamp` from CLI with core's `_coerce_datetime`, `_normalize_timestamp`, `_parse_last_modified`
  - [ ] 1.4.2 Create a single unified datetime utility section in `ontology_download.py` with clear functions for each use case
  - [ ] 1.4.3 Update CLI to reuse these core functions
  - [ ] 1.4.4 Add comprehensive unit tests for ISO, HTTP, and version timestamp formats

- [ ] 1.5 **Remove CLI metadata probing duplication**
  - [ ] 1.5.1 Remove calls to `_collect_plan_metadata(plans, config)` from `_handle_plan` and `_handle_plan_diff` in `cli.py`
  - [ ] 1.5.2 Verify that `plan_all` already enriches `PlannedFetch.metadata` with `last_modified`, `etag`, and `content_length` via `_populate_plan_metadata`
  - [ ] 1.5.3 Delete the now-unused `_collect_plan_metadata` helper function from `cli.py`
  - [ ] 1.5.4 Delete the helper `_extract_response_metadata` if no longer used
  - [ ] 1.5.5 Run CLI integration tests to verify `plan` and `plan-diff` commands still function correctly
  - [ ] 1.5.6 Verify that `_plan_to_dict` correctly serializes metadata from `plan.metadata` dictionary

- [ ] 1.6 **Unify latest symlink management**
  - [ ] 1.6.1 Remove the `_update_latest_symlink` function from `cli.py`
  - [ ] 1.6.2 Update `_handle_prune` to call `STORAGE.set_latest_version(ontology_id, retained[0]["path"])` after pruning
  - [ ] 1.6.3 Verify that the storage backend's `set_latest_version` handles both symlinks and fallback text files correctly
  - [ ] 1.6.4 Add integration test for prune operation verifying correct latest marker update

- [ ] 1.7 **Collapse repeated exception blocks**
  - [ ] 1.7.1 Locate the four consecutive identical `except Exception as exc` blocks in `validate_pronto` function in `ontology_download.py`
  - [ ] 1.7.2 Reduce to a single exception handler at the appropriate scope
  - [ ] 1.7.3 Verify behavior is unchanged (writes JSON, logs warning, returns `ok=False`)
  - [ ] 1.7.4 Add regression test that triggers an exception in Pronto validation

## 2. Correctness & Robustness

- [ ] 2.1 **Fix URL security validation bug**
  - [ ] 2.1.1 Locate the call to `validate_url_security(planned.plan.url, config.defaults.http.allowed_hosts)` in `_populate_plan_metadata` function
  - [ ] 2.1.2 Change the second argument from `config.defaults.http.allowed_hosts` to `config.defaults.http` (passing the full `DownloadConfiguration`)
  - [ ] 2.1.3 Verify that `validate_url_security` signature accepts `Optional[DownloadConfiguration]` as the second parameter
  - [ ] 2.1.4 Add unit test that creates a config with a non-None allowlist and calls `_populate_plan_metadata` to verify no `AttributeError`
  - [ ] 2.1.5 Add test verifying that URL security validation correctly rejects URLs not in the allowlist during planning

- [ ] 2.2 **Make validators concurrent with guardrails**
  - [ ] 2.2.1 Add `max_concurrent_validators: int = Field(default=2, ge=1, le=8)` to `ValidationConfig` class in `ontology_download.py`
  - [ ] 2.2.2 Import `ThreadPoolExecutor` and `as_completed` from `concurrent.futures`
  - [ ] 2.2.3 Rewrite `run_validators` function to:
    - Accept the same parameters and return the same structure
    - Create a thread pool with `max_workers=config.defaults.validation.max_concurrent_validators`
    - Define an inner worker function that calls the appropriate validator and handles exceptions
    - Submit all validation requests to the pool
    - Collect results using `as_completed` and return the same dictionary structure
  - [ ] 2.2.4 Ensure per-validator JSON artifacts are still written to `validation_dir`
  - [ ] 2.2.5 Ensure logging still occurs for each validator with appropriate `extra` fields
  - [ ] 2.2.6 Add unit tests verifying validators run concurrently up to the configured limit
  - [ ] 2.2.7 Add integration test verifying that validator results remain identical between sequential and concurrent execution

- [ ] 2.3 **Add inter-process version locking**
  - [ ] 2.3.1 Import `os` and create a `@contextmanager` function `_version_lock(ontology_id: str, version: str) -> Iterator[None]`
  - [ ] 2.3.2 Implement the lock using `msvcrt.locking` on Windows and `fcntl.flock` on Unix/Linux
  - [ ] 2.3.3 Store lock files in `CACHE_DIR / "locks"` with sanitized filenames `{safe_id}__{safe_version}.lock`
  - [ ] 2.3.4 Wrap the download operation in `fetch_one` with the version lock context manager just before calling `download_stream`
  - [ ] 2.3.5 Ensure the lock is released in all code paths (success, exception, early return)
  - [ ] 2.3.6 Add integration test simulating concurrent downloads of the same version and verifying only one completes

## 3. Performance & Scale

- [ ] 3.1 **Implement true streaming normalization**
  - [ ] 3.1.1 Create a helper function `_sort_triple_file(source: Path, destination: Path) -> None` that attempts platform `sort` command first, then falls back to Python merge-sort implementation
  - [ ] 3.1.2 Implement `_python_merge_sort` as a robust fallback using tempfiles and `heapq.merge` for memory-efficient external sorting
  - [ ] 3.1.3 Rewrite `normalize_streaming` function to:
    - Serialize the graph to an unsorted N-Triples file in a temp directory
    - Call `_sort_triple_file` to produce a sorted N-Triples file
    - Stream-read the sorted file line by line
    - Apply deterministic blank node renumbering during streaming (using `_BNODE_PATTERN` regex)
    - Write sorted `@prefix` lines first, then blank line, then canonicalized triples
    - Compute SHA-256 incrementally as lines are written
    - Never materialize the full triple list in Python memory
  - [ ] 3.1.4 Preserve the existing signature: `normalize_streaming(source, output_path=None, *, graph=None, chunk_bytes=1<<20) -> str`
  - [ ] 3.1.5 Add unit tests with various graph sizes verifying deterministic output and correct SHA-256
  - [ ] 3.1.6 Add performance test demonstrating reduced memory footprint for large graphs

- [ ] 3.2 **Add streaming threshold configuration**
  - [ ] 3.2.1 Verify `streaming_normalization_threshold_mb` field already exists in `ValidationConfig` (default 200)
  - [ ] 3.2.2 Ensure the validator logic checks file size and selects streaming vs in-memory mode accordingly
  - [ ] 3.2.3 Ensure `normalization_mode` is recorded in validation output JSON and logged
  - [ ] 3.2.4 Add unit test verifying correct mode selection based on file size threshold

## 4. Extensibility (Future-Proofing)

- [ ] 4.1 **Add resolver plugin infrastructure**
  - [ ] 4.1.1 Import `metadata` from `importlib` in `resolvers.py`
  - [ ] 4.1.2 Create function `_load_resolver_plugins(logger: Optional[logging.Logger] = None) -> None`
  - [ ] 4.1.3 Query entry points for group `docstokg.ontofetch.resolver` using `entry_points().select(group=...)`
  - [ ] 4.1.4 For each entry point, attempt to load and instantiate the resolver, catching and logging any exceptions
  - [ ] 4.1.5 Register successfully loaded resolvers in the `RESOLVERS` dict using their `NAME` attribute or entry point name
  - [ ] 4.1.6 Call `_load_resolver_plugins()` at module initialization (after `RESOLVERS` dict is defined)
  - [ ] 4.1.7 Add unit test verifying plugin loading with a mock entry point
  - [ ] 4.1.8 Add documentation example showing how to register a custom resolver via entry points

- [ ] 4.2 **Add validator plugin infrastructure**
  - [ ] 4.2.1 Import `metadata` from `importlib` in `ontology_download.py`
  - [ ] 4.2.2 Create function `_load_validator_plugins(logger: Optional[logging.Logger] = None) -> None`
  - [ ] 4.2.3 Query entry points for group `docstokg.ontofetch.validator`
  - [ ] 4.2.4 For each entry point, load the callable and register in `VALIDATORS` dict, catching exceptions
  - [ ] 4.2.5 Call `_load_validator_plugins()` at module initialization (after `VALIDATORS` dict is defined)
  - [ ] 4.2.6 Add unit test verifying plugin loading with a mock entry point
  - [ ] 4.2.7 Add documentation example showing how to register a custom validator via entry points

- [ ] 4.3 **Add manifest schema migration shim**
  - [ ] 4.3.1 Create function `_migrate_manifest_inplace(payload: dict) -> None` in `ontology_download.py`
  - [ ] 4.3.2 Implement version detection based on `payload.get("schema_version")`
  - [ ] 4.3.3 For current version "1.0", perform no-op migration (just set schema_version if missing)
  - [ ] 4.3.4 Add extensible structure for future migrations with clear version progression
  - [ ] 4.3.5 Call `_migrate_manifest_inplace(payload)` in `_read_manifest` after JSON parsing but before validation
  - [ ] 4.3.6 Add unit tests verifying migration from mock old schema versions
  - [ ] 4.3.7 Document the migration pattern for future schema evolution

## 5. Testing & Documentation

- [ ] 5.1 **Add security & correctness tests**
  - [ ] 5.1.1 Add test for URL allowlist enforcement rejecting non-allowlisted hosts
  - [ ] 5.1.2 Add test for IDN/punycode safety checks rejecting malicious Unicode domains
  - [ ] 5.1.3 Add test for HEAD request returning 405 and verifying GET fallback
  - [ ] 5.1.4 Add test for ETag-based cache hit returning status "cached"
  - [ ] 5.1.5 Add test for resume logic with `.part` files
  - [ ] 5.1.6 Add test for ZIP traversal protection rejecting paths with `..`
  - [ ] 5.1.7 Add test for compression bomb protection rejecting excessive expansion ratios

- [ ] 5.2 **Add concurrency & locking tests**
  - [ ] 5.2.1 Add test simulating multiple validators running concurrently
  - [ ] 5.2.2 Add test simulating concurrent attempts to download the same ontology version
  - [ ] 5.2.3 Add test verifying lock files are created and cleaned up correctly

- [ ] 5.3 **Add streaming normalization tests**
  - [ ] 5.3.1 Add test with small RDF graph verifying in-memory mode
  - [ ] 5.3.2 Add test with large RDF graph exceeding threshold verifying streaming mode
  - [ ] 5.3.3 Add test verifying deterministic SHA-256 output across multiple runs
  - [ ] 5.3.4 Add test verifying blank node canonicalization consistency

- [ ] 5.4 **Update CLI integration tests**
  - [ ] 5.4.1 Add test for `ontofetch plan` command with metadata enrichment
  - [ ] 5.4.2 Add test for `ontofetch plan-diff` command with baseline comparison
  - [ ] 5.4.3 Add test for `ontofetch prune --keep N` verifying latest symlink update
  - [ ] 5.4.4 Add test for `ontofetch doctor` command verifying all diagnostics

- [ ] 5.5 **Update documentation**
  - [ ] 5.5.1 Update API reference with new configuration fields
  - [ ] 5.5.2 Add migration guide for legacy import paths in MIGRATION.md
  - [ ] 5.5.3 Update CHANGELOG.md with all breaking changes and new features
  - [ ] 5.5.4 Add example for registering custom resolvers and validators via plugins
  - [ ] 5.5.5 Update operator runbook with new configuration options (max_concurrent_validators, streaming_normalization_threshold_mb)

## 6. Final Verification

- [ ] 6.1 Run full test suite with all new tests passing
- [ ] 6.2 Run linter and type checker with no new errors
- [ ] 6.3 Verify CLI help text is accurate and complete
- [ ] 6.4 Perform manual smoke test of key workflows:
  - [ ] 6.4.1 Download a small ontology (e.g., pizza.owl)
  - [ ] 6.4.2 Download a large ontology exercising streaming normalization
  - [ ] 6.4.3 Run `plan` and `plan-diff` commands
  - [ ] 6.4.4 Run `prune` command and verify latest marker
  - [ ] 6.4.5 Run `doctor` command and verify all checks pass
- [ ] 6.5 Update version number for minor release (breaking change)
- [ ] 6.6 Tag release and update CHANGELOG.md
