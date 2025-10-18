## Context
- `DocsToKG.OntologyDownload.api` currently mixes API exports, CLI parsing, formatting, and manifest helpers in a single 2.7k-line file. This makes incremental changes risky and obscures the boundary between library callers and CLI tooling.
- `DocsToKG.OntologyDownload.io` bundles filesystem utilities, archive extraction, DNS caching, retry logic, streaming downloads, rate limiting, and session pooling. Unit tests for networking behaviours have to load filesystem helpers (and vice versa), slowing imports and complicating future refactors.
- Public exports are duplicated manually in `api.py` and `__init__.py`, creating drift risk as symbols change.
- `validation.load_validator_plugins` keeps its own `_VALIDATOR_PLUGINS_LOADED` flag without synchronisation while the plugins module already maintains its own locking. Concurrent invocations can race, and reload semantics differ from `plugins.ensure_plugins_loaded`.
- Many call sites repeatedly invoke `ResolvedConfig.from_defaults()` (`api`, `planning`, `io`), re-reading configuration defaults on hot paths and pulling environment overrides each time.

## Goals / Non-Goals
- Goals:
  - Define explicit module boundaries so CLI-only code (parser construction, command handlers, formatting) resides outside the API facade.
  - Split IO concerns so networking/session/rate-limit logic is independent from filesystem/extraction helpers.
  - Provide a single manifest that drives `PUBLIC_API_MANIFEST`, `__all__`, and lazy attribute exports.
  - Make validator plugin loading thread-safe and reload-aware by delegating to the canonical plugin registry.
  - Add a cached accessor for default configuration resolution with explicit invalidation hooks for CLI overrides.
- Non-Goals:
  - Changing resolver behaviour or introducing new CLI commands.
  - Rewriting existing logging/telemetry formats.
  - Altering plugin discovery entry points beyond load-time synchronisation.

## Decisions
1. **Module boundaries**
   - Keep `api.py` focused on library-facing functions (`plan_*`, `fetch_*`, validators, download helpers, metadata).
   - Introduce new modules:
     ```
     DocsToKG/OntologyDownload/
       cli.py          # build_parser(), dispatch(), command handler registry
       formatters.py   # format_table(), format_plan_rows(), format_results_table(), validation formatter
       manifests.py    # lockfile read/write, manifest parsing helpers
       exports.py      # PUBLIC_EXPORTS definition driving PUBLIC_API_MANIFEST/__all__
       io/__init__.py  # download_stream(), extract_archive_safe() re-exporting specialised helpers
       io/network.py   # StreamingDownloader, HTTP headers, retry logic, session pool
       io/rate_limit.py# TokenBucket, SharedTokenBucket, RateLimiterRegistry
       io/filesystem.py# sanitize_filename(), extract_zip_safe(), extract_tar_safe(), checksum helpers
     ```
   - `cli_main` becomes a thin wrapper that forwards to `cli.dispatch(argv, default_config=get_default_config())`.
   - Manifest/lockfile helpers move into `manifests.py`, which is imported by CLI handlers and API functions needing lockfile utilities.

2. **Export manifest**
   - `exports.py` defines a constant structure, e.g.:
     ```python
     PUBLIC_EXPORTS = {
         "FetchSpec": ("DocsToKG.OntologyDownload.planning", "class"),
         ...
     }
     ```
   - `api.py` builds `PUBLIC_API_MANIFEST` and `__all__` by iterating over `PUBLIC_EXPORTS`. `__getattr__` resolves attributes through the manifest, ensuring a single source of truth.

3. **Validator plugin loader**
   - Replace `_VALIDATOR_PLUGINS_LOADED` with `_VALIDATOR_LOAD_LOCK = threading.RLock()` and `_VALIDATOR_CACHE: Optional[MutableMapping[str, ValidatorPlugin]]`.
   - `load_validator_plugins` acquires the lock, calls `plugins.ensure_plugins_loaded(reload=reload)`, updates the caller-provided registry if needed, and caches the registry reference for subsequent calls.

4. **Default configuration caching**
   - Add `get_default_config()` and `invalidate_default_config_cache()` in `settings.py`.
   - Cache stored in a module-level variable protected by `threading.Lock()`.
   - CLI overrides (`--log-level`, allowed hosts, etc.) call the invalidation hook before obtaining defaults.
   - Planning, IO, validation modules import the helper instead of calling `ResolvedConfig.from_defaults()` directly.

5. **Testing requirements**
   - Add/import tests that create the new modules in isolation (e.g., `tests/ontology_download/test_cli.py`, `test_io_network.py`).
   - Ensure CLI smoke tests run via `direnv exec . python -m DocsToKG.OntologyDownload.cli --help` or equivalent.

## Risks / Trade-offs
- Splitting modules increases the number of imports; we mitigate by keeping re-exporting `io/__init__.py` and manifest helper functions.
- Introducing caching for default configurations risks staleness; the invalidation hook is mandatory whenever overrides change, and tests must cover this path.
- Plugin loader changes must mirror existing behaviour; difference in timing could mask plugin failures, so logs should remain unchanged.

## Migration Plan
1. Create new modules with extracted functions using existing tests as guard rails.
2. Update imports across the codebase to use the new module paths while keeping reference exports in `DocsToKG.OntologyDownload`.
3. Implement the export manifest and verify `dir()`/lazy attribute behaviour.
4. Update validator loader and ensure concurrency tests (or synthetic tests) simulate parallel loads and reloads.
5. Replace direct calls to `ResolvedConfig.from_defaults()` and add integration tests verifying cache invalidation.
6. Run unit tests, CLI smoke tests, and `openspec validate` before requesting approval.

## Open Questions
- Should we expose additional helper APIs (e.g., `cli.build_parser`) as public exports or keep them internal?
- Do we need feature flags to toggle the cached configuration behaviour for long-running processes that watch environment variables?
