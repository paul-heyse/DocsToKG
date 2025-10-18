## 1. Implementation
- [x] 1.1 Create new modules `src/DocsToKG/OntologyDownload/cli.py`, `formatters.py`, and `manifests.py` that encapsulate parser construction, command dispatch, table rendering, and lockfile/manifest utilities previously embedded in `api.py`.
- [x] 1.2 Update `cli_main` to delegate argument parsing and handler execution to `cli.py`, ensuring each subcommand handler resides in `manifests.py` or other dedicated modules without re-importing `api.py`.
- [x] 1.3 Introduce `src/DocsToKG/OntologyDownload/io/network.py`, `io/rate_limit.py`, and `io/filesystem.py`, moving the corresponding logic out of `io.py`, and leave `io/__init__.py` exporting the composed `download_stream` and related helpers.
- [x] 1.4 Add `src/DocsToKG/OntologyDownload/exports.py` capturing the public export manifest and have both `api.py` and `__init__.py` derive `PUBLIC_API_MANIFEST`/`__all__`/`__getattr__` wiring from that manifest.
- [x] 1.5 Replace `_VALIDATOR_PLUGINS_LOADED` in `validation.py` with a loader that uses a module-level `threading.RLock`, defers to `plugins.ensure_plugins_loaded(reload=...)`, and keeps the registry consistent across reloads.
- [x] 1.6 Implement a cached accessor such as `settings.get_default_config()` backed by `ResolvedConfig.from_defaults()`, add `invalidate_default_config_cache()`, and move all callers (API, planning, IO, validation) to the new helper.
- [x] 1.7 Adjust imports, typing exports, and unit/integration tests to respect the new module boundaries (CLI smoke tests, download pipeline tests, validator plugin tests).
- [x] 1.8 Update developer documentation or docstrings referencing the old monolithic modules to point at the new structure.

## 2. Validation
- [x] 2.1 Run targeted unit tests covering CLI, IO, and validation modules plus `direnv exec . openspec validate refactor-ontology-download-architecture --strict`.
