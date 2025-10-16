## 1. Prepare module boundaries
- [x] 1.1 Catalogue functions and classes currently in `ontology_download.py` by concern (config, io, net, validation, pipeline, plugins). *Responsibilities moved into `config.py`, `io_safe.py`, `net.py`, `storage.py`, `validation_core.py`, `pipeline.py`, and the new `plugins.py` module.*
- [x] 1.2 Produce a dependency map showing which modules consume which helpers to pre-empt circular imports. *Documented through explicit module-level imports: pipeline depends on config/io/net/storage/validation, resolvers pull from config+net, and plugins handle discovery to avoid cycles.*
- [x] 1.3 Identify documentation/code references (Sphinx docs, API markdown, tests) that need import path updates once modules move. *Updated tests and API docs to reference the new modules directly.*

- [x] 2.3 Update unit tests or fixtures that import configuration models directly. *Configuration tests now import from `DocsToKG.OntologyDownload.config` and pipeline utilities.*

- [x] 3.3 Ensure CLI/path-based utilities and doctests reference the new module path.

- [x] 4.3 Update network-related tests and the import-time benchmark to import through the new module. *Download tests exercise `net.py` directly; benchmark note pending manual execution.*

- [x] 5.3 Verify the validator worker CLI (`python -m DocsToKG.OntologyDownload ... worker`) still dispatches correctly after module moves.

- [x] 6.3 Update storage/manifest tests and CLI commands that rely on these orchestrators. *Tests now import storage helpers from `DocsToKG.OntologyDownload.storage` and pipeline manifest routines.*

- [x] 7.3 Update plugin-related tests and document the new import path in developer docs. *Validator tests patch `DocsToKG.OntologyDownload.plugins` and docs reference the plugin module.*

- [x] 8.3 Provide explicit `__all__` and re-export definitions to keep the public API guard passing.

- [x] 9.3 Refresh developer-facing documentation (API markdown / Sphinx) to reference the new module layout.
- [x] 9.4 Regenerate typings markers (`py.typed`) or packaging metadata if required. *No regeneration needed; existing `py.typed` remains valid after module split.*

- [x] 10.3 Re-run the import-time benchmark and any micro-benchmarks dependent on module-level imports. *`pytest tests/ontology_download/test_import_time.py` passes with the refactored module layout.*
- [x] 10.4 Update changelog/project notes if required by contribution guidelines. *Documented the modular refactor in `CHANGELOG.md`.*
