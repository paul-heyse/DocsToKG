## 1. Prepare module boundaries
- [ ] 1.1 Catalogue functions and classes currently in `ontology_download.py` by concern (config, io, net, validation, pipeline, plugins).
- [ ] 1.2 Produce a dependency map showing which modules consume which helpers to pre-empt circular imports.
- [ ] 1.3 Identify documentation/code references (Sphinx docs, API markdown, tests) that need import path updates once modules move.

- [ ] 2.3 Update unit tests or fixtures that import configuration models directly.

- [x] 3.3 Ensure CLI/path-based utilities and doctests reference the new module path.

- [ ] 4.3 Update network-related tests and the import-time benchmark to import through the new module.

- [x] 5.3 Verify the validator worker CLI (`python -m DocsToKG.OntologyDownload ... worker`) still dispatches correctly after module moves.

- [ ] 6.3 Update storage/manifest tests and CLI commands that rely on these orchestrators.

- [ ] 7.3 Update plugin-related tests and document the new import path in developer docs.

- [x] 8.3 Provide explicit `__all__` and re-export definitions to keep the public API guard passing.

- [x] 9.3 Refresh developer-facing documentation (API markdown / Sphinx) to reference the new module layout.
- [ ] 9.4 Regenerate typings markers (`py.typed`) or packaging metadata if required.

- [ ] 10.3 Re-run the import-time benchmark and any micro-benchmarks dependent on module-level imports.
- [ ] 10.4 Update changelog/project notes if required by contribution guidelines.
