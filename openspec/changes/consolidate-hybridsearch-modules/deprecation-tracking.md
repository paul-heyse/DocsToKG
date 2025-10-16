# HybridSearch Module Deprecation Plan

- **v0.5.x (current)** – Shims for `operations`, `results`, `similarity`, `retrieval`,
  `schema`, and `tools` emit `DeprecationWarning` on import. Documentation and tests point to
  the consolidated modules.
- **v0.5.x + 2 weeks** – Announce in release notes and project channels; remind integrators to
  migrate using `docs/hybrid_search_module_migration.md`.
- **v0.6.0** – Remove shim modules and associated warning tests. CI workflows and packaging
  metadata will enforce the new module layout exclusively.
- **v0.6.0 release notes** – Highlight removal of legacy modules and link to the migration
  guide for historical reference.
