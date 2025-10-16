# Changelog

All notable changes to DocsToKG are documented in this file.

## [Unreleased]

_No unreleased changes._

## [0.2.0] - 2025-02-15

### Breaking Changes
- Removed the legacy ``DocsToKG.OntologyDownload.(core|config|download|storage|validators|utils|...)`` module aliases in favour of the public ``DocsToKG.OntologyDownload`` facade and direct ``.ontology_download`` / ``.cli`` imports. Update imports before upgrading.
- Planner metadata enrichment now enforces download host allowlists during planning; configurations that previously logged warnings will now raise ``ConfigError`` when a resolver returns a disallowed host.

### Added
- Regression coverage ensuring validator results are identical between sequential
  and concurrent execution and that streaming normalization flushes chunked
  output for large ontologies.
- Integration test for the ``ontofetch doctor`` command covering filesystem,
  dependency, network, and manifest diagnostics.
- Documentation examples showing how to register custom resolver and validator
  plugins via ``docstokg.ontofetch`` entry points.

### Changed
- Validator configuration docs now highlight the ``max_concurrent_validators``
  limit and the operations runbook covers tuning streaming thresholds and
  validator concurrency for large datasets.
- Migration guide elaborates on removing legacy module aliases and explains how
  to apply ``_migrate_manifest_inplace`` for future schema upgrades.

### Deprecated
- Convenience re-exports of ``time`` and ``requests`` from
  ``DocsToKG.ContentDownload.resolvers`` emit deprecation warnings. Removal is
  scheduled for the 2025.12 minor release (see
  `openspec/changes/enhance-content-download-reliability/notes/deprecation-removal-plan.md`).

### Fixed
- Tests cover HEAD pre-check redirects, resolver concurrency error isolation, and configuration validation edge cases.
- Planning metadata enrichment no longer passes bare allowlists to ``validate_url_security`` and correctly rejects URLs outside the configured host policy without raising attribute errors.
- Property-based tests ensure conditional request headers and deduplication utilities behave correctly across arbitrary inputs.
