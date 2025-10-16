## ADDED Requirements
### Requirement: Modular Ontology Downloader Core
The ontology downloader core MUST organise code into dedicated modules for configuration, IO safety, networking, validation, orchestration, and plugin discovery while preserving the existing public API surface.

#### Scenario: Core modules are partitioned by concern
- **GIVEN** the repository layout under `DocsToKG/OntologyDownload`
- **WHEN** a developer inspects the package
- **THEN** they find `config.py`, `io_safe.py`, `net.py`, `validation_core.py`, `pipeline.py`, and `plugins.py`
- **AND** each module contains only the responsibilities outlined in the change proposal.

#### Scenario: Public API remains stable
- **GIVEN** an integration importing `DocsToKG.OntologyDownload`
- **WHEN** the package is imported after the refactor
- **THEN** every symbol previously exposed via `DocsToKG.OntologyDownload.__all__` resolves without modification.

#### Scenario: CLI worker entrypoint is intact
- **GIVEN** an operator runs `python -m DocsToKG.OntologyDownload worker pronto`
- **WHEN** the subprocess module dispatches to validator workers
- **THEN** the refactored module layout still invokes the correct handler without dependency errors.

#### Scenario: Developer documentation reflects new module layout
- **GIVEN** the API reference and developer guides under `docs/`
- **WHEN** the refactor is complete
- **THEN** documentation references the new module paths (`config.py`, `io_safe.py`, etc.) without broken links.
