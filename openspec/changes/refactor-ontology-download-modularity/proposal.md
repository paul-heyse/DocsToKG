## Why
- `DocsToKG.OntologyDownload.ontology_download` has grown past 7k LOC, making targeted maintenance (e.g., optdeps/storage retirement, facade hardening) slow and risky.
- Upcoming work introduces new modules and optional facades; without modular seams it's difficult to iterate collaboratively.
- Splitting configuration, IO safety, networking, validation, pipeline, and plugin discovery into dedicated modules reduces merge conflicts and speeds up future contributions.

## What Changes
- Create lightweight modules under `DocsToKG.OntologyDownload`:
  - `config.py` for pydantic models, env overrides, and YAML loading helpers.
  - `io_safe.py` for URL/filename sanitisation, archive extraction, and hashing utilities.
  - `net.py` for retry helpers, token bucket, and streaming downloader logic.
  - `validation_core.py` for validation DTOs, canonicalisation, worker subprocess glue, and validator runners.
  - `pipeline.py` for fetch planning/execution, manifests, and storage coordination.
  - `plugins.py` for resolver/validator entry-point discovery and registration.
- Reduce `ontology_download.py` to an orchestrator that re-exports split functionality and hosts minimal glue.
- Ensure the public facade (`DocsToKG.OntologyDownload.__init__`) and optdeps/storage shims continue to surface the same API without code changes for consumers.
- Update unit tests, CLI entry points, and developer docs (module references in Sphinx/API markdown) to reflect the new structure without breaking existing documentation links.
- Adjust import-time benchmark and public API guard tests, if needed, to import through the refactored module layout.

## Impact
- No behavioural changes: public API and CLI stay identical, enabling drop-in replacement.
- Refactor unblocks future workstreams (validation splitting, storage changes) by letting different files be owned in parallel.
- Developers gain faster navigation and clearer ownership, improving testability and review velocity.
