## Why
- Our downloader still subclasses `pooch.HTTPDownloader`, forcing double copies (cache → destination), extra HEAD preflights, and bespoke progress callbacks that will disappear once we rely on HTTPX + Hishel. To complete the refactor we must stream directly from HTTPX, preserving size guards, manifest fields, and resume semantics without pooch glue.
- Planner probes and checksum helpers currently perform redundant HEAD calls and maintain handcrafted conditional logic. With the shared HTTPX client in place, we need to consolidate probes onto a single GET+cache flow so metadata and validation stay in sync.
- Legacy surface area (`SessionPool`, `StreamingDownloader`, `pooch`, `requests`) and test fixtures still assume the old transport. Removing these stragglers—and updating docs/tests—ensures future agents work against the canonical HTTPX/Hishel stack without fallback pathways.

## What Changes
- Reimplement `download_stream` and supporting helpers to stream HTTPX responses directly to disk (single GET, chunked writes, manifest reconciliation) while keeping progress telemetry, byte caps, content-type validation, conditional requests, and hash verification intact.
- Port planner metadata probes, checksum fetchers, and any remaining HTTP helpers to use the shared HTTPX client (no `requests` sessions, no redundant HEAD). Cache-aware probes should rely on Hishel for validators and preserve existing metadata fields.
- Remove the `StreamingDownloader` subclass, `pooch` dependency, and `SESSION_POOL` shims from OntologyDownload, keeping public exports stable by forwarding through the new HTTPX implementation.
- Refresh the test harness to use `httpx.MockTransport`, add coverage for 304/cached flows, size guard failures, redirect auditing, and CLI output stability, and update documentation (README, AGENTS, library briefs) to describe the new streaming path.

## Impact
- **Affected specs:** ontology-download
- **Affected code:** `src/DocsToKG/OntologyDownload/io/network.py`, `io/__init__.py`, `net.py` helpers, `planning.py`, `checksums.py`, `settings.py`, dependency manifests (`pyproject.toml`, `optdeps.py`), CLI/tests under `tests/ontology_download/**`, documentation in `OntologyDownload/README.md`, `AGENTS.md`, and `LibraryDocumentation/*`.
