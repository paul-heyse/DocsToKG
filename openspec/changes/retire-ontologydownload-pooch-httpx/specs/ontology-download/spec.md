# Ontology Download Specification Deltas

## ADDED Requirements

### Requirement: HTTPX Streaming Downloader
`DocsToKG.OntologyDownload.io.download_stream` SHALL stream ontology payloads directly from the shared HTTPX client (no intermediate pooch layer) while enforcing existing policy and manifest semantics.

#### Scenario: Single GET with atomic write
- **WHEN** `download_stream` executes
- **THEN** it SHALL issue a single `GET` request through the shared HTTPX client (falling back to conditional cache revalidation when Hishel provides a validator)
- **AND** it SHALL write chunks to a temporary file, enforce `max_uncompressed_size_gb` by counting bytes as they arrive, and atomically move the completed file into place.

#### Scenario: Progress and telemetry preserved
- **WHEN** bytes are streamed
- **THEN** the downloader SHALL emit progress logs/telemetry at configurable intervals (e.g., every 10% or N MB) including URL, bytes downloaded, and total size when known
- **AND** the existing structured logging schema (`stage="download"`, `status`, `elapsed_ms`) SHALL be retained.

#### Scenario: Manifest metadata sourced from HTTPX response
- **WHEN** the GET completes
- **THEN** `DownloadResult` SHALL populate `etag`, `last_modified`, `content_type`, and `content_length` from the HTTPX response headers (or the cached metadata on 304)
- **AND** checksum enforcement (`expected_hash`) SHALL operate on the streamed file without re-downloading.

#### Scenario: Cached response short-circuits
- **WHEN** Hishel returns a 304 revalidation
- **THEN** `download_stream` SHALL treat the artifact as cached, avoid writing the body, compute/verify hash via the cached file, and return `DownloadResult.status == "cached"`.

### Requirement: Planner and Checksum Probes Share HTTPX Client
Planner metadata probes and checksum fetch utilities SHALL rely on the shared HTTPX client without issuing redundant HEAD requests, while preserving rate limits and security validation.

#### Scenario: Planner uses single GET by default
- **WHEN** `planner_http_probe` fetches metadata
- **THEN** it SHALL call `get_http_client()` once, issue a single GET (unless an allow-listed host explicitly requires HEAD), and reuse the response to populate plan metadata
- **AND** it SHALL honour rate-limit buckets, polite headers, and cache validators via Hishel.

#### Scenario: Checksum fetch streams via HTTPX
- **WHEN** `_fetch_checksum_from_url` downloads checksum manifests
- **THEN** it SHALL stream using `httpx.Client.stream`, enforce byte ceilings during iteration, and classify retryable errors based on HTTPX exceptions.

### Requirement: Legacy Networking Surfaces Removed
OntologyDownload SHALL remove `pooch` and `requests.Session` usage from production code, keeping public exports stable while forwarding to HTTPX equivalents.

#### Scenario: No pooch dependency
- **WHEN** OntologyDownload modules are imported
- **THEN** they SHALL not import or reference `pooch` (aside from optional compatibility shims guarded for downstream consumers)
- **AND** project dependency metadata (`pyproject.toml`, `optdeps.py`) SHALL drop `pooch` for OntologyDownload.

#### Scenario: SessionPool removed from public API
- **WHEN** `DocsToKG.OntologyDownload.io` is inspected
- **THEN** `SessionPool` and `SESSION_POOL` SHALL no longer appear in `__all__`
- **AND** existing exports (`download_stream`, `DownloadResult`, `validate_url_security`) SHALL keep their signatures while delegating to the new implementation.

### Requirement: Tests and Documentation Align with HTTPX Semantics
Test harnesses and documentation SHALL reflect the HTTPX+Hishel downloader, ensuring deterministic fixtures and updated operator guidance.

#### Scenario: Tests rely on httpx.MockTransport
- **WHEN** ontology downloader tests execute
- **THEN** they SHALL use `httpx.MockTransport` (or equivalent in-memory transports) to simulate responses, including redirects, 304s, and failure cases, without monkeypatching `requests`.

#### Scenario: CLI output and docs remain stable
- **WHEN** CLI regression tests run
- **THEN** table/JSON outputs SHALL continue to expose `content_type`, `content_length`, `etag`, and `status`
- **AND** README / AGENTS / LibraryDocumentation pages SHALL describe the HTTPX streaming path, cache directory layout, and removal of pooch/SessionPool.

## REMOVED Requirements

### Requirement: Pooch-based streaming downloader
The system SHALL no longer rely on `pooch.HTTPDownloader` or `SessionPool.lease` for ontology HTTP operations.

#### Scenario: Legacy downloader classes discarded
- **WHEN** ontology download modules are inspected
- **THEN** classes/functions tied to the pooch downloader SHALL be removed or clearly marked deprecated with no remaining call sites.
