## Why
- The current OntologyDownload networking stack centralises connection reuse in a bespoke `SessionPool` built on `requests`, which prevents us from standardising timeout phases, HTTP/2, SSL context management, and telemetry hooks across planner probes, checksum fetchers, and the streaming downloader.
- Conditional request behaviour is duplicated: bespoke ETag logic lives in the downloader while planner/checksum probes bypass caching entirely. Without an RFC-9111 layer we cannot safely share cache state or reduce redundant HEAD/GET traffic.
- Upcoming refactors (streaming without `pooch`, retry policy simplification) depend on having a single, injectable HTTPX client that tests can swap with `MockTransport`. We need to land that client and remove the legacy session pool before touching higher layers.

## What Changes
- Introduce `DocsToKG.OntologyDownload.net` exposing a singleton `httpx.Client` wrapped by a Hishel disk cache (`CACHE_DIR/http/`), with explicit timeout/pool limits, HTTP/2 enabled, SSL context sourced from Certifi, and event hooks that stamp telemetry headers and call `raise_for_status()`.
- Provide `get_http_client()` / `configure_http_client()` helpers so tests (and `DownloadConfiguration.set_session_factory`) can inject transports or substitute clients while production code uses the shared instance.
- Extend `DownloadConfiguration` with HTTPX-specific knobs (connect/pool timeouts, HTTP/2 toggle, pool limits) and route its factory plumbing through the new helper APIs.
- Update all call sites that currently lease `requests` sessions—planner HTTP probes, checksum URL fetchers, `StreamingDownloader`, and any retry helpers—to consume the HTTPX client helpers instead of `SESSION_POOL.lease`, keeping rate-limit/token-bucket orchestration intact.
- Remove the `SessionPool` class, its exports, and any direct `requests.Session` plumbing from `io/__init__.py`, configuration helpers, and tests; replace them with HTTPX-centric shims that preserve public API surface (`download_stream`, `validate_url_security`, RDF alias constants).
- Refresh documentation and fixtures so OntologyDownload agents know the authoritative transport stack is HTTPX+Hishel, and ensure tests rely on `httpx.MockTransport` rather than monkeypatching `requests`.

## Impact
- **Affected specs:** ontology-download
- **Affected code:** `src/DocsToKG/OntologyDownload/io/network.py`, new `src/DocsToKG/OntologyDownload/net.py`, `io/__init__.py`, `planning.py`, `checksums.py`, `settings.py` (session factory hooks), `testing/__init__.py`, downloader/planner README & AGENTS guides, test fixtures under `tests/ontology_download/**`.
