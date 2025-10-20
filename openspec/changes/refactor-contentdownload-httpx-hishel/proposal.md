## Why
- The ContentDownload stack still relies on bespoke `requests.Session` factories, manual adapter tuning, and ad-hoc conditional request handling, making it difficult to standardise timeout phase control, HTTP/2, TLS configuration, caching semantics, and telemetry across resolvers, head pre-checks, and streaming downloads.
- Without a shared transport cache we re-fetch identical resources for conditional probes, robots lookups, and retries; consolidating on HTTPX with a Hishel RFC-9111 cache lets us delete duplicate ETag/Last-Modified plumbing and minimise redundant HEAD/GET traffic.
- Aligning ContentDownload with the HTTPX+Hishel strategy already planned for OntologyDownload unlocks a single mocking surface (`httpx.MockTransport`) for tests and paves the way for future async adoption while keeping Tenacity-based retry semantics intact.

## What Changes
- Introduce `DocsToKG.ContentDownload.httpx_transport.get_http_client()` (singleton `httpx.Client` with HTTP/2, `httpx.Limits(max_connections=128, max_keepalive_connections=32, keepalive_expiry=15.0)`, `httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)`, Certifi-backed SSL context, optional proxy mounts) plus `configure_http_client(...)`/`reset_http_client_for_tests()` so production code and tests share one transport surface.
- Wrap the client in a Hishel disk cache rooted at `${CACHE_DIR}/http/ContentDownload`, exposing `purge_http_cache()` for ops/tests, and install HTTPX hooks that stamp telemetry fields, enforce `response.raise_for_status()`, and measure retry/sleep metrics.
- Replace `ThreadLocalSessionFactory`, `create_session`, and direct `requests` usage in `DocsToKG.ContentDownload.networking` with HTTPX-based helpers while adapting the Tenacity retry controller to operate on `httpx.Response` objects and `httpx` exception classes, preserving existing retry-after caps and final-response semantics.
- Migrate all ContentDownload call sites—resolvers, head pre-check, robots cache, download pipeline, streaming downloads—to consume the shared HTTPX client and ensure cached 304 responses populate existing `CachedResult` / `ModifiedResult` objects with unchanged telemetry.
- Update configuration plumbing, docs, and tests to recognise the new transport (e.g., dependency pins, CLI options, unit tests that patch `httpx.MockTransport`, removal of session-leasing fixtures), while keeping public API signatures and telemetry payloads stable.

## Impact
- **Affected specs:** content-download
- **Affected code:** `src/DocsToKG/ContentDownload/networking.py`, new `src/DocsToKG/ContentDownload/httpx_transport.py` (or equivalent), `download.py`, `pipeline.py`, resolver modules, `tests/content_download/**`, `requirements*.txt`, `docs/` surfaces describing networking, caching, and troubleshooting.
