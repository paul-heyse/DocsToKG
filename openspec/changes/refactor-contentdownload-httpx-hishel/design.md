## Context
ContentDownload still depends on bespoke `requests` session factories (`ThreadLocalSessionFactory`, `create_session`) and manual adapter configuration. Conditional requests, cache revalidation, and streaming are implemented with ad-hoc helpers plus `pooch`-style downloader logic. This mirrors the pre-refactor state of OntologyDownload, whereas our new HTTP networking direction (per the `HTTPX+Hishel_Transition_Plan`) is to consolidate on a single HTTPX client with a Hishel cache. Aligning ContentDownload with that approach removes duplicated transport code, unlocks shared test infrastructure (`httpx.MockTransport`), and keeps Tenacity retry semantics consistent across products.

## Goals / Non-Goals
- **Goals**
  - Provide a singleton HTTPX client (HTTP/2 enabled) with explicit timeouts, pool limits, SSL context, and proxy support for all ContentDownload networking.
  - Layer a Hishel RFC-9111 cache under the client to handle conditional GETs/HEADs automatically and store entries under `CACHE_DIR/http/`.
  - Adapt `DocsToKG.ContentDownload.networking` so Tenacity retries operate on `httpx.Response` objects while preserving existing public APIs and telemetry.
  - Migrate streaming downloads, resolvers, head pre-checks, and robots cache to the shared HTTPX transport without breaking downstream interfaces.
  - Update tests and documentation to rely on HTTPX/Hishel primitives (e.g., `MockTransport`) instead of patching `requests`.
- **Non-Goals**
  - Introducing async ContentDownload flows (we stay sync for now, though HTTPX prepares us for async later).
  - Rewriting Tenacity retry policies beyond the minimum adjustments required to work with HTTPX responses/exceptions.
  - Changing public CLI flags or manifest output formats beyond documenting the new transport.

## Decisions
- **Transport library:** Adopt `httpx` (sync client) with `http2=True`, configured via `httpx.Client(http2=True, limits=httpx.Limits(max_connections=128, max_keepalive_connections=32, keepalive_expiry=15.0), timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0))`. This mirrors the OntologyDownload refactor and gives us deterministic timeout semantics.
- **Caching:** Use `hishel.CacheControlTransport` around the shared client, persisting entries below `${CACHE_DIR}/http/ContentDownload`. Cache keys will use the already-normalised URL canonicaliser to avoid duplicate variants. Provide a public `purge_http_cache()` helper for ops/tests.
- **Configuration surface:** Expose `get_http_client()`, `configure_http_client(proxy_mounts=None, transport=None, event_hooks=None)`, and `reset_http_client_for_tests()` from `httpx_transport.py`, enabling dependency injection (e.g., `httpx.MockTransport`) without touching module-level globals directly.
- **Tenacity integration:** Retain the Tenacity retry controller but swap the predicates/results to operate on `httpx.Response` plus `httpx.TimeoutException`, `httpx.TransportError`, and `httpx.ProtocolError`. Tenacity will continue to honour retryable status codes and `Retry-After` headers while closing intermediate responses via HTTPX.
- **Streaming:** Replace `requests` streaming with `httpx.Client.stream("GET", …)`, writing to the existing temp-path strategy and atomically promoting files on success. Range/resume guards remain disabled unless reintroduced later.
- **Telemetry:** Attach HTTPX request/response hooks that add structured log extras (method, host, cache outcome, attempt number), enforce `response.raise_for_status()` once per successful try, and measure per-attempt latency for telemetry parity.

## Alternatives
- **Stay on `requests`:** Rejected; we would continue to maintain custom pooling, caching, and header handling, and the existing HTTPX transition plan would be inconsistent between OntologyDownload and ContentDownload.
- **Adopt `aiohttp` instead of HTTPX:** Rejected; ContentDownload is currently synchronous and HTTPX gives us both sync and async APIs, plus a first-party Hishel integration.
- **Custom cache layer:** Rejected; Hishel already implements RFC-9111 semantics and matches the plan used elsewhere, reducing maintenance.

## Risks / Trade-offs
- **Dependency footprint:** Adding HTTPX + Hishel increases dependencies; mitigated by pinning versions and documenting zero-install guard rails.
- **Behavioural drift:** Switching transports may expose subtle header/order differences; mitigated by preserving Tenacity wrappers, maintaining telemetry fields, and expanding tests to cover cache hits/misses and streaming semantics.
- **Cache storage size:** Disk cache introduces new storage usage; we will expose configuration/clean-up guidance and ensure cache keys respect existing URL canonicalisation.

## Migration Plan
1. Introduce the HTTPX client module with configurable singleton, Hishel caching, and hooks; wire up dependency pins.
2. Refactor `networking.py` to remove session factories and operate purely on HTTPX responses within Tenacity.
3. Update streaming helpers, resolvers, and robots cache to use the new transport; ensure 304 handling returns `CachedResult`.
4. Replace `requests` mocks in tests with HTTPX `MockTransport` fixtures; update patch points for Tenacity sleep/wait overrides.
5. Refresh documentation (README, AGENTS, troubleshooting) to describe the new transport and cache, including instructions for purging caches.

## Open Questions
- Should we allow operators to choose between in-memory vs disk Hishel stores for constrained environments?
- Do we need a migration script to clear legacy cache directories when upgrading?
- Are there downstream consumers relying on `create_session` that require a deprecated shim or formal removal notice?
