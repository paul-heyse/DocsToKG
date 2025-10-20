# Adopt requests-cache for ContentDownload Metadata Requests

## Why
- Metadata resolvers and pipeline helpers perform repeated HTTP GET/HEAD calls against the same providers, burning rate limits and time despite heavy overlap between runs. The current code relies on ad-hoc guards (recent-fetch maps, manual 304 handling, bespoke cache directories) that are scattered, inconsistent, and difficult to maintain.
- `requests-cache` provides a drop-in `CachedSession` with persistent backends, TTL controls, conditional requests, and stale-on-error behavior. Migrating to it allows us to delete custom cache logic while gaining reliable retry-aware caching, per-host policies, and better observability.
- Consolidating cache policy in `networking.py` ensures every resolver, pipeline probe, and head-precheck uses the same cache semantics, preventing drift and simplifying future tuning (backends, TTLs, offline mode).

## What Changes
- Replace the plain `requests.Session` factory in `DocsToKG.ContentDownload.networking` with a configurable `requests_cache.CachedSession`, including default SQLite backend, per-host TTL map, and filters that exclude binary/large responses.
- Thread cache configuration knobs through CLI/resolver config (e.g., cache path, backend selection, global TTL, offline mode, `stale_if_error`, `stale_while_revalidate`) so operators can tune behavior without touching code.
- Update all metadata HTTP callers (resolvers, pipeline, head precheck, robots fetches, landing page probes) to use the cached session; delete redundant dedupe/caching utilities and conditional request handling.
- Add telemetry for cache hits/misses (`from_cache`, age, stale served) to manifest summaries/logs, enabling monitoring of cache effectiveness and stale responses.
- Refresh tests and documentation to reflect the requests-cache integration, including fixtures that simulate cache hits, eviction, and stale handling.

## Impact
- **Affected specs:** content-download
- **Affected code:** `src/DocsToKG/ContentDownload/{networking.py, pipeline.py, download.py, resolvers/**, AGENTS.md, README.md, LibraryDocumentation/requests-cache*.md}`, configuration plumbing (`args.py`, `pyalex_shim.py`, resolver config loaders), telemetry summaries, tests under `tests/content_download/**` and `tests/resolvers/**`.
