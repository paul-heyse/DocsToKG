## 1. Introduce Cached Session & Policy in `networking.py`
- [ ] 1.1 Add `requests_cache` dependency imports (`CachedSession`, `ExpirationTime`, `NEVER_EXPIRE`, `DO_NOT_CACHE`, etc.) and define module constants for cache file name, default TTL, ignore-parameter list, and cacheable hosts.
- [ ] 1.2 Implement a `_build_cache_settings(config: Optional[ResolverConfig]) -> CacheSettings` helper that constructs backend path, expire defaults, per-host `urls_expire_after`, `stale_if_error`, `stale_while_revalidate`, allowable methods/codes, and filter functions (reject binaries, >5MB responses).
- [ ] 1.3 Create `create_cached_session(**overrides)` that memoizes a single `CachedSession` per process/thread factory, applying the settings above, attaching hooks for logging cache hits, and exposing the session through `ThreadLocalSessionFactory`.
- [ ] 1.4 Ensure the session honours HTTP validators by setting `cache_control=True`, `always_revalidate=True`, and installs a request hook that strips ignored query parameters before cache key generation.
- [ ] 1.5 Add support for “offline mode” (environment variable or config flag) that sets `only_if_cached=True` on all outgoing requests and surface a convenience context manager to temporarily disable caching for operations that must hit origin.
- [ ] 1.6 Update NAVMAP, docstring, and `__all__` exports to reflect the new cache factory and policy helpers.

## 2. Configuration & CLI Plumbing
- [ ] 2.1 Extend CLI arguments/config (`args.py`, resolver config structures) to expose cache settings: enable/disable flag, cache directory/path, global TTL, backend choice (`sqlite`, `redis`, `memory`), offline toggle, and stale durations.
- [ ] 2.2 Update `ResolverConfig` and `DownloadOptions` dataclasses to carry cache preferences, including per-host TTL overrides and a list of additional ignored query parameters supplied by operators.
- [ ] 2.3 Ensure `pyalex_shim.ConfigProxy` and any other configuration wrappers surface defaults for cache settings and validate user-supplied values (positive TTLs, known backends, etc.).
- [ ] 2.4 Add serialization/deserialization for cache config in manifest summaries or run metadata so downstream reporting captures the active cache policy.

## 3. Apply Cached Session to Callers & Remove Legacy Logic
- [ ] 3.1 Replace all `requests.Session` creations in `networking.create_session` and `ThreadLocalSessionFactory` with the cached session; ensure Tenacity wrappers continue to receive the same session object.
- [ ] 3.2 Update resolvers (`crossref`, `semantic_scholar`, `unpaywall`, `openalex`, `doaj`, `zenodo`, `figshare`, `osf`, `pmc`, `europe_pmc`, `hal`, `landing_page`, `wayback`, `core`, etc.) to import the cached session factory; remove custom recent-fetch maps, manual 304 handling, or per-resolver TTL comments.
- [ ] 3.3 Modify pipeline helpers (`head_precheck`, `RobotsCache`, landing page probes) to use the cached session and drop any ad-hoc caching/conditional request code.
- [ ] 3.4 Remove redundant helpers or data structures that existed solely for caching (e.g., in-memory URL caches, persisted metadata freshness tables) while preserving business-level dedupe.
- [ ] 3.5 Ensure binary download paths (`download.stream_candidate_payload`) explicitly disable caching (e.g., using `requests_cache.disabled()` or request-level `expire_after=DO_NOT_CACHE`) when streaming PDFs to avoid accidental cache writes.

## 4. Telemetry, Logging & Observability
- [ ] 4.1 Instrument responses to record `from_cache`, `created_at`, `expires`, and `cache_key` (when available) in attempt telemetry/manifest entries.
- [ ] 4.2 Add aggregate counters for cache hits/misses, stale-served (`stale_if_error`), and offline-fallback runs in `summary.build_summary_record` or equivalent metrics.
- [ ] 4.3 Emit debug logs when responses are served from cache or refreshed, ensuring logs include host, age, TTL, and whether stale content was used.
- [ ] 4.4 Provide operator tooling (CLI flag or script) to purge cache directories and to inspect cache statistics (`session.cache.responses`, `delete(expired=True)`).

## 5. Testing & QA
- [ ] 5.1 Create unit tests for the cache factory validating per-host TTL calculation, ignored params stripping, binary filter behavior, and offline mode.
- [ ] 5.2 Update resolver tests to mock the cached session and assert that repeat requests hit the cache (`from_cache=True`) while PDF downloads bypass it.
- [ ] 5.3 Add integration tests that simulate `stale_if_error` and `stale_while_revalidate` by injecting failing responses and ensuring stale data is served as specified.
- [ ] 5.4 Extend pipeline/end-to-end tests to verify cache metrics appear in summaries and that cache hits reduce external calls compared to baseline.
- [ ] 5.5 Document and run the regression command set (pytest targets, manual smoke run with cache enabled, offline-only scenario) as part of QA artifacts.

## 6. Documentation & Rollout
- [ ] 6.1 Update `src/DocsToKG/ContentDownload/README.md` and `AGENTS.md` with cache configuration guidance, backend options, offline mode usage, and instructions for purging or inspecting the cache.
- [ ] 6.2 Revise `LibraryDocumentation/requests-cache.md` to include a “DocsToKG integration” section referencing the new session factory, default TTL map, and operational knobs.
- [ ] 6.3 Annotate `requests-cache_transition_plan.md` to link to the openspec change and note any deviations resolved during implementation.
- [ ] 6.4 Add release notes / changelog entries explaining the migration, expected cache behavior, and recommended operational adjustments.
- [ ] 6.5 Communicate rollout plan (enable cache in staging, monitor hit-rate metrics, roll to production once stable) and capture follow-up tasks for tuning TTL overrides after telemetry review.
