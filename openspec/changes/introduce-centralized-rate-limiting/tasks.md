## 1. Implementation
- [ ] 1.1 Create `RatePolicy` and `RolePolicy` dataclasses (or TypedDicts) describing per-role `rates`, `max_delay_ms`, `mode`, and `count_head` flags inside `src/DocsToKG/ContentDownload/networking.py` (or a new `ratelimit.py` helper module).
- [ ] 1.2 Add a module-level registry `HOST_POLICIES: Dict[str, RolePolicy]` containing default entries for currently-known hosts (OpenAlex, Crossref, arXiv, Unpaywall, publisher domains surfaced in resolver presets).
- [ ] 1.3 Implement a `validate_policies()` startup hook that iterates every role’s rate list, runs `pyrate_limiter.validate_rate_list`, and raises/logs a fatal error if ordering rules are violated; invoke this during client construction.
- [ ] 1.4 Build a `get_limiter(host: str, role: str, overrides: RateOverride | None)` helper that caches `Limiter` instances keyed by `(host, role)` and ensures the chosen bucket backend is initialised exactly once per key.
- [ ] 1.5 Create a `build_bucket(host, role, backend_config)` helper that supports `memory`, `multiprocess`, `sqlite`, `redis`, and `postgres` backends; wire backend-specific parameters (path, DSN, redis URL) from configuration defaults.
- [ ] 1.6 Implement a `RateLimitedTransport` class that wraps `httpx.Transport`, intercepts every send, resolves `host` and request `extensions["role"]`, acquires tokens (respecting `count_head`), applies `max_delay_ms`, and raises a rich `RateLimitError` when the limiter cannot acquire within budget.
- [ ] 1.7 Update `httpx_transport.py` (or equivalent client factory) so the Hishel `CacheTransport` is layered above the new `RateLimitedTransport`, ensuring cache hits bypass the limiter while cache misses honour the limiter.
- [ ] 1.8 Ensure every call site that uses the shared HTTP client sets an explicit `extensions["role"]` value (`"metadata"`, `"landing"`, `"artifact"`) before dispatch; add `metadata` defaulting logic for legacy paths.
- [ ] 1.9 Remove or refactor existing sleeps, per-host semaphores, and the `TokenBucket` implementation so networking is the single authority; keep compatibility shims when CLI flags are still public.
- [ ] 1.10 Adjust Tenacity retry wrappers to treat limiter exceptions as non-retryable while continuing to retry upstream 429/503 responses and socket errors.

## 2. Configuration
- [ ] 2.1 Introduce CLI flags `--rate host=limits`, `--rate-mode host=mode`, `--rate-backend backend[:params]`, and matching environment variables; document their syntax in `README.md`.
- [ ] 2.2 Parse CLI overrides into structured objects that merge with defaults and feed into the `HOST_POLICIES` registry before validation.
- [ ] 2.3 Preserve existing CLI throttling flags by mapping them into the new configuration layer (e.g., `--domain-token-bucket` seeds rates, `--max-concurrent-per-host` converts to limiter wait mode); emit deprecation warnings where behaviour changes.
- [ ] 2.4 Emit a startup table logging effective host policies (host, role, rates, mode, max delay, backend) so operators can confirm configuration.

## 3. Telemetry & Observability
- [ ] 3.1 Add counters for `rate_limiter_acquire_total{host,role}`, `rate_limiter_block_total{host,role}`, and histograms for `rate_limiter_wait_ms{host,role}` in the existing telemetry module.
- [ ] 3.2 Record limiter events alongside Hishel cache metadata on each attempt so manifests and metrics capture cache-hit vs limited-hit breakdowns.
- [ ] 3.3 Ensure `RateLimitError` instances propagate host, role, wait duration, and next-available timestamps to manifest/summary logging.
- [ ] 3.4 Update run summaries to surface limiter statistics (total waits, blocks, average wait duration) next to retry/backoff metrics.

## 4. Testing
- [ ] 4.1 Add unit tests that simulate cache hits versus misses and assert that only misses consume limiter tokens.
- [ ] 4.2 Add tests for multi-window behaviour by configuring two rates, issuing bursts that violate the fast window but respect the slow window, and asserting waits/blocks match expectations.
- [ ] 4.3 Test per-role `max_delay_ms` differences by forcing congestion and verifying metadata requests raise quickly while artifact requests wait within allowance.
- [ ] 4.4 Verify Tenacity does not retry limiter exceptions yet still honours server `Retry-After` by simulating both cases.
- [ ] 4.5 Add integration tests or contract tests that confirm CLI overrides (rates, backend, mode) alter runtime behaviour and startup logging.
- [ ] 4.6 Provide regression tests covering legacy throttling flags to confirm their new mappings produce equivalent or explicitly documented behaviour.

## 5. Documentation
- [ ] 5.1 Update `src/DocsToKG/ContentDownload/README.md` networking/politeness sections with the new rate-limiter architecture, configuration flags, telemetry, and migration notes from deprecated options.
- [ ] 5.2 Add docstrings and inline comments explaining RatePolicy roles, limiter caching, and transport layering so future contributors understand why the limiter sits beneath Hishel.
- [ ] 5.3 Extend internal runbooks or operations docs to include guidance on choosing backends (memory vs redis vs sqlite) and tuning rates per provider.

## 6. Migration & Rollout
- [ ] 6.1 Provide a migration checklist to clean up existing run configurations (remove manual sleeps, update CLI invocations) and circulate to operational owners.
- [ ] 6.2 Stage the change behind a feature flag or config toggle, if needed, so pilot runs can compare old vs new limiter behaviour before full cutover.
- [ ] 6.3 Monitor early runs for unexpected limiter blocks or 429s, and tune default policies accordingly before declaring the migration complete.
