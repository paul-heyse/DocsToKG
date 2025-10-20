## 1. Implementation
- [x] 1.1 Introduce a new module section `# --- Rate limiting ---` near the top of `src/DocsToKG/ContentDownload/networking.py` (or a sibling `ratelimit.py`) defining `@dataclass class RolePolicy` with fields `rates: Dict[str, List[Rate]]`, `max_delay_ms: Dict[str, int]`, `mode: Dict[str, Literal["wait","raise"]]`, `count_head: Dict[str, bool]`, and optional `weight: Dict[str, int]`; include a helper `RolePolicy.for_role(role)` that returns a typed view for a single role.
- [x] 1.2 Add a module-level `HOST_POLICIES: Dict[str, RolePolicy]` populated with defaults for `api.openalex.org`, `api.crossref.org`, `export.arxiv.org`, `api.unpaywall.org`, and the publisher domains referenced in resolver presets; annotate the structure with inline comments describing each rate window (per-second, per-minute, etc.).
- [x] 1.3 Implement `def validate_policies(policies: Mapping[str, RolePolicy]) -> None` that iterates each host/role list, calls `validate_rate_list` from `pyrate_limiter`, and logs/raises `ValueError` with the offending host/role when ordering rules fail; invoke this during HTTP client construction so invalid overrides abort early.
- [x] 1.4 Add a `LimiterCache` helper (simple dict guarded by `threading.Lock`) that stores `Limiter` instances keyed by `(host, role)` and ensures the limiter is built once per combination. Expose `get_limiter(host, role, *, overrides=None)` that returns the cached limiter and records the selected backend for telemetry.
- [x] 1.5 Implement `build_bucket(host, role, backend_config)` that reads a new config structure (see Section 2) and instantiates the correct bucket type: `InMemoryBucket` (default), `MultiprocessBucket` (requires `multiprocessing.Manager` lock), `SQLiteBucket` (accepts file path, file-lock flag), `RedisBucket` (parses redis URL and namespace), `PostgresBucket` (uses psycopg3 connection string). Document required parameters in docstrings and raise friendly errors when dependencies are missing.
- [x] 1.6 Create a new class `class RateLimitedTransport(httpx.BaseTransport)` wrapping an inner `httpx.BaseTransport`. Override `handle_request(self, request)` and `handle_async_request` (if async is supported) to:
  - extract `host = request.url.host` and `role = request.extensions.get("role", "metadata")`,
  - short-circuit when the policy marks `count_head=False` and the method is `HEAD`,
  - look up the limiter via `get_limiter`,
  - call `limiter.try_acquire(host_key, weight=policy.weight.get(role, 1), max_delay_ms=policy.max_delay_ms[role])`,
  - measure wait time via `Limiter.try_acquire` (use `Limiter` return metadata),
  - on success, set `request.extensions.setdefault("docs_network_meta", {})["rate_limiter_wait_ms"]`,
  - on `BucketFullException` or exceeded wait budget, raise `RateLimitError` populated with host, role, waited_ms, `next_allowed_at` timestamp.
- [x] 1.7 Update `DocsToKG.ContentDownload.httpx_transport._create_client_unlocked` so the inner `HTTPTransport` is wrapped in `RateLimitedTransport` before being given to Hishel `CacheTransport`. Ensure the cache transport sits above the limiter and existing event hooks still execute.
- [x] 1.8 Extend `request_with_retries` to accept a new keyword argument `role: str = "metadata"`; set `request.extensions["role"] = role` before dispatching so the transport can locate the correct limiter. Propagate the role parameter through all internal helper invocations (including streaming paths and head prechecks).
- [x] 1.9 Audit the following modules and update every call to `request_with_retries` to pass a role:
  - `download.py` (artifact downloads → `"artifact"`, metadata/json fetches → `"metadata"`),
  - `resolvers/base.py` and individual resolvers (HTML/JSON probes → `"landing"` or `"metadata"` as appropriate),
  - `networking.head_precheck` (HEAD probe → `"metadata"`),
  - `pipeline.py` download workers when prechecking or fetching landing pages.
  Add inline comments where role selection is non-obvious.
- [x] 1.10 Remove pipeline-level throttles that will be redundant:
  - delete the `TokenBucket` class in `networking.py` (if still used) and its imports,
  - remove `_respect_domain_limit`, `_ensure_host_bucket`, `_acquire_host_slot`, and related fields in `ResolverPipeline`,
  - replace their usage with a lightweight helper that derives `(host, role)` and lets the limiter transport block instead of sleeping.
- [x] 1.11 Update Tenacity setup inside `request_with_retries` so the retry predicate treats `RateLimitError` (and any `BucketFullException` raised through wrappers) as non-retryable; only HTTP 429 with server-provided retry-after should continue to retry via Tenacity. Add regression tests to verify behaviour.
- [x] 1.12 Extend `DocsToKG.ContentDownload.errors.RateLimitError` to accept `host`, `role`, `waited_ms`, and `next_allowed_at` keyword arguments and expose them via attributes; update `log_download_failure` to include these fields in the emitted metadata dict.
- [x] 1.13 Ensure `RateLimitError` instances raised in the transport propagate through pipeline/download call sites without being wrapped in generic exceptions (update except blocks to catch and rethrow when needed).

## 2. Configuration
- [x] 2.1 Add CLI flags in `DocsToKG.ContentDownload.args.build_parser`:
  - `--rate HOST=WINDOWS` accepts comma-separated windows like `10/s,5000/h`,
  - `--rate-mode HOST={raise,wait:ms}` controls limiter mode and maximum wait milliseconds,
  - `--rate-backend backend[:options]` selects `memory`, `multiprocess`, `sqlite:path=/tmp/rl.db`, `redis:url=redis://localhost:6379/0`, or `postgres:dsn=…`,
  - `--rate-max-delay HOST.role=milliseconds` for fine-grained overrides,
  - matching `DOCSTOKG_RATE_*` environment variables (document names).
  Use custom `argparse.Action` classes with clear error messages for bad input.
- [x] 2.2 Extend `ResolvedConfig` (and dependent code in `cli.py`/`runner.py`) to carry parsed `RateConfig` objects so downstream modules do not reparse CLI strings. Add tests covering the new dataclass fields.
- [x] 2.3 Implement `merge_rate_overrides(defaults, overrides)` that folds CLI/env overrides into `HOST_POLICIES` prior to calling `validate_policies`; ensure overrides honour per-role distinctions and missing hosts create new registry entries.
- [x] 2.4 Interpret legacy throttling flags in `resolve_config`:
  - Map `--sleep` and `resolver_min_interval_s` to resolver-level pacing (retain existing behaviour).
  - Translate `--domain-token-bucket` and `--domain-min-interval` into rate policies (e.g., convert `3/second` into `Rate(3, Duration.SECOND)` with `raise` mode) and log `DeprecationWarning` instructing users to adopt `--rate`.
  - Disable `--max-concurrent-per-host` by default and log that concurrency is now governed by the limiter wait budget.
- [x] 2.5 Emit a structured policy table at startup from `runner.py` or `cli.py` (before the run begins) showing host, role, ordered rates, mode, max_delay, backend, and count_head flag. Ensure logging respects JSON/structured output when enabled.

## 3. Telemetry & Observability
- [x] 3.1 Extend `DocsToKG.ContentDownload.telemetry.RunTelemetry` (or the relevant sink writer) to record new metrics: `rate_limiter_acquire_total{host,role}`, `rate_limiter_wait_ms_sum`, `rate_limiter_block_total`, and `rate_limiter_policy_backend{host}` gauges. Ensure metrics are thread-safe.
- [x] 3.2 Update `AttemptRecord` (or the manifest payload) to include optional fields `rate_limiter_wait_ms`, `rate_limiter_mode`, `rate_limiter_backend`, and `from_cache` so cached hits and waits can be analysed together.
- [x] 3.3 Modify `summary.emit_console_summary` and `statistics.DownloadStatistics` to aggregate limiter waits/blocks, include them in the console report, and persist them in `manifest.metrics.json`.
- [x] 3.4 Enhance `log_download_failure` to attach limiter metadata when `RateLimitError` occurs; confirm manifest sinks persist the extra keys without schema violations.
- [x] 3.5 Add structured logging (`LOGGER.info("rate-policy", …)`) during startup showing resolved policies and `LOGGER.debug("rate-acquire", …)` for each limiter acquisition when debug logging is enabled.

## 4. Testing
- [ ] 4.1 Add unit tests that simulate cache hits versus misses and assert that only misses consume limiter tokens.
- [ ] 4.2 Add tests for multi-window behaviour by configuring two rates, issuing bursts that violate the fast window but respect the slow window, and asserting waits/blocks match expectations.
- [ ] 4.3 Test per-role `max_delay_ms` differences by forcing congestion and verifying metadata requests raise quickly while artifact requests wait within allowance; include both sync and streaming (`client.stream`) calls.
- [x] 4.4 Verify Tenacity does not retry limiter exceptions yet still honours server `Retry-After` by simulating both cases inside `request_with_retries` tests.
- [ ] 4.5 Add integration tests or contract tests that confirm CLI overrides (rates, backend, mode) alter runtime behaviour and startup logging (e.g., using click runner or argparse harness).
- [ ] 4.6 Provide regression tests covering legacy throttling flags to confirm their new mappings produce equivalent or explicitly documented behaviour.
- [ ] 4.7 Add smoke tests ensuring `RateLimitedTransport` can be constructed with each backend type (skip redis/postgres tests when dependencies unavailable, but guard with feature flags).

## 5. Documentation
- [ ] 5.1 Update `src/DocsToKG/ContentDownload/README.md` networking/politeness sections with the new rate-limiter architecture, configuration flags, telemetry, and migration notes from deprecated options.
- [ ] 5.2 Add docstrings and inline comments explaining RatePolicy roles, limiter caching, and transport layering so future contributors understand why the limiter sits beneath Hishel.
- [ ] 5.3 Extend internal runbooks or operations docs to include guidance on choosing backends (memory vs redis vs sqlite) and tuning rates per provider.
- [ ] 5.4 Update `docs-instruct/DO NOT DELETE - Refactor review/ContentDownload/ContentDownload_pyrate-limiter.md` with implementation notes once complete (if that document remains source of truth).

## 6. Migration & Rollout
- [ ] 6.1 Provide a migration checklist to clean up existing run configurations (remove manual sleeps, update CLI invocations) and circulate to operational owners.
- [ ] 6.2 Stage the change behind a feature flag or config toggle, if needed, so pilot runs can compare old vs new limiter behaviour before full cutover.
- [ ] 6.3 Monitor early runs for unexpected limiter blocks or 429s, and tune default policies accordingly before declaring the migration complete.
