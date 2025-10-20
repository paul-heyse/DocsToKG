## Context
- ContentDownload currently combines custom `TokenBucket`, per-host semaphores, and optional `--sleep` delays to keep HTTP requests polite. These mechanisms live in multiple modules, complicating tuning and increasing the risk of duplicated or conflicting limits.
- The networking stack is built on `httpx` with a Hishel `CacheTransport` for on-disk caching and Tenacity for retries/backoff; cache hits should not consume rate budget, but today our throttling sits above the cache layer so cached responses still pay the cost.
- The refactor brief (see `ContentDownload_pyrate-limiter.md`) calls for a single `pyrate-limiter` layer that supports multi-window quotas, distinct roles (`metadata`, `landing`, `artifact`), and backend swappability (in-memory now, SQLite/Redis later) while remaining observable via existing telemetry.
- The pyrate-limiter reference (`LibraryDocumentation/pyrate-limiter.md`) highlights critical constraints: rate lists must be ordered and validated, bucket factories leak on a cadence to keep memory bounded, and limiter acquire semantics must handle weights and optional bounded waits.

## Goals / Non-Goals
- Goals:
  - Centralize all HTTP rate limiting in one module that builds and caches `pyrate_limiter.Limiter` instances keyed by `(host, role)`.
  - Layer the limiter beneath Hishel so cache hits bypass quotas and only real network traffic is governed.
  - Provide configuration knobs for host policies, limiter modes, and backend selection without code edits.
  - Emit detailed telemetry (counts, waits, blocks) so operators can tune rates confidently.
  - Preserve Tenacity’s responsibility for upstream-driven retries while avoiding retry storms on self-imposed rate limits.
- Non-Goals:
  - Implement distributed backends on day one (Redis/Postgres support is planned but not a launch requirement).
  - Rewrite resolver logic or manifest schemas beyond the telemetry extensions required to surface limiter data.
  - Change ContentDownload’s concurrency model (workers/coroutines) beyond removing redundant throttling constructs.

## Decisions
- **Limiter placement:** Insert a custom `RateLimitedTransport` between Hishel’s `CacheTransport` and the raw HTTP transport inside `httpx_transport._create_client_unlocked`. This ensures cache hits never reach the limiter while every cache miss or revalidation must pass through the limiter’s `try_acquire`. Both sync (`handle_request`) and streaming pathways must be wrapped because `request_with_retries(..., stream=True)` is used during large artifact downloads.
- **Role taxonomy:** Introduce three canonical roles—`metadata`, `landing`, `artifact`—to capture the markedly different politeness envelopes for API queries vs. HTML landing pages vs. large binary downloads. Roles are attached as `request.extensions["role"]` so the transport can differentiate without needing separate clients. `request_with_retries` gains a `role` keyword that defaults to `"metadata"` and every call site (download pipeline, resolvers, robots cache) is updated explicitly, with helper functions for ambiguous cases (e.g., HEAD prechecks remain `"metadata"`).
- **Policy registry:** Maintain a `HOST_POLICIES` mapping (host → per-role policy) that merges defaults with CLI/env overrides. Each role policy describes an ordered `rates` list (`List[Rate]`), limiter `mode` (`"wait"` with `max_delay_ms` or `"raise"`), `count_head` behaviour, and optional `weight` multipliers for artifact downloads. A dedicated `validate_policies` helper invokes `pyrate_limiter.validate_rate_list` for every (host, role) combination at startup and raises descriptive errors when ordering is invalid.
- **Backend abstraction:** Start with `InMemoryBucket` for single-process runs but factor construction into a helper that can switch to `MultiprocessBucket`, `SQLiteBucket`, `RedisBucket`, or `PostgresBucket` using configuration-provided parameters. This follows the library’s advice to encapsulate bucket choice while keeping limiter call sites stable.
- **Error propagation:** Wrap `BucketFullException`/`LimiterDelayException` in a project-specific `RateLimitError` carrying host, role, elapsed wait, computed `next_allowed_at`, limiter mode, and backend name. Tenacity recognises this exception class as non-retryable, preventing exponential backoff loops on self-imposed limits while still retrying upstream-driven 429/503 responses.
- **Configuration surface:** Add composable CLI flags (`--rate`, `--rate-mode`, `--rate-backend`, `--rate-max-delay`) plus environment variable equivalents. Legacy throttling flags are parsed and either mapped into the new policy registry (`--domain-token-bucket` → `Rate` list, `--max-concurrent-per-host` → wait-mode with bounded delay) or deprecated with clear migration messaging. `ResolvedConfig` is extended to carry a typed `RateConfig` payload into the runner.
- **Telemetry:** Emit counters and histograms via the existing telemetry infrastructure, using dimensions `{host, role}`. Each limiter acquisition annotates `docs_network_meta` so attempt records gain `rate_limiter_wait_ms`, `rate_limiter_backend`, and `rate_limiter_mode`. Startup logs print a structured table of active policies for auditability. `summary.emit_console_summary` surfaces totals/averages for waits and blocks.
- **Pipeline cleanup:** Delete `TokenBucket`, domain-level sleep loops, and host semaphores inside `ResolverPipeline`; instead rely on the limiter’s bounded wait semantics. Keep circuit breakers as-is (unrelated to rate limiting). Provide a shim so CLI options setting these features emit deprecation notices pointing to the new `--rate` syntax.

## Risks / Trade-offs
- **Default policy accuracy:** Mis-estimating provider quotas could cause unnecessary waits or upstream 429s. Mitigation: seed defaults with conservative values from provider documentation, make overrides easy, and closely monitor telemetry during rollout.
- **Increased startup work:** Validating rates and constructing limiter caches adds boot-time complexity. Mitigation: cache limiters lazily per `(host, role)` to avoid upfront cost and fail fast with explicit messages when configuration is invalid.
- **Legacy flag confusion:** Removing or remapping flags like `--domain-token-bucket` may surprise operators. Mitigation: provide compatibility shims, emit deprecation warnings, and document the new workflow with side-by-side examples.
- **Multi-process contention:** The initial `InMemoryBucket` backend cannot coordinate across processes; deployments using multiple workers per host may need a persistent backend sooner than planned. Mitigation: expose backend selection now and document when to switch to `MultiprocessBucket` or SQLite with file locks.
- **Telemetry volume:** Adding per-request limiter logging could inflate manifest size. Mitigation: aggregate metrics in counters/histograms and limit per-attempt payload to high-value fields.
- **Dependency surface:** Redis/Postgres bucket support requires additional optional dependencies. Mitigation: guard backend loading with import checks and provide actionable error messages instructing operators to install extras when selecting those backends.

## Migration Plan
1. Implement the new limiter transport and policy registry behind a feature toggle or configuration flag.
2. Map legacy throttling flags into the new configuration, emitting warnings so users can update scripts; collect operator feedback on preferred defaults.
3. Run dual-path validation (old vs new limiter) in staging or controlled runs, comparing telemetry for waits, 429 rates, throughput, and manifest size impact.
4. Adjust default policies based on observed data, update documentation, and obtain sign-off from operations.
5. Remove the feature toggle, delete redundant throttling code paths, and finalise documentation updates.
6. Schedule a retrospective to capture tuning guidance and any follow-up work (e.g., enabling Redis/SQLite backends).

## Open Questions
- Which hosts should ship with first-class default policies beyond OpenAlex, Crossref, arXiv, and Unpaywall? Do we need resolver-specific overrides for long-tail publisher domains?
- Should artifact downloads ever use weighted acquisitions (e.g., weight proportional to expected file size) to better reflect bandwidth consumption?
- Do we need a dry-run mode for the limiter to log potential waits without enforcing them for initial calibration runs?
- How should we expose policy changes in run telemetry for downstream analytics (e.g., include limiter configuration snapshots in manifest summaries)?
- Should `RateLimitedTransport` support asynchronous transports now, or can we defer until the codebase migrates to async workflows?
- What guardrails are needed to keep resolver-specific roles (e.g., custom landing-page scrapers) aligned with the three canonical roles without proliferating new categories?
