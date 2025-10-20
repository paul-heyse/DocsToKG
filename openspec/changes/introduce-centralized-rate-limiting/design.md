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
- **Limiter placement:** Insert a custom `RateLimitedTransport` between Hishel’s cache transport and the raw HTTP transport. This ensures cache hits do not consume tokens and aligns with pyrate-limiter guidance that only actual events should be recorded (`limiter.try_acquire` is invoked immediately before we hit the wire).
- **Role taxonomy:** Introduce three canonical roles—`metadata`, `landing`, `artifact`—to capture the markedly different politeness envelopes for API queries vs. HTML landing pages vs. large binary downloads. Roles are attached as `request.extensions["role"]` so the limiter can distinguish them without duplicating clients.
- **Policy registry:** Maintain a `HOST_POLICIES` mapping (host → per-role policy) that merges defaults with CLI/env overrides. Each role policy includes an ordered `rates` list, `mode` (`wait` with bounded delay vs `raise`), `max_delay_ms`, `count_head`, and optional `weight`. Policies are validated at startup via `pyrate_limiter.validate_rate_list`.
- **Backend abstraction:** Start with `InMemoryBucket` for single-process runs but factor construction into a helper that can switch to `MultiprocessBucket`, `SQLiteBucket`, `RedisBucket`, or `PostgresBucket` using configuration-provided parameters. This follows the library’s advice to encapsulate bucket choice while keeping limiter call sites stable.
- **Error propagation:** Wrap `BucketFullException`/`LimiterDelayException` in a project-specific `RateLimitError` carrying host/role context, attempted weight, elapsed wait, and next-allowed timestamp. Tenacity recognises this exception class as non-retryable, preventing exponential backoff loops on self-imposed limits.
- **Configuration surface:** Add composable CLI flags (`--rate`, `--rate-mode`, `--rate-backend`, `--rate-max-delay`) plus environment variable equivalents. Legacy throttling flags are parsed and either mapped into the new policy registry or deprecated with clear migration messaging.
- **Telemetry:** Emit counters and histograms via the existing telemetry infrastructure, using dimensions `{host, role}`. Each limiter decision is recorded alongside Hishel cache metadata so we can compute cache-hit savings vs. rate-limited waits. Startup logs print a structured table of active policies for auditability.

## Risks / Trade-offs
- **Default policy accuracy:** Mis-estimating provider quotas could cause unnecessary waits or upstream 429s. Mitigation: seed defaults with conservative values from provider documentation, make overrides easy, and closely monitor telemetry during rollout.
- **Increased startup work:** Validating rates and constructing limiter caches adds boot-time complexity. Mitigation: cache limiters lazily per `(host, role)` to avoid upfront cost and fail fast with explicit messages when configuration is invalid.
- **Legacy flag confusion:** Removing or remapping flags like `--domain-token-bucket` may surprise operators. Mitigation: provide compatibility shims, emit deprecation warnings, and document the new workflow with side-by-side examples.
- **Multi-process contention:** The initial `InMemoryBucket` backend cannot coordinate across processes; deployments using multiple workers per host may need a persistent backend sooner than planned. Mitigation: expose backend selection now and document when to switch to `MultiprocessBucket` or SQLite with file locks.
- **Telemetry volume:** Adding per-request limiter logging could inflate manifest size. Mitigation: aggregate metrics in counters/histograms and limit per-attempt payload to high-value fields.

## Migration Plan
1. Implement the new limiter transport and policy registry behind a feature toggle or configuration flag.
2. Map legacy throttling flags into the new configuration, emitting warnings so users can update scripts.
3. Run dual-path validation (old vs new limiter) in staging or controlled runs, comparing telemetry for waits, 429 rates, and throughput.
4. Adjust default policies based on observed data, update documentation, and obtain sign-off from operations.
5. Remove the feature toggle, delete redundant throttling code paths, and finalise documentation updates.
6. Schedule a retrospective to capture tuning guidance and any follow-up work (e.g., enabling Redis/SQLite backends).

## Open Questions
- Which hosts should ship with first-class default policies beyond OpenAlex, Crossref, arXiv, and Unpaywall? Do we need resolver-specific overrides for long-tail publisher domains?
- Should artifact downloads ever use weighted acquisitions (e.g., weight proportional to expected file size) to better reflect bandwidth consumption?
- Do we need a dry-run mode for the limiter to log potential waits without enforcing them for initial calibration runs?
- How should we expose policy changes in run telemetry for downstream analytics (e.g., include limiter configuration snapshots in manifest summaries)?
