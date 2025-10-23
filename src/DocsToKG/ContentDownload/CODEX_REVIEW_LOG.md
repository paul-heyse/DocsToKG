# Codex Review Log — `src/DocsToKG/ContentDownload`
This log is maintained by `scripts/codex-review-parallel.sh`.

<!-- 2025-10-22 16:55:34Z UTC -->
No eligible files after excludes.

<!-- 2025-10-22 17:01:20Z UTC -->
No eligible files after excludes.

<!-- 2025-10-22 17:13:46Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 01:39:11Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 02:48:00Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 03:05:49Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 03:55:00Z UTC -->
No eligible files after excludes.

<!-- 2025-10-23 04:04:17Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: Token bucket ignored the configured burst allowance, so rate limiting was stricter than configured and could starve bursty resolvers.
- Fix:
  - Track `capacity + burst` as the effective ceiling and honor it during refill/refund.
- TODO: Add a focused unit test covering burst replenishment in the resolver HTTP client token bucket.

<!-- 2025-10-23 04:06:49Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 04:06:56Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 1 (Pass 2)
- Broken: Dispatcher forced a lease even when the worker queue was full, so `_jobs_queue.put(block=False)` raised `queue.Full` and the dispatcher loop thrashed with errors instead of backoff.
- Fix: Respect actual free slot count before leasing; skip leasing when at capacity so `put` never runs without space.
- TODO: Follow up with a lightweight dispatcher unit test that covers full-queue behaviour before the next orchestrator refactor.

### Batch 0 (Pass 2)
- Broken: String-based `cacheable_methods` left leading spaces (`" HEAD"`), so HEAD revalidation stopped working whenever YAML/env overrides used human-friendly commas.
- Fix: Strip whitespace from string/list inputs while loading cache controller config and manually verified spaced strings resolve to canonical methods.
- TODO: Audit env/CLI override parsing for similar whitespace edge cases before widening the configuration surface again.

<!-- 2025-10-23 04:24:41Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Fixed boolean config overrides being misread as truthy strings, which forced cache `body_key` and controller `allow_heuristics` flags on even when explicitly disabled via env/CLI.
- Added `_coerce_bool` helper and reused it where we ingest those fields so future overrides remain type-safe.
- TODO: Backfill targeted tests for boolean CLI/env overlays to guard against regressions.

### Batch 1 (Pass 1)
- Found resolver HTTP client backoff casting milliseconds to `int`, producing zero-second retries that hammer upstreams and ignore jitter caps.
- Switched to float math for delay calculations, clamping to configured caps and safeguarding against negative sleeps.
- TODO: Add a focused retry/backoff unit test covering fractional delays and cache-hit token refunds.

<!-- 2025-10-23 04:29:31Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Cache router silently dropped `swrv_s` for non-metadata roles, so landing-page policies never served stale-while-revalidate despite being configured.
- Removed the metadata-only guard when carrying `swrv_s` into `CacheDecision`, allowing any cached role to honor its configured SWRV window.
- TODO: Add a regression test covering landing-role SWRV propagation when the test matrix is next updated.

### Batch 1 (Pass 2)
- Keyed limiter `release` silently created a brand-new semaphore when a key was missing, inflating concurrency caps instead of detecting the logic error.
- Reworked `_get_semaphore` to optionally skip creation, gate TTL eviction on idle semaphores, and make `release` raise if no tracked semaphore exists.
- TODO: Consider tracking active permit counts explicitly so TTL eviction can safely reap entries without private attribute checks.

<!-- 2025-10-23 04:46:08Z UTC -->
## Pass 1 — find and fix real bugs
### Batch 1 (Pass 1)
- Per-role limiter never released its `BoundedSemaphore` permit, so concurrency hit zero after a few requests.
- Added explicit tracking of semaphore acquisition, rolling back on failures and releasing post-request.
- TODO: Add stress test exercising `max_concurrent` behaviour once concurrency harness is in place.

<!-- 2025-10-23 04:50:48Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 05:59:19Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 05:59:22Z UTC -->
## Pass 2 — find and fix real bugs

<!-- 2025-10-23 05:59:25Z UTC -->
## Pass 3 — find and fix real bugs

### Batch 0 (Pass 3)
- Broken: `is_fresh` treated responses exactly at `max-age` (and stale grace cutoffs) as stale, forcing unnecessary revalidation despite RFC 7234 permitting equality.
- Fix:
  - Relaxed the freshness and stale-serving comparisons in `CacheControlDirective` helpers to honor equality with configured TTLs.
- TODO: Backfill a boundary-condition test to lock in the `<=` behaviour for `max-age` and stale extension windows.

<!-- 2025-10-23 05:59:31Z UTC -->
## Pass 4 — find and fix real bugs

<!-- 2025-10-23 05:59:43Z UTC -->
## Pass 5 — find and fix real bugs
### Batch 0 (Pass 5)
- Broken: Tenacity retry predicate assumed `RetryCallState.outcome` objects and called `.exception()`, so when Tenacity passed the raw exception the policy raised `AttributeError` and aborted retries instead of classifying the failure.
- Fix:
  - Split predicate handling for exceptions vs. results and return both callables from `_make_retry_predicate`.
  - Wire `retry_if_exception` and `retry_if_result` to the appropriate predicate so Tenacity no longer triggers spurious errors.
- TODO: Backfill a unit test that exercises Tenacity retries on exception paths to guard the signature contract.

<!-- 2025-10-23 06:00:14Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:45:07Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:45:10Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 7)
- Broken: `compute_lease_acquisition_latency` hard-coded every lease to 120 s because it subtracted `created_at` from itself, so the p50/p99 SLOs always read 120 000 ms regardless of real data.
- Fix:
  - Rebuilt the query to derive latency from the first recorded operation (fallback to the lease row when still active) and discard non-positive samples.
- TODO: Persist an explicit `first_leased_at` timestamp so the SLO no longer needs to infer from operation ledgers.

<!-- 2025-10-23 06:52:53Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 06:55:41Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: Atomic rename path leaked directory file descriptors by calling `os.open()` without closing, eventually exhausting descriptors on long-lived workers.
- Fix:
  - Wrap the directory sync in a try/finally that closes the descriptor before performing `os.replace`.
- TODO: Evaluate adding a post-rename `fsync` once we confirm durability requirements.

<!-- 2025-10-23 06:57:04Z UTC -->
## Pass 3 — find and fix real bugs

<!-- 2025-10-23 07:00:21Z UTC -->
## Pass 4 — find and fix real bugs
### Batch 0 (Pass 4)
- Broken: `HttpConfig.proxies` never reached the shared `httpx.Client`, so deployments configured with forward proxies sent traffic directly and failed in locked-down networks.
- Fix:
  - Plumbed `proxies` through `get_http_session` when constructing the shared client.
  - Extended the debug log to note whether a proxy mapping is active for easier diagnosis.
- TODO: Add a smoke test that verifies proxy URLs are honoured by the shared client.

<!-- 2025-10-23 07:01:53Z UTC -->
## Pass 5 — find and fix real bugs

<!-- 2025-10-23 07:04:31Z UTC -->
## Pass 6 — find and fix real bugs
### Batch 0 (Pass 6)
- Broken: `stream_download_to_file` leaked its temporary file descriptor whenever the HTTP request failed before the writer context started, eventually exhausting `EMFILE` limits under retry pressure.
- Fix:
  - Replace `mkstemp` usage with a `NamedTemporaryFile` context so the descriptor closes on all control-flow paths.
  - Close streaming `httpx` responses on early failures to avoid connection pool leaks during retries.
- TODO: Add a regression test that simulates repeated pre-stream failures and asserts file descriptor counts stay stable.

<!-- 2025-10-23 07:10:59Z UTC -->
## Pass 7 — find and fix real bugs

<!-- 2025-10-23 07:15:02Z UTC -->
## Pass 8 — find and fix real bugs
### Batch 0 (Pass 8)
- Broken: Deployment `FeatureFlag.should_enable` ignored the `enabled` toggle, so flags marked disabled still rolled out when their strategy hashed users into treatment buckets.
- Fix:
  - Short-circuit `should_enable` when `enabled` is false so the toggle reliably disables all strategies.
- TODO: Add a unit test that registers a canary flag with `enabled=False` and asserts `should_enable` never returns true.
