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

<!-- 2025-10-23 05:59:31Z UTC -->
## Pass 4 — find and fix real bugs

<!-- 2025-10-23 05:59:43Z UTC -->
## Pass 5 — find and fix real bugs

<!-- 2025-10-23 06:00:14Z UTC -->
## Pass 1 — find and fix real bugs
