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
