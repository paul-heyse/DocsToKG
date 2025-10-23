# Codex Review Log — `src/DocsToKG/OntologyDownload`
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
- Broken: rate limiter was configured without blocking semantics so polite client proceeded with HTTP calls even after slot acquisition failed.
- Fix: enable pyrate limiter blocking (max_delay + retry) and raise `PolicyError` when acquisition returns False to stop the HTTP request.
- TODO: add integration coverage that exercises fail-fast mode and verifies blocked requests never reach the transport.
