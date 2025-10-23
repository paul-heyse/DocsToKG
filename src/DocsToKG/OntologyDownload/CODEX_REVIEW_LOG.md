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

<!-- 2025-10-23 04:29:43Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: HTTP retry policies ignored `Retry-After` guidance, causing immediate reattempts that violate polite backoff promises under 429/503 responses.
- Fix: Added a `_RetryAfterOrBackoff` wait strategy that parses `Retry-After` hints and wired it into the default, idempotent, aggressive, and rate-limit retry builders so they fall back to jittered exponential waits when no header is present.
- TODO: Add a retry-policy unit test that asserts we honour header and HTTP-date variants of `Retry-After`.

<!-- 2025-10-23 04:46:08Z UTC -->
## Pass 1 — find and fix real bugs

### Batch 0 (Pass 1)
- Broken: `retry_http_request` retried every `HTTPStatusError`, so 4xx responses were needlessly reissued and non-idempotent calls could run multiple times.
- Fix: limit the decorator's stop condition with `stop_after_attempt` and reuse the existing predicate so only connection faults and 5xx responses trigger retries.
- TODO: add decorator-focused tests that emulate 4xx and 5xx responses to confirm retry boundaries.

<!-- 2025-10-23 04:48:24Z UTC -->
## Pass 2 — find and fix real bugs

### Batch 0 (Pass 2)
- Broken: `PoliteHttpClient.close()` never released the shared HTTPX client, leaking open connection pools despite callers invoking `close_polite_http_client()`.
- Fix: delegate client shutdown to `close_http_client()` so connection pools are torn down and the singleton can rebuild cleanly on the next request.
- TODO: add a shutdown test that asserts `get_http_client()` returns a fresh instance after closing the polite client.

<!-- 2025-10-23 05:59:19Z UTC -->
## Pass 1 — find and fix real bugs

<!-- 2025-10-23 05:59:23Z UTC -->
## Pass 2 — find and fix real bugs
