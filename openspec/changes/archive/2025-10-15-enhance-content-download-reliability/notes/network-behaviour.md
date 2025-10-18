# Network Behaviour Expectations

Runbook guidance distilled from the retry refactor regression tests.

## Retry Envelope

- Total attempts equal `max_retries + 1`. Verified via
  `tests/test_download_retries.py::test_retry_determinism_matches_request_with_retries`
  using a 429/429/200 sequence.
- `Retry-After` overrides exponential backoff when it specifies a longer delay;
  captured delay values are asserted in the test suite.

## Request Topology

- Per-download HEAD probes are eliminated. The downloader issues a single GET
  per candidate URL; pipeline-level HEAD pre-check remains opt-in through
  `ResolverConfig.enable_head_precheck`.
- `tests/test_download_retries.py::test_download_candidate_avoids_per_request_head`
  tracks HEAD call counts to prevent regressions.

## Failure Handling

- 404/410 responses shortcut retries and are logged as `http_error`.
- Connection errors, timeouts, and 5xx status codes trigger the central retry
  helper with jittered exponential backoff.
- HTML served with PDF content type is detected via trailing buffer heuristics
  and flagged as `pdf_corrupt`.

## Operator Checklist

- Inspect `.metrics.json` files for counters such as `retry_attempts_exhausted` when
  diagnosing slowdowns.
- Use structured retry logs (`retry_attempt` events) to correlate delays with
  Retry-After guidance versus backoff.
- Adjust retry parameters centrally (`http.request_with_retries`)—HTTP adapters
  run with `max_retries=0`, so no per-session tuning is needed.
