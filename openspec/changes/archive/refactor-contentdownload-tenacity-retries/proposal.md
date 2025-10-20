# Refactor ContentDownload Retries with Tenacity

## Why
- The ContentDownload stack keeps a hand-built retry loop in `networking.request_with_retries`, duplicating backoff math, response cleanup, and retry logging across modules. Maintaining this logic has become error-prone and obscures how retry knobs map to runtime behaviour.
- Tenacity is already available in the environment and offers composable stop/wait/retry primitives (including Retry-After aware waits) that match the documented transition plan. Centralising on it lets us delete bespoke jitter helpers while gaining observability hooks.
- Tests and operators rely on predictable retry semantics; consolidating the policy into Tenacity with explicit patch points removes the need to monkeypatch `time.sleep` directly and ensures future tuning happens in one place.

## What Changes
- Replace the manual loop inside `src/DocsToKG/ContentDownload/networking.py:request_with_retries` with a Tenacity `Retrying` policy that composes `retry_if_exception_type` for networking errors, `retry_if_result` for retryable HTTP statuses, and `stop_after_attempt`/`stop_after_delay` derived from existing parameters.
- Introduce a Tenacity `wait_base` implementation that prefers parsed `Retry-After` headers (bounded by `retry_after_cap` and `backoff_max`) before falling back to `wait_random_exponential`, and wire it into the Tenacity policy so header-driven sleeps and jitter share one code path.
- Remove `_calculate_equal_jitter_delay`, direct `time.sleep` usage, and ad-hoc retry logging from `networking.py`, ensuring `before_sleep` hooks close stale `requests.Response` objects and emit comparable debug/warning logs.
- Audit ContentDownload callers so resolvers, `download.py`, `head_precheck`, and the robots cache use the Tenacity-backed entrypoint, deleting any local retry/sleep helpers and keeping `networking.request_with_retries` as the single retry surface.
- Update ContentDownload docs/tests to describe and exercise the Tenacity behaviour (including patch points for the Tenacity sleep function) and refresh fixtures that asserted on legacy equal-jitter delays.

## Impact
- **Affected specs:** content-download
- **Affected code:** `src/DocsToKG/ContentDownload/{networking.py,download.py,pipeline.py,resolvers/**,AGENTS.md,README.md}`, `tests/content_download/test_networking.py`, `tests/resolvers/**`, `LibraryDocumentation/Tenacity_Transition_Plan.md`
