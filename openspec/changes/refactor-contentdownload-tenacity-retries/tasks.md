## 1. Embed Tenacity Retry Engine in `networking.py`
- [x] 1.1 Import Tenacity primitives (`Retrying`, `RetryError`, `RetryCallState`, `retry_if_exception_type`, `retry_if_result`, `stop_after_attempt`, `stop_after_delay`, `wait_random_exponential`, `before_sleep_log`) and wire them into the module `__all__`/typing as needed.
- [x] 1.2 Introduce module-level constants: `TENACITY_SLEEP = time.sleep`, `DEFAULT_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}`, and a dedicated logger for Tenacity callbacks (reuse `LOGGER` to keep structured messages consistent).
- [x] 1.3 Implement `RetryAfterJitterWait` (subclass `tenacity.wait.wait_base`) that accepts `respect_retry_after`, `retry_after_cap`, `backoff_max`, and a fallback `wait_random_exponential` strategy; in `__call__`, read the last `requests.Response` from `retry_state.outcome`, parse `Retry-After` via `parse_retry_after_header`, bound it, and fall back to the exponential jitter when the header is missing or disabled.
- [x] 1.4 Add `_before_sleep_close_response(retry_state: RetryCallState)` which closes the previous `requests.Response` (if any) and logs attempt number, delay, method, URL, and status/exception so behaviour matches current debug output.
- [x] 1.5 Create `_build_retrying_controller(...)` that receives the validated parameters and returns a configured `Retrying` instance with: combined retry predicate (`retry_if_exception_type` for `requests.Timeout`, `requests.ConnectionError`, generic `requests.RequestException` plus `retry_if_result` for retryable statuses), `wait=RetryAfterJitterWait(...)`, `stop=stop_after_attempt(max_retries + 1)` plus optional `stop_after_delay(max_retry_duration)`, `sleep=TENACITY_SLEEP`, `reraise=True`, `before_sleep` hooks (both `_before_sleep_close_response` and `before_sleep_log(LOGGER, logging.DEBUG)`), and `retry_error_callback` that returns the last `requests.Response` when retries are exhausted by HTTP statuses.
- [x] 1.6 Refactor `request_with_retries` to: (a) perform existing parameter validation; (b) coerce timeout tuples; (c) resolve the callable (`session.request` vs `session.<method>`); (d) invoke the Tenacity controller; (e) enforce content policy on the final response; (f) emit a debug summary containing attempt counts and elapsed wait.
- [x] 1.7 Ensure Tenacity uses custom logic to treat exhausted HTTP retries as success (returning the last response) while truly exceptional exhaustion re-raises (`RetryError` wrapping a `RequestException`), and honour the `respect_retry_after` flag for skipping header-based waits.
- [x] 1.8 Remove the manual retry loop, `_calculate_equal_jitter_delay`, the `random` import, and any stale helper references; update docstrings/NAVMAP to describe the Tenacity-backed implementation.
- [x] 1.9 Update module-level typing/comments so linters know `TENACITY_SLEEP` is intentionally patchable for tests.

## 2. Align Callers with the Tenacity Entry Point
- [x] 2.1 Review `download.stream_candidate_payload`, `download.prepare_candidate_download`, and other downloader helpers to ensure they rely solely on the refactored `request_with_retries` (aside from the existing one-shot streaming retry), removing legacy jitter references and verifying context-manager behaviour remains correct.
- [x] 2.2 Update `head_precheck` and `_head_precheck_via_get` to pass explicit retry caps/timeouts into the Tenacity helper, confirm HEAD attempts use `max_retries=1`, and ensure responses are closed in `finally` blocks.
- [x] 2.3 Switch `RobotsCache._fetch` to call the Tenacity helper (with `max_retries=1`) and document why robots lookups still parse empty policies on failure.
- [x] 2.4 Delete proxy wrappers such as `resolvers/wayback.request_with_retries` and `resolvers/pmc.request_with_retries`; update those modules to import `DocsToKG.ContentDownload.networking.request_with_retries` directly.
- [x] 2.5 Audit every resolver module (`crossref`, `semantic_scholar`, `unpaywall`, `openaire`, `landing_page`, `wayback`, `pmc`, `figshare`, `europe_pmc`, `doaj`, `hal`, `zenodo`, `core`, etc.) for raw `session.get/head/request` usage, migrating each call to the shared helper or explicitly documenting safe exceptions (e.g., synchronous HTML parsing that never retries).
- [x] 2.6 Confirm configuration plumbing (`ResolverConfig`, `DownloadOptions`, CLI args) continues to feed values into the helper (`max_retries`, `backoff_factor`, `backoff_max`, `retry_after_cap`, `max_retry_duration`, `respect_retry_after`), renaming or extending fields (e.g., `max_retry_seconds`) if Tenacity semantics require clearer names, and update docstrings/tools accordingly.
- [x] 2.7 Remove any leftover imports or patches referencing `_calculate_equal_jitter_delay` or `networking.time.sleep` in pipeline/utilities/tests; ensure `__all__` and NAVMAP entries remain accurate after deletions.
- [x] 2.8 Verify `create_session` continues to mount adapters with `max_retries=0` to avoid double retry layers, adding comments or assertions if necessary.

## 3. Logging & Telemetry Consistency
- [x] 3.1 Ensure the new Tenacity hooks emit debug/warning logs equivalent to the current manual implementation (attempt number, status/exception, planned delay, cumulative attempts).
- [x] 3.2 Capture aggregate retry statistics (total attempts, total sleep) via `RetryCallState` or Tenacity metrics and surface them through existing logging or telemetry pathways so downstream analysis retains parity.
- [x] 3.3 When HTTP retries exhaust, log the same warning currently produced before returning the final response so operators notice degraded origins.

## 4. Tests & QA
- [x] 4.1 Expand `tests/content_download/test_networking.py` to cover: happy path, retry-on-status (with `retry_after_cap`), retry-on-exception, disabled `respect_retry_after`, `max_retry_duration` early termination, and ensuring prior responses are closed.
- [x] 4.2 Add unit checks that Tenacity leaves the final streaming response open (usable with `with`), while intermediate responses trigger their `close` method.
- [x] 4.3 Update resolver and pipeline tests (`tests/resolvers/test_resolvers_core.py`, `tests/content_download/test_pipeline.py`, property-based suites) to patch `networking.TENACITY_SLEEP` instead of `time.sleep` and to assert they call the shared helper rather than removed proxies.
- [x] 4.4 Add regression coverage for streaming retries to confirm Tenacity does not replay bodies after `iter_content` failures (simulate chunked encoding errors).
- [x] 4.5 Run the ContentDownload test matrix (`pytest -q tests/content_download/test_networking.py`, `test_runner_download_run.py`, `test_rate_control.py`, resolver suites) and capture exemplar logs showing Tenacity behaviour for QA artefacts.

## 5. Documentation & Runbooks
- [x] 5.1 Update `src/DocsToKG/ContentDownload/README.md` networking sections to describe the Tenacity policy, configurable knobs, and the patchable `TENACITY_SLEEP`.
- [x] 5.2 Expand `src/DocsToKG/ContentDownload/AGENTS.md` with instructions for adjusting retry parameters, guidance on patching Tenacity sleeps in tests, and a reminder that the networking module is the sole retry surface.
- [x] 5.3 Amend `LibraryDocumentation/Tenacity.md` with a “DocsToKG adoption” section linking to the new helper and documenting policy defaults.
- [x] 5.4 Mark `LibraryDocumentation/Tenacity_Transition_Plan.md` as completed (or archive it) and reference the openspec change for future readers.
- [x] 5.5 Update troubleshooting guides or ancillary docs (e.g., `requests-cache.md`, `url-normalize.md`) if they reference legacy jitter logic; ensure new examples show how to tune Tenacity.
- [x] 5.6 Add release notes/changelog entries summarising the Tenacity migration for operators.
