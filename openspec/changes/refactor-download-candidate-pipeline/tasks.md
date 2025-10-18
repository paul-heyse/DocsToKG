## 1. Decompose download workflow
- [ ] 1.1 Draft a phase diagram for the reworked `download_candidate`, enumerating preflight, cache reuse, streaming, and finalization entry/exit conditions; document it inline for reviewer context.
- [ ] 1.2 Create a helper module (or section) defining `prepare_candidate_download`, `stream_candidate_payload`, and `finalize_candidate_download` helpers plus lightweight dataclasses for their inputs/outputs.
- [ ] 1.3 Refactor `download_candidate` to orchestrate through the helpers, ensuring existing logging, metrics, and `DownloadStrategy` hooks continue to fire.
- [ ] 1.4 Update strategy code to consume any new helper context objects without duplicating logic (e.g. pass reused `DownloadStrategyContext` or new structs explicitly).
- [ ] 1.5 Add unit tests that exercise each helper in isolation (mocking HTTP/session) covering conditional GET reuse, HEAD preflight bypass when `skip_head_precheck=True`, and plain success path streaming.
- [ ] 1.6 Add an integration-style test around `download_candidate` validating that the orchestrated path still emits progress callbacks and manifest-ready outcomes.

## 2. Harden partial download cleanup
- [ ] 2.1 Ensure streaming abort paths (e.g., chunked encoding errors, client cancellations, content-policy skips) invoke `cleanup_sidecar_files` with the relevant classification hint before returning.
- [ ] 2.2 Gate the cleanup so that when range resume is explicitly enabled and supported we retain the partial file for retry (documented in code comments).
- [ ] 2.3 Write regression tests that create temporary artifact directories and confirm `.part` files disappear after streaming failures for PDF and HTML payloads.
- [ ] 2.4 Verify that successful retries continue to honor existing cache manifests (i.e., cleanup does not wipe successfully completed artifacts) and add assertions if needed.

## 3. Consolidate classification validation
- [ ] 3.1 Move all validation responsibilities into `build_download_outcome`, ensuring it normalizes reason/reason_detail fields for every MISS and CACHED branch.
- [ ] 3.2 Remove the secondary `validate_classification` call in `process_one_work`, updating runner logging to trust outcome metadata while maintaining telemetry payloads.
- [ ] 3.3 Update affected tests (runner, pipeline, manifest logger) to assert that reason codes arrive intact from the downloader instead of being post-processed.
- [ ] 3.4 Add a regression test ensuring HTML-tail rejection returns the expected reason and that the runner records it verbatim.

## 4. Unify download configurations
- [ ] 4.1 Introduce a unified `DownloadConfig` (name TBD) that encapsulates the shared option surface, providing `.to_context()` and `.from_options()` helpers for compatibility.
- [ ] 4.2 Migrate the CLI, runner, and pipeline constructors to use the new config while retaining backwards-compatible properties for existing call sites (e.g., `DownloadOptions` shims).
- [ ] 4.3 Ensure new config exposes domain policies, host accept overrides, skip-head flags, progress callbacks, and resume flags in one place; add property-based tests or fixtures to confirm round-tripping into `DownloadContext`.
- [ ] 4.4 Remove duplicated default logic between the old classes and document any deprecated attributes in module docstrings and `README.md`.
- [ ] 4.5 Audit telemetry/metrics code for assumptions about the old option types and update accordingly, including tests covering resume state persistence.
