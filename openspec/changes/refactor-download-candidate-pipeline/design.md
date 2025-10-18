## Context
- `download_candidate` currently interleaves preflight checks, conditional cache evaluation, streaming, progress reporting, MIME detection, and post-processing in a single ~900 line function. This structure obstructs targeted tests and makes it difficult to reason about edge cases such as conditional GET, range resumes, or classification overrides.
- `_MaxBytesExceeded` raises during streaming leave `.part` files in the artifact directories because the error path returns immediately without invoking `cleanup_sidecar_files`.
- `build_download_outcome` already validates classifications, but `process_one_work` re-runs `validate_classification`, risking divergence and requiring duplicated manifest massaging.
- `DownloadOptions` (CLI surface) and `DownloadContext` (runtime surface) expose overlapping but non-identical fields. Options misses knobs such as `domain_content_rules`, `host_accept_overrides`, and `skip_head_precheck`, so new capabilities must be wired manually in multiple places.

## Goals / Non-Goals
- Goals:
  - Decompose `download_candidate` into cohesive helpers that encapsulate preflight/caching, streaming & hashing, and outcome assembly.
  - Guarantee partial artifacts are cleaned when streaming stops prematurely because of size caps or policy violations.
  - Rely on downloader-level classification validation so runner code treats the returned outcome as authoritative.
  - Introduce a unified download configuration object (or wrapper) that ensures CLI and pipeline share the same field surface without duplication.
  - Add unit/functional tests covering new helper boundaries and partial cleanup semantics.
- Non-Goals:
  - Re-enable range resume or change retry semantics.
  - Adjust resolver ordering logic or add new resolver types.
  - Redesign telemetry/metrics collection beyond necessary call-site updates.

## Decisions
- Decision: Introduce a `DownloadExecution` helper module that exposes `prepare_download`, `run_stream`, and `finalize_download` functions, with `download_candidate` acting as the orchestrator. Each helper will receive a narrowed context struct (`DownloadPreflight`, `DownloadStreamResult`, etc.) so we can test conditional flows independently and keep the main function declarative.
  - Alternatives considered: convert `download_candidate` into a class with methods; rejected for now because stateless helpers keep call-sites simpler and avoid lifecycle concerns.
- Decision: On `_MaxBytesExceeded`, call `cleanup_sidecar_files` with the current classification hint (PDF/HTML/XML) before returning the size-limit outcome. The helper already deletes the `.part` suffix files and respects dry-run semantics.
  - Alternatives considered: toggle `keep_partial_on_error=False`. Rejected because we still want partial persistence for the retry-on-error branch once resume support lands; explicit cleanup keeps behaviour visible.
- Decision: Move classification validation entirely into `build_download_outcome` and ensure the result carries authoritative `reason`/`detail` fields, including cache hits and policy rejections. `process_one_work` will stop re-validating and simply trust/respect the returned outcome.
- Decision: Replace `DownloadOptions` with a thin dataclass that wraps `DownloadContext` (e.g. exposes `.to_context()`) or merge the two into a single dataclass with convenience constructors. The unified object will list every supported knob once, and CLI arguments feed into it directly.
  - Alternatives considered: keep both types but add synchronised field copy logic. Rejected because manual sync is the source of the existing drift.

## Implementation Notes
- Helper orchestration: `download_candidate` will resemble a template method—call `prepare_download` (wrapping HEAD preflight, conditional caching, resume setup), branch on cache hit, then hand control to `stream_candidate_payload`, which manages progress callbacks, range headers, hashing, and raising `_MaxBytesExceeded`. `finalize_candidate_download` will collate strategy outputs, invoke `build_download_outcome`, and manage manifest state.
- Dataclasses: Introduce `DownloadPreflightResult` (holding headers, status, conditional helpers, destination path decisions) and `StreamExecutionResult` (bytes written, hash, tail window, retry hints) to limit shared mutable state. These keep function signatures tight and ease unit testing.
- Testing approach: add new tests under `tests/contentdownload/test_download_execution.py` that mock `requests.Response`, `ConditionalRequestHelper`, and filesystem writes to validate each helper. Complement with an integration test using `responses` (or local stub) ensuring progress callbacks fire at the expected intervals and the final outcome matches the original behaviour.
- Cleanup semantics: `_MaxBytesExceeded`, policy-based aborts, and other stream interruptions will call `cleanup_sidecar_files`. When resume support returns, the helper will check `ctx.enable_range_resume` before deleting `.part` files; tests will cover both branches.
- Configuration unification: Implement a `DownloadConfig` (name TBD) with `from_cli_args`, `to_context`, and `apply_manifest_resume` helpers. `DownloadOptions` becomes a compatibility shim returning a `DownloadConfig`, emitting deprecation warnings if direct mutation occurs.
- Metrics/telemetry: review `runner.py`, `telemetry.py`, and `statistics.py` for references to `DownloadOptions` attributes; update to pull from `DownloadConfig` to preserve logging payloads.
- Manifest flow: ensure the new orchestration continues to populate `strategy_context` for `DownloadStrategy.finalize_artifact` so resolver metrics stay accurate.

- Risk: Splitting `download_candidate` could accidentally change side effects. Mitigation: add regression tests for conditional GET success, size-limit failure, timeout errors, and cache reuse, and keep helper signatures narrow to minimise accidental behaviour changes.
- Risk: Unifying configuration objects might break callers expecting legacy attribute names. Mitigation: provide compatibility properties or shims (e.g. keep `DownloadOptions` as a subclass with deprecated aliases) and cover with integration tests.
- Risk: Cleanup on `_MaxBytesExceeded` might conflict with future resume work. Mitigation: ensure helper only deletes partial file when resume is disabled or unsupported; document behaviour in code.

## Migration Plan
1. Introduce the new helper module/structure and route `download_candidate` through it without changing observable behaviour; add unit tests for each helper and an integration test for the orchestrator.
2. Wire `_MaxBytesExceeded` and related abort paths to `cleanup_sidecar_files`, with logic conditioned on resume settings, and add filesystem-backed regression tests.
3. Remove duplicate validation in `process_one_work`, updating tests that expected overridden reason codes and verifying manifests/metrics receive downloader-provided reasons.
4. Implement the unified download configuration object, adjust CLI/pipeline constructors, and deprecate any legacy fields; update telemetry code and fixtures to reference the new config.
5. Update docs (`README.md`, module docstrings) to describe the new configuration path and helper phases; run the full ContentDownload test suite.

## Open Questions
- Do we need a compatibility window where both `DownloadOptions` and a new unified object coexist, or can we migrate callers in one change?
- Should progress callbacks become part of the unified configuration surface, and do we need additional tests around them when breaking up the streaming logic?
