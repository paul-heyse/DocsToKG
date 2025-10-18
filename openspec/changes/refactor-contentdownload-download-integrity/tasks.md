# Implementation Tasks

## 1. Reason Code Corrections

- [ ] 1.1 Audit `_build_download_outcome`, `DownloadOutcome`, and telemetry writers to map current reason code defaults for each artifact type.
- [ ] 1.2 Update `_build_download_outcome` to only set `ReasonCode.CONDITIONAL_NOT_MODIFIED` when the HTTP status (or resolver outcome) is conditional (304, etag match, `download_skipped=True` with conditional flag).
- [ ] 1.3 Ensure `reason_code` remains `None` for new successful downloads, including non-PDF artifacts and resolver fallbacks.
- [ ] 1.4 Update any helper functions (`_download_pdf_like`, `_download_html_like`, etc.) that stamp reason codes to defer to the corrected logic.
- [ ] 1.5 Add regression unit tests covering: fresh download → `None`, conditional 304 → `CONDITIONAL_NOT_MODIFIED`, retry success → `None`, hard failure → existing failure code.
- [ ] 1.6 Verify telemetry serialization (`DownloadOutcome.as_dict`, JSONL writer, SQLite sink) gracefully emits `null` when `reason_code` is `None`.
- [ ] 1.7 Update manifest-writing logic so persisted outcomes align with corrected reason codes.
- [ ] 1.8 Review downstream analytics consumers (internal dashboards, notebooks) and document any changes needed to interpret the corrected reason codes.
- [ ] 1.9 Update release notes / changelog to highlight the telemetry correction.

## 2. Range Resume Deprecation

- [ ] 2.1 Trace how `allow_resume` is set (CLI flags, config files) and catalogue call sites that depend on resume semantics.
- [ ] 2.2 Hard-disable resume paths: ensure range requests are never issued and atomic writes always perform full downloads.
- [ ] 2.3 Emit a structured warning when callers request resume to communicate its deprecation.
- [ ] 2.4 Remove or ignore resolver hints that currently advertise resume capability.
- [ ] 2.5 Update telemetry to record that resume was disabled (e.g., `reason_code=None`, tag `resume_disabled=true`) for observability.
- [ ] 2.6 Add regression test simulating interrupted downloads verifying the system re-fetches the file fully and preserves hash integrity.
- [ ] 2.7 Document the deprecation in CLI help, user guides, and changelog entries.

## 3. Voluntary Skip Classification

- [ ] 3.1 Introduce new enum entry `ReasonCode.SKIP_LARGE_DOWNLOAD` (or agreed name) in `ContentDownload/download.py`.
- [ ] 3.2 Update the branches enforcing `skip_large_downloads` to emit the new reason while keeping `DOMAIN_MAX_BYTES` for host-level thresholds.
- [ ] 3.3 Adjust manifest logging, telemetry sinks, and metrics aggregation code to recognize the new reason value.
- [ ] 3.4 Add unit test verifying skip events increment `SKIP_LARGE_DOWNLOAD` metrics but leave domain budget counters unchanged.
- [ ] 3.5 Update dashboards or analytics queries (documented in repo) to report voluntary skips separately.
- [ ] 3.6 Document the behavioral change in release notes and CLI reference.

## 4. Manifest Warm-Up Optimization

- [ ] 4.1 Profile current startup path to capture baseline memory/time impact of `load_manifest_url_index` on large manifests.
- [ ] 4.2 Refactor `resolve_config` so it no longer unconditionally calls `load_manifest_url_index`.
- [ ] 4.3 Extract a lazy manifest accessor (generator or paginated query helper) that retrieves entries on demand.
- [ ] 4.4 Ensure the new accessor still de-duplicates URLs and enforces normalization rules previously handled in-memory.
- [ ] 4.5 Add CLI/config flag to opt into eager warm-up (`--warm-manifest-cache`), defaulting to lazy behavior.
- [ ] 4.6 Update callers (download loop, manifest lookup) to consume the lazy accessor without materializing the full table.
- [ ] 4.7 Add unit tests covering lazy iteration, pagination boundaries, and opt-in eager warm-up.
- [ ] 4.8 Benchmark startup with ≥250k-row manifest before/after change and record results in proposal notes.
- [ ] 4.9 Ensure SQLite connections are managed safely when iterating lazily (connection reuse, closing cursors).
- [ ] 4.10 Update documentation explaining new flag and performance characteristics.

## 5. Cache Validation Fast-Path

- [ ] 5.1 Map all entry points that call `_validate_cached_artifact` (conditional hits, manifest verification) to understand expectations.
- [ ] 5.2 Implement short-circuit logic: compare file size and mtime before computing SHA-256.
- [ ] 5.3 Add guard to skip digest recomputation if size/mtime match cached metadata.
- [ ] 5.4 Introduce configuration toggle (`DownloadOptions.verify_cache_digest`) defaulting to `False` that forces digest verification when needed.
- [ ] 5.5 Add in-memory bounded cache (LRU) for recent digest results to avoid redundant hashing within a run.
- [ ] 5.6 Update telemetry to record whether validation occurred via fast-path or digest verification (for observability).
- [ ] 5.7 Add unit tests covering fast path hit, digest forced, digest mismatch leading to re-download, and cache eviction.
- [ ] 5.8 Benchmark validation on large artifacts (e.g., 100 MB PDF) to confirm runtime reduction.
- [ ] 5.9 Document trade-offs of turning on digest verification in administrator guide.

## 6. Spec and Documentation Updates

- [ ] 6.1 Update `openspec/changes/refactor-contentdownload-download-integrity/specs/content-download/spec.md` with new/modified requirements and scenarios.
- [ ] 6.2 Cross-reference telemetry changes in `specs/telemetry` (if applicable) or note why no change is needed.
- [ ] 6.3 Update architectural docs (`docs/ContentDownloadReview.md`, README snippets) to reflect new behaviors.
- [ ] 6.4 Add changelog entry summarizing telemetry, resume, and performance fixes.

## 7. Validation and Rollout

- [ ] 7.1 Run unit and integration test suites focusing on ContentDownload (ensure virtualenv activated).
- [ ] 7.2 Add new regression test cases to CI to guard against metric regressions (e.g., asserting reason-code distribution).
- [ ] 7.3 Execute manual end-to-end run with sample manifest exercising resume, skip, and conditional paths; capture telemetry output for review.
- [ ] 7.4 Validate performance benchmarks recorded in Sections 4 and 5 meet success criteria.
- [ ] 7.5 Prepare migration guidance for operators (including resume flag behavior and new reason code).
- [ ] 7.6 Request maintainer review, incorporate feedback, and secure approval before implementation proceeds.
- [ ] 7.7 Run `openspec validate refactor-contentdownload-download-integrity --strict` to confirm spec compliance prior to submission.
