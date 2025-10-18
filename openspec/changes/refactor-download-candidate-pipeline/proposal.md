## Why
The download review surfaced systemic problems in `download_candidate`: a single ~900 line function is attempting validation, cache negotiation, streaming, classification, and bookkeeping. That sprawl makes the hot path impossible to reason about, leaves partial `.part` files behind whenever streaming aborts short circuit the workflow, duplicates classification validation further up the stack, and creates configuration drift between `DownloadOptions` and `DownloadContext`. We need targeted structural work before implementation changes become safe or testable.

## What Changes
- Factor the download workflow into focused helpers (preflight/cache reuse, streaming & persistence, and post-processing) so we can test and profile each stage independently while keeping existing telemetry intact.
- Ensure partial artifacts are cleaned up whenever downloads abort early, including policy-triggered skips, by reusing the existing sidecar cleanup tooling.
- Trust the downloader’s outcome classification and remove redundant validation in the runner while keeping manifest logging consistent with downloader-provided reasons.
- Collapse `DownloadOptions` and `DownloadContext` into a single source of truth so new flags (e.g. `host_accept_overrides`, `skip_head_precheck`, `progress_callback`) automatically flow through CLI, pipeline, and resumable runs without manual mapping.
- Retire global max-byte limit switches (CLI flags, context fields, ontology config) so downloads rely on warning thresholds and policy skips instead of hard aborts.
- Backfill regression tests around the new helpers, manifest propagation, and cleanup path to guarantee behaviour stays intact.

## Impact
- Affected specs: content-download
- Affected code: `src/DocsToKG/ContentDownload/download.py`, `runner.py`, `core.py`, resolver pipeline tests
