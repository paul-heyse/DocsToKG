# Refine ContentDownload Reliability and Telemetry

## Why

A recent code review of `src/DocsToKG/ContentDownload` highlighted multiple defects that distort telemetry, risk data loss, and create unnecessary performance overhead:

1. `_build_download_outcome` marks every successful PDF-like download as `conditional_not_modified`, causing success metrics to masquerade as cache hits.
2. The range-resume pathway advertises HTTP range support, yet still writes resumed payloads via `atomic_write`, producing truncated artifacts that drop the already persisted prefix.
3. User-triggered `skip_large_downloads` events are logged as `DOMAIN_MAX_BYTES`, blending voluntary skips with enforced budget caps.
4. `resolve_config` eagerly materializes the entire manifest SQLite index, which does not scale when manifests hold hundreds of thousands of rows.
5. `_validate_cached_artifact` recomputes a full SHA-256 digest for every 304 validation, even when size and mtime already guarantee freshness.

Left unaddressed, these issues undermine downstream analytics, corrupt artifacts when resume is toggled, and lengthen startup/validation on large corpora.

## What Changes

### 1. Correct reason code assignment for successful downloads

- Update `_build_download_outcome` (and any wrappers) so that `ReasonCode.CONDITIONAL_NOT_MODIFIED` is only emitted for genuine conditional responses (HTTP 304 or equivalent resolver signals).
- Leave `reason_code` unset (`None`) for freshly downloaded artifacts; retain existing behavior for explicit error/skip states.
- Backfill unit tests covering success, conditional, retry, and failure outcomes to ensure reason code integrity.
- Review telemetry schemas (`DownloadOutcome.as_dict()` and emitters) to guarantee `None` reason codes serialize predictably.

### 2. Deprecate range resume to prevent truncation

- Audit the resume flag surface (`DownloadOptions.allow_resume`, CLI flag, resolver hints) and document the current caller expectations.
- Hard-disable range resume so the downloader always performs full fetches, even when the flag is enabled; emit a deprecation warning clarifying that resume is unsupported.
- Remove/disable any resolver hints that advertise partial content support.
- Add regression coverage that simulates interrupted downloads and confirms artifacts are re-fetched in full with matching hashes.

### 3. Separate voluntary skips from domain budget enforcement

- Introduce a dedicated reason classification (e.g., `ReasonCode.SKIP_LARGE_DOWNLOAD`) for user-initiated skips.
- Adjust download loop logic so `skip_large_downloads` surfaces the new reason, while genuine domain budget exceedances retain `DOMAIN_MAX_BYTES`.
- Update telemetry serialization, manifest logging, and downstream consumers to recognize the new reason value.
- Add reporting tests asserting that voluntary skip counts no longer increment domain budget metrics.

### 4. Avoid manifest warm-up blowups

- Refactor `resolve_config` (and helpers) to stop loading the full manifest index at startup.
- Decompose `load_manifest_url_index` into a paginated/lazy accessor so only required manifests are materialized (e.g., limit to the current run id or artifact type).
- Introduce a CLI flag or configuration hook to opt into eager warm-up for small manifests (preserving existing behavior for regression comparison).
- Benchmark cold-start on large manifests (>250k rows) to demonstrate improved startup latency and memory.

### 5. Short-circuit expensive cache validation

- Update `_validate_cached_artifact` to perform fast-path checks (size + mtime) before recomputing SHA-256.
- Gate digest recomputation behind a toggle (`DownloadOptions.verify_cache_digest`), defaulting to `False` but retaining opt-in for high-assurance deployments.
- Cache recent digests in-memory (bounded LRU) so repeated validations avoid re-reading large artifacts.
- Extend unit tests to cover size/mtime hits, digest fallback, and digest-on-demand scenarios; add perf guardrails for large fixtures.

## Impact

- **Primary capability**: `content-download`
- **Key modules**: `ContentDownload/download.py`, `ContentDownload/pipeline.py`, `ContentDownload/manifest.py?`, telemetry emitters, CLI option wiring.
- **Telemetry**: Metrics and manifest summaries reflect true success, skip, and conditional counts.
- **Reliability**: Range resume no longer produces truncated artifacts.
- **Performance**: Startup and repeated 304 validations scale with large manifests and cached PDFs.

## Success Criteria

1. Successful downloads emit `reason_code=None`, while 304/conditional flows emit `ReasonCode.CONDITIONAL_NOT_MODIFIED`.
2. Resume mode remains disabled regardless of flag settings; no truncated files in regression suite.
3. Voluntary skips log `ReasonCode.SKIP_LARGE_DOWNLOAD` (or equivalent) without incrementing domain budget counters.
4. Startup memory footprint and runtime remain bounded when manifest tables exceed 250k rows.
5. Cache validation benchmarks show avoided SHA-256 recomputation on repeated 304 hits while preserving opt-in verification.

## Open Questions

- Are downstream analytics consumers prepared for a new reason code value, or do we need a migration playbook?
- Which telemetry sinks (JSONL, SQLite, Prometheus) require schema/version updates to track the new reason?
- How prominently should the resume deprecation warning appear in CLI output and documentation?

## Resume Surface Audit

- `DownloadOptions.allow_resume` no longer exists; the only remaining toggle is `DownloadContext.enable_range_resume`, which is defaulted to `False` and forcefully cleared inside `download_candidate`.
- No CLI flag or configuration file sets `enable_range_resume`. Historical guidance (`OPTIMIZATION_QUICK_START.md`, `OPTIMIZATIONS_APPLIED.md`) has been updated to mark range resume as deprecated and ignored.
- Resolver metadata previously used to advertise resume (keys prefixed with `resume_`) is now stripped before telemetry emission so downstream consumers never see stale hints.

## Cache Validation Entry Points

`_validate_cached_artifact` is invoked solely inside `download_candidate` when conditional requests succeed. There are no secondary call sites (manifest verification, CLI utilities), so the fast-path guarantees cover every cache validation in the system.

## Benchmark Results

| Scenario | Runtime | Peak Memory | Notes |
| --- | --- | --- | --- |
| Baseline `load_manifest_url_index` (250k rows) | 0.72 s | 151 MB | Materialises entire manifest into memory |
| Lazy `ManifestUrlIndex` init | <1 ms | <0.01 MB | No rows loaded until first lookup |
| 10 lazy lookups (250k rows) | 0.18 s | 0.01 MB | Fetches rows on demand without global cache |
| `ManifestUrlIndex` eager mode | 0.71 s | 159 MB | Matches legacy behaviour via `--warm-manifest-cache` |
| Cache fast-path (100 MB artifact, mtime match) | <0.1 ms | <0.01 MB | Size+mtime short-circuit, digest skipped |
| Forced digest (same artifact) | 55 ms | 0.02 MB | SHA-256 recomputation when `verify_cache_digest=True` |

## Manual Verification

- Ran a synthetic end-to-end pipeline exercising fresh download, conditional hit, voluntary skip, and resume-requested flows (script in session history).
- Telemetry JSONL (`tmp/manual_run/manifest.jsonl`) captures:
  - Fresh download with `reason=null`.
  - Conditional response tagged `reason=conditional_not_modified`.
  - Voluntary skip tagged `reason=skip_large_download`.
  - Resume request annotated with `metadata.resume_disabled=true`.
- These runs confirm manifest persistence, attempt logging, and new reason codes behave as specified.

## Analytics & Reporting Updates

- Dashboard guidance now splits voluntary skips (`skip_large_download`) from domain quota enforcement and highlights the `resume_disabled` metadata tag for observability.
- Operators consuming SQLite/JSONL exports should treat missing reason codes as successful downloads and adjust aggregations accordingly; updated instructions live in `docs/07-reference/monitoring/index.md` and `docs/03-architecture/content-download-resolver-architecture.md`.
