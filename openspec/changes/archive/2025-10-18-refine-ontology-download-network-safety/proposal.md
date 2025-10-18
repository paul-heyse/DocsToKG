# Harden OntologyDownload Network Hygiene and Concurrency

## Why

Targeted review of `src/DocsToKG/OntologyDownload` identified security, policy-alignment, and concurrency gaps that the current architecture change set does not address:

1. **Planner metadata probes bypass URL security**: `_fetch_last_modified` and related helpers call `requests.head`/`requests.get` on `plan.url` without first passing the URL through `validate_url_security`. As a result, planner executions may contact disallowed or private hosts that the download pipeline would later reject, duplicating risk around credential leakage and policy violations.
2. **Planner HTTP probes ignore polite networking primitives**: The planner path spins up ad-hoc `requests` calls, ignoring `polite_http_headers`, `SessionPool`, and the per-host rate limiting already enforced during artifact downloads. This breaks throttling guarantees, can overwhelm remote APIs when multiple planners run, and slips telemetry coverage.
3. **Ontology index updates race across versions**: `_append_index_entry` rewrites `index.json` using only the `(ontology_id, version)` lock. Concurrent runs for different ontology versions can interleave and drop entries because no ontology-level guard exists.

Realigning planner behaviour with existing download safeguards and strengthening index coordination will prevent policy regressions, keep telemetry consistent, and avoid data loss when running concurrent backfills.

## What Changes

### 1. Apply URL security validation to planner probes

- Update `_fetch_last_modified` and `_populate_plan_metadata` to invoke `validate_url_security(plan.url, active_config.defaults.http)` before issuing any HTTP request.
- Ensure failures raise the same structured exception type used by download paths, preventing planners from masking policy violations.
- Add unit tests covering approved URLs, disallowed schemes/hosts, and telemetry/log output for blocked probes.
- Document in developer notes that planner metadata requests now honour the same URL allow/deny lists as downloads.

### 2. Reuse polite networking primitives for planner HTTP calls

- Refactor planner probe logic to acquire a pooled session from `SessionPool`, derive headers from `polite_http_headers`, and wrap requests with the existing rate limiter obtained via `get_bucket`.
- Introduce a small helper (`planner_http_probe`) that takes a URL and returns `(response, headers)` while recording telemetry consistent with download probing.
- Ensure planner probes reuse per-host buckets so concurrent planners respect customer-configured rate caps.
- Expand tests (unit + integration) to verify header injection, bucket acquisition, telemetry emission, and retry semantics mirror the download pipeline.

### 3. Add ontology-scoped locking for index maintenance

- Introduce an ontology-level lock (e.g., `index_lock = LockSet("ontology-index")`) that serialises `index.json` writes across all versions.
- Wrap `_append_index_entry` (and any other index writers) in the new lock while retaining the existing version locks for artifact materialisation.
- Add regression tests simulating concurrent updates for different versions to confirm entries are preserved and ordering is stable.
- Update documentation for operators describing the locking hierarchy and its impact on throughput when backfills overlap.

### 4. Allow planner probing opt-out for restricted environments

- Add a configuration switch/CLI flag (e.g., `--disable-planner-probes` or `settings.planner.probing_enabled`) that short-circuits metadata requests after URL validation succeeds.
- Ensure the planner records telemetry/log entries noting that probing was skipped intentionally and defaults remain opt-in.
- When probing is disabled, planner metadata MUST be derived from cached manifest data or defaults without issuing network calls.
- Document the operational trade-offs (loss of freshness signals vs. compliance with strict network policies) and expose the setting in configuration reference materials.

## Impact

- **Primary capability**: `ontology-download`
- **Affected modules**: `OntologyDownload/planning.py`, `OntologyDownload/io/network.py` (or equivalent), `OntologyDownload/security.py` / `validation.py`, `OntologyDownload/index.py` (or `io/filesystem.py`) depending on lock placement, `tests/ontology_download/**`, telemetry helpers.
- **Security**: Planner metadata probes now respect URL and credential policies identical to download paths.
- **Networking**: Reuse of polite sessions guarantees consistent headers, rate limiting, and telemetry for all HTTP interactions.
- **Concurrency**: Index updates become robust against cross-version races, preventing silent omission of downloads.
- **Docs**: Update README / developer notes highlighting planner alignment with download safeguards, new locking behaviour, and probe opt-out configuration.

## Success Criteria

1. Planner metadata requests fail fast when `validate_url_security` rejects the URL; planners no longer contact unapproved hosts.
2. Planner HTTP probes reuse `SessionPool`, `polite_http_headers`, and per-host rate limit buckets; automated tests verify header injection and throttle behaviour.
3. Concurrent runs updating `index.json` for multiple versions preserve all entries with deterministic ordering; stress tests show no dropped records.
4. Telemetry captures planner probe attempts using the same event type and schema fields as download probes, including explicit markers when probing is disabled by configuration so downstream analytics remain consistent.
5. Documentation and operator guides describe the new behaviour and configuration knobs (rate limiter tuning, probe opt-out).

## Open Questions

- Will the new ontology-level lock introduce noticeable contention for large-scale backfills, and do we need metrics to monitor lock wait durations?
