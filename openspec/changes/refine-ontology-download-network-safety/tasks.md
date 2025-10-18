# Implementation Tasks

## 1. URL Security Enforcement

- [ ] 1.1 Locate planner helpers that issue HTTP requests (`_fetch_last_modified`, `_populate_plan_metadata`, any siblings) and document current call graph.
- [ ] 1.2 Inject `validate_url_security(plan.url, active_config.defaults.http)` (or equivalent) before every planner-issued HTTP call.
- [ ] 1.3 Ensure blocked URLs raise the same exception class as download-time violations; update import paths accordingly.
- [ ] 1.4 Add unit tests covering:
  - [ ] 1.4.1 Allowed HTTPS URL passes validation and proceeds to HTTP request (mocked).
  - [ ] 1.4.2 Disallowed host raises security error and prevents the request from being issued.
  - [ ] 1.4.3 Invalid scheme (e.g., `file://`) triggers validation failure.
  - [ ] 1.4.4 Telemetry/log entries record the validation failure with planner context.
- [ ] 1.5 Update documentation/dev notes describing that planners now honour `validate_url_security`.

## 2. Polite Networking Integration

- [ ] 2.1 Identify existing polite networking primitives (`SessionPool`, `polite_http_headers`, `get_bucket`, telemetry hooks) used by download paths.
- [ ] 2.2 Design a planner probe helper (e.g., `planner_http_probe(url, config, *, method="HEAD")`) that:
  - [ ] 2.2.1 Acquires a session from `SessionPool`.
  - [ ] 2.2.2 Retrieves polite headers using `polite_http_headers`.
  - [ ] 2.2.3 Respects per-host rate limiting via `get_bucket`.
  - [ ] 2.2.4 Emits telemetry matching download probes.
- [ ] 2.3 Refactor `_fetch_last_modified` and `_populate_plan_metadata` to use the new helper instead of raw `requests` calls.
- [ ] 2.4 Ensure retries/backoff (if any) mirror the download pipeline; document deviations if exact parity is not feasible.
- [ ] 2.5 Add tests verifying:
  - [ ] 2.5.1 Headers include polite values (`User-Agent`, contact info) for planner requests.
  - [ ] 2.5.2 Planner probes respect rate limiting (simulate token bucket exhaustion and assert delayed/queued behaviour).
  - [ ] 2.5.3 Telemetry records planner probe attempts with expected fields.
  - [ ] 2.5.4 Session reuse occurs across consecutive planner probes (verify `SessionPool` instrumentation).
- [ ] 2.6 Update developer docs to highlight reuse of polite networking primitives.

## 3. Ontology Index Locking

- [ ] 3.1 Audit index maintenance functions (`_append_index_entry`, `write_index_json`, etc.) to understand current locking.
- [ ] 3.2 Introduce an ontology-scoped lock (new lock manager or reuse existing infrastructure) that serialises `index.json` writes across versions.
- [ ] 3.3 Wrap index write operations in the new lock while preserving version-level locking for artifact downloads.
- [ ] 3.4 Update logging to surface lock acquisition/wait events for observability.
- [ ] 3.5 Add concurrency regression tests:
  - [ ] 3.5.1 Simulate two threads updating different versions; assert both entries persist in resulting `index.json`.
  - [ ] 3.5.2 Stress test with repeated alternating updates and verify deterministic ordering or documented behaviour.
- [ ] 3.6 Document the locking hierarchy and potential throughput implications for overlapping backfills.

## 4. Telemetry & Documentation

- [ ] 4.1 Align telemetry schemas so planner probes emit the same (or clearly documented new) event type as downloads.
- [ ] 4.2 Update README/operator docs describing:
  - [ ] 4.2.1 URL validation during planning.
  - [ ] 4.2.2 Shared polite networking primitives.
  - [ ] 4.2.3 Ontology index locking behaviour.
  - [ ] 4.2.4 Planner probe opt-out configuration and trade-offs.
- [ ] 4.3 Add changelog entry summarising the network hygiene hardening.
- [ ] 4.4 Run `openspec validate refine-ontology-download-network-safety --strict` prior to requesting approval.

## 5. Planner Probe Opt-out

- [ ] 5.1 Design configuration surface for probe control (config file key + CLI flag).
- [ ] 5.2 Implement plumbing so planner checks the configuration after URL validation and short-circuits probe execution when disabled.
- [ ] 5.3 Ensure telemetry/logging records that probes were skipped intentionally and captures the controlling configuration value.
- [ ] 5.4 Add unit/integration tests covering enabled vs. disabled modes, verifying no HTTP requests occur when disabled.
- [ ] 5.5 Document default behaviour, recommended use cases, and operational caveats in configuration reference and operator guides.
