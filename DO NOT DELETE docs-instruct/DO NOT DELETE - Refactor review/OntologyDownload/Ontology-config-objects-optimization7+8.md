Absolutely—here’s a **deep, code-free, agent-ready** plan for the **next two pillars**: (7) **Observability that answers questions** and (8) **Safety & policy, defense-in-depth**. I’ll anchor each to concrete files, event names, fields, gates, tests, and acceptance criteria so an AI programming agent can implement this end-to-end with minimal ambiguity.

---

# 7) Observability that answers questions (not just logs)

## 7.1 Objectives (what “done” means)

* One **event schema** used everywhere (network, download, extract, validate, storage, DB, CLI).
* **Single source of truth** for events in *both* JSON logs (console/file) and the DuckDB `events` table (or Parquet), so queries and dashboards are instant.
* **Correlatable runs**: every event carries `run_id`, `config_hash`, and a **stable causal chain** (`request_id`, `artifact_id`, `file_id`, `version_id`).
* Built-in **answers**: CLI & SQL that report deltas, health, capacity, and top offenders without reading raw logs.

## 7.2 Files & module layout

* `observability/events.py` – canonical **Event model** + helpers (`emit()`, `context()`, IDs).
* `observability/emitters.py` – sinks (stdout JSON, file JSONL, DuckDB appender, optional Parquet).
* `observability/schema.py` – JSON Schema for events (rendered to docs).
* Instrumentation shims:

  * `network/instrumentation.py` – `net.request` events
  * `ratelimit/instrumentation.py` – `ratelimit.acquire` events
  * `extract/instrumentation.py` – `extract.*` events
  * `storage/instrumentation.py` – `storage.*` events
  * `catalog/instrumentation.py` – `db.*` events
* `cli/obs_cmd.py` – `ontofetch obs tail|stats|export`

## 7.3 Event schema (canonical)

**Top-level fields present in *every* event**:

* `ts` (UTC ISO 8601), `type` (namespaced), `level` (INFO|WARN|ERROR)
* `run_id`, `config_hash`, `service`, `version_id` (nullable), `artifact_id` (nullable), `file_id` (nullable)
* `context` (dict): `{host, pid, app_version, python, os, libarchive_version}`
* `payload` (dict): event-specific fields (see below)

**Core event types** (all lowercase, dot-separated; add only from a central registry):

* **Lifecycle**: `extract.start`, `pre_scan.done`, `extract.done`, `extract.error`, `audit.emitted`
* **Network**: `net.request` (one per HTTP call)
* **Rate limits**: `ratelimit.acquire`, `ratelimit.cooldown`
* **Storage**: `storage.put.start|done|error`, `storage.move|copy_delete`, `storage.latest.set`
* **DB**: `db.migrate.applied`, `db.tx.commit|rollback`, `db.backup.done`
* **CLI**: `cli.command.start|done|error` (command, args redacted as needed)

**`net.request.payload` shape** (example):

```
{
  "method": "GET", "url_redacted": "...", "host": "…",
  "status": 200, "attempt": 1,
  "http2": true, "reused_conn": true,
  "cache": "hit|revalidated|miss|bypass",
  "ttfb_ms": 23.1, "elapsed_ms": 71.3, "bytes_read": 8192
}
```

**`extract.done.payload`** (example):

```
{
  "entries_total": 437, "entries_included": 281, "entries_skipped": 156,
  "bytes_declared": 128734662, "bytes_written": 93455672,
  "ratio_total": 6.12, "duration_ms": 1843,
  "space": {"available_bytes": 4.2e+11, "needed_bytes": 1.03e+08, "margin": 1.1}
}
```

**Error taxonomy**
All `*.error` events carry `payload.error_code` (e.g., `E_TRAVERSAL`, `E_ENTRY_RATIO`, `E_NET_CONNECT`, `E_STORAGE_MOVE`) and a compact `details` dict (never secrets). (Reuse the codes you standardized earlier—one catalog, one meaning.)

## 7.4 Emission & sinks

* **Emit API**:

  * `events.emit(type, level, payload, **ids)` – validates against the Event model, injects context, writes to registered sinks.
  * `events.context(run_id, config_hash, service, version_id, artifact_id, file_id)` – returns a *frozen* context binder for hot paths to avoid arg repetition.
* **Sinks** (pluggable; on by default):

  * JSON to stdout (for docker / journalctl)
  * Append to DuckDB `events` (batch appender per command) **or** Parquet (if `settings.db.parquet_events=true`)
  * Optional file JSONL (for dev; rotates daily)
* **Back-pressure**: buffering channel with drop strategy for DEBUG noise; INFO/WARN/ERROR are lossless.

## 7.5 End-to-end instrumentation points

* HTTPX hooks → `net.request`
* Rate-limit `acquire()` → `ratelimit.acquire` (blocked_ms, outcome), plus `ratelimit.cooldown` on 429 cool-downs
* Extractor milestones → `extract.start|pre_scan.done|done|error`
* Storage ops → `storage.*` with path_rel, bytes, ms
* DB boundaries → `db.tx.*`, `db.migrate.applied`, `db.backup.done`
* CLI wrapper → `cli.command.*` capturing command name, duration, exit code (args scrubbed/redacted)

## 7.6 Queries that *answer questions* (ship with product)

* **CLI** (`ontofetch obs stats`) and DuckDB canned queries:

  1. **SLO summary**: p50/p95 for `net.request.elapsed_ms` by `service`, cache hit ratio, error rate buckets.
  2. **Top offenders**: which hosts/services cause most `ratelimit.blocked_ms` and `extract.error`?
  3. **Throughput**: bytes_written over time by service, top 20 archives by size.
  4. **Stability**: error_code histogram over last N runs.
  5. **Delta validation**: `delta summary A B` cross-checked with `extract.done` bytes and counts.

## 7.7 Privacy & redaction

* **Never** log Authorization or cookies.
* `url_redacted`: strip query strings unless a safelist param is explicitly whitelisted.
* Allow a `--no-emit` mode (tests) and `settings.telemetry.emit_events=false`.

## 7.8 JSON Schema & docs

* Generate **event JSON Schema** (validation for sinks) and publish in `docs/schemas/events.schema.json`.
* `ontofetch obs export --schema` prints it; CI asserts the checked-in schema matches the generator.

## 7.9 Testing

* **Unit**: Event model validation; sink fan-out; drop strategy for DEBUG; redaction rules.
* **Integration**: One full `plan → pull → extract → validate → latest` run asserts event counts and a few aggregates (bytes, durations).
* **Performance**: ensure event emission adds < 2% overhead on hot paths (measure with/without).

## 7.10 Acceptance checklist

* [ ] Canonical Event model + JSON Schema shipped; sinks wired.
* [ ] All subsystems emit namespaced events with `run_id` and `config_hash`.
* [ ] Error taxonomy used consistently; one `*.error` per failure.
* [ ] CLI `obs stats|tail|export` answers the 5 stock questions above.
* [ ] Emission overhead negligible; privacy redaction verified.

---

# 8) Safety & policy, defense-in-depth

> Goal: make **every boundary** a *gate* with explicit policy, consistent errors, clear telemetry—and zero bespoke “ad-hoc” checks scattered around.

## 8.1 Safety architecture (“policy as code”)

* `policy/registry.py` – a **single registry** of gates (URL, HTTP, filesystem, extraction, storage, DB).
* `policy/contracts.py` – typed contracts (inputs/outputs) for each gate.
* `policy/errors.py` – one error catalog + helpers (builds `E_*` codes, adds context).
* `policy/metrics.py` – per-gate counters (passes, rejects, ms).
* Each gate returns either `OK(policy_result)` or `Reject(error_code, details)`; callers **never** raise or format policy errors manually—only through central helpers.

## 8.2 Gates (where to enforce & what)

1. **Configuration gate** (on startup / `settings validate`)

   * Enforce strict types, bounds, enum validity, normalized values
   * Output `config_hash`, `normalized_allowed_hosts`, `allowed_port_set`
2. **URL & network gate** (pre-request; per hop on redirects)

   * Scheme in {http, https}; userinfo forbidden
   * Host punycoded; allowlist match (exact/suffix/IP/CIDR); per-host ports; global port set
   * HTTP→HTTPS upgrade unless `allow_plain_http_for_host_allowlist`
   * DNS resolution per `strict_dns`; private/loopback blocked unless allowed
   * **Redirect audit**: each hop re-validated; never forward auth across hosts
3. **Filesystem & path gate** (before writing anything)

   * Encapsulation root; **dirfd/openat** semantics; `O_NOFOLLOW|O_EXCL`
   * Path normalization to NFC; reject `..`, absolute paths, Windows drive letters
   * Casefold collision detection (reject or allow policy)
   * Length/depth constraints; reserved names on Windows; trailing dot/space
4. **Extraction policy gate** (pre-scan, then per entry)

   * Entry type allowlist: regular files; reject symlink/hardlink/device/FIFO/socket
   * Zip-bomb guards: global (`max_total_ratio`) and per-entry (`max_entry_ratio`)
   * Per-file size guard; entry count guard; include/exclude globs
   * Permissions normalization (0644 files / 0755 dirs); strip suid/sgid/sticky
5. **Storage gate**

   * `LATEST.json` only written atomically (temp+move); object stores perform copy+delete fallback (if used later)
   * Path traversal blocked in `remote_rel`; rename guarded
6. **DB gate (transactional invariants)**

   * Boundary choreography: **only commit DB after FS success** (never torn)
   * Foreign key invariants enforced in code: artifact/version/file existence
   * Latest pointer and marker must match; doctor catches and resolves drifts

## 8.3 Policy objects & override strategy

* `ExtractionPolicy`, `UrlSecurityPolicy`, `StoragePolicy`, `DbBoundaryPolicy` – immutable, built from `Settings`.
* Presets: `StrictPolicy` (default), `PartnerPolicy` (more permissive for curated sources).
* CLI `--policy partner` flips only the intended knobs (documented). No hidden side effects.

## 8.4 Error catalog (one list; used everywhere)

* Network/TLS: `E_NET_CONNECT`, `E_NET_READ`, `E_NET_PROTOCOL`, `E_TLS`
* URL/DNS: `E_SCHEME`, `E_USERINFO`, `E_HOST_DENY`, `E_PORT_DENY`, `E_DNS_FAIL`, `E_PRIVATE_NET`
* Redirect: `E_REDIRECT_DENY`, `E_REDIRECT_LOOP`
* Filesystem/paths: `E_TRAVERSAL`, `E_CASEFOLD_COLLISION`, `E_DEPTH`, `E_SEGMENT_LEN`, `E_PATH_LEN`, `E_PORTABILITY`
* Extraction: `E_SPECIAL_TYPE`, `E_BOMB_RATIO`, `E_ENTRY_RATIO`, `E_FILE_SIZE`, `E_FILE_SIZE_STREAM`, `E_ENTRY_BUDGET`
* Storage: `E_STORAGE_PUT`, `E_STORAGE_MOVE`, `E_STORAGE_MARKER`
* DB/boundaries: `E_DB_TX`, `E_DB_MIGRATION`, `E_LATEST_MISMATCH`

All surfaced through `policy.errors.raise_with_event(error_code, details)` which:

* Emits a `*.error` event with the error code & details
* Raises the correct exception class (`PolicyError`, `ConfigError`, `IOError`, …)

## 8.5 Defense-in-depth specifics (beyond extraction)

* **Network**: global client defaults (redirects off; SNI; min TLS 1.2), per-service **auth plugins**; cookies disabled by default; prevent credential leakage across hosts.
* **Rate limits**: (already built) fail-fast vs block modes; 429 cool-downs to reduce blast radius.
* **DB**: `doctor` & `prune` are *safe by default* (dry-run, confirmation, clear exit codes). Deletions performed in small batches with rollback safety.
* **CLI**: destructive ops require `--yes` or typed confirmation; `--dry-run` everywhere.

## 8.6 Telemetry for policy

* **For each gate**: emit `policy.gate` event with `{gate, outcome, ms, error_code?}`.
* Add counters: per gate `passed`, `rejected`, `avg_ms`.
* Include minimal, non-sensitive fields to help triage (e.g., host, path_rel basename, limits that tripped).

## 8.7 Testing strategy

* **Unit** (gate-by-gate): white-box tests for each policy rule (both accept and reject), including edge cases (IDNs, CIDRs, Windows names).
* **Property-based**:

  * URL host/port generators → no false positives/negatives; normalization and match logic idempotent.
  * Path generators (Unicode combining marks, bidi, deep trees) → invariants hold; no escapes; collisions detected.
* **Integration**:

  * Harvest end-to-end scenarios that purposely trigger each error code; assert single `*.error`, correct CLI exit code, and zero partial writes.
* **Cross-platform**:

  * Windows runners for reserved names/long path policies; macOS runners for NFD/NFC normalization.
* **Chaos**:

  * Crash between FS write and DB commit → doctor offers fix; no data loss; invariants restored.

## 8.8 Acceptance checklist

* [ ] **One** policy registry & error catalog used by all subsystems (no ad-hoc validation).
* [ ] URL/redirect gate validates every target; private/loopback blocked per policy.
* [ ] Filesystem gate enforces **dirfd/openat + O_NOFOLLOW|O_EXCL**; no symlink/device extraction.
* [ ] Extraction gate applies global & per-entry ratio, file size, entry budget, include/exclude globs.
* [ ] Storage & DB boundaries guarantee **no torn writes**; latest pointer & marker always in sync or doctor flags mismatch.
* [ ] Every rejection emits a structured `*.error` with an `E_*` code and minimal details.
* [ ] Comprehensive, cross-platform tests pass; property tests cover tricky inputs.

## 8.9 Suggested PR sequence

1. **PR-P1**: Policy registry & error catalog; gate contracts; minimal URL gate + tests.
2. **PR-P2**: Filesystem/path gate (encapsulation+openat); extraction pre-scan guards; tests.
3. **PR-P3**: Storage & DB gates; boundary choreography; doctor fixes; tests.
4. **PR-P4**: Telemetry for gates; CLI error mapping; cross-platform CI lanes; property tests.

---

## What you gain

* **Observability** that surfaces the *answers*, not just raw logs—embedded in your product with ready-made dashboards and queries.
* **Safety** that is explicit, uniform, and auditable—policy is code, gates are centralized, and every rejection tells you *why*, *where*, and *how long* it took to decide.
