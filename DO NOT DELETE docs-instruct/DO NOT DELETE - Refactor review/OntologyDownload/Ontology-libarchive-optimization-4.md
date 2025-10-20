Absolutely — here’s a **code-free, agent-ready implementation plan** for the remaining optimizations. I’ve kept the same structure you liked (Goal → Config & Data → Implementation Steps → Failure Modes → Telemetry → Tests), and I’ve added precise event/field names, units, and sequencing so an AI agent can wire everything up without guesswork.

---

# 4) Observability & Diagnostics

## 4.1 Structured metrics

**Goal**
Make every extraction diagnosable (success or failure) and capacity-plannable with consistent, machine-readable logs.

**Config & Data**

* No new user-facing config required.
* Internally standardize a **run context** captured once per archive:

  * `run_id` (UUIDv4), `archive_path`, `archive_sha256[:12]`, `encapsulated_root`, `format`, `filters`.

**Implementation Steps**

1. **Event names** (emit as structured logs with a `type` field):

   * `extract.start` (before pre-scan)
   * `extract.pre_scan.done` (after pre-scan succeeds)
   * `extract.extract.start` (right before first write)
   * `extract.extract.done` (after final rename)
   * `extract.error` (on any fatal error; one per failure)
   * `extract.audit.emitted` (if audit JSON written)

2. **Common fields** (included in *every* event):

   * `run_id`, `archive` (absolute path), `archive_sha256`, `encapsulated_root` (absolute path), `format`, `filters` (array), `duration_ms` (from `extract.start`).

3. **Metric fields**

   * Pre-scan counters (emit at `pre_scan.done`):

     * `entries_total`, `entries_included`, `entries_skipped`,
     * `bytes_declared` (sum of included entries’ uncompressed sizes; unknowns filled with your conservative estimate),
     * `max_depth`, `size_mix` (JSON: `{small_pct, medium_pct, large_pct}`),
     * `space.available_bytes`, `space.needed_bytes`, `space.margin`.
   * Extract counters (emit at `extract.done`):

     * `bytes_written`, `ratio_total` (`bytes_declared/ archive_size_on_disk`),
     * `prealloc.bytes_reserved`, `io.buffer_size_bytes`, `atomic.renames`, `atomic.dirfsyncs`,
     * `hash.mode`, `hash.bytes_hashed`, `hash.algorithms`.

4. **Severity levels**

   * `extract.start|pre_scan.done|extract.start|extract.done|audit.emitted` → `INFO`
   * Rejectable policy violations → `WARN` plus a final `extract.error` at `ERROR`
   * Irrecoverable I/O/system errors → `ERROR`

**Failure Modes**

* N/A for metrics themselves; if emission fails, fall back to a single `extract.error` with `E_OBSERVABILITY` (non-fatal for the extraction unless the error caused the failure).

**Telemetry**

* As above, all events carry the common fields and their specific metrics.

**Tests**

* Unit tests assert presence and types of keys on each event.
* Golden-log tests for a happy path and at least one failure path (e.g., traversal) to validate fields and levels.
* Verify `duration_ms` increases monotonically and `run_id` is stable across all events within one run.

---

## 4.2 Failure taxonomy

**Goal**
Make triage instantaneous by standardizing error codes and payloads.

**Config & Data**

* Centralized error catalog (string constants):

  * `E_TRAVERSAL`, `E_LINK_TYPE`, `E_BOMB_RATIO`, `E_PORTABILITY`, `E_UNSUPPORTED_FORMAT`, `E_SPACE`, `E_TIMEOUT`, `E_PREALLOC`, `E_WRITE`, `E_FSYNC_FILE`, `E_FSYNC_DIR`, `E_ENTRY_RATIO`, `E_FILE_SIZE`, `E_FILE_SIZE_STREAM`, `E_DUP_ENTRY`, `E_CASEFOLD_COLLISION`, `E_DEPTH`, `E_SEGMENT_LEN`, `E_PATH_LEN`, `E_UNICODE_DECODE`, `E_CRC_MISMATCH`, `E_SIZE_MISMATCH`, `E_SHORT_READ`, `E_SPECIAL_TYPE`, `E_MULTI_TOP`, `E_OBSERVABILITY`.

**Implementation Steps**

1. Create a single **raise/emit helper** (design, not code):

   * Input: `error_code`, `message`, `details` (dict), `exception` (optional).
   * Behavior: (a) logs `extract.error` with `error_code`, `message`, `details`, then (b) raises the mapped exception (`PolicyError`/`ConfigError`/`IOError`) consistently.

2. Everywhere you abort, call the helper with:

   * `details` including `entry_name`, `normalized_path`, `declared_size`, `bytes_written`, `limit`, etc., depending on the error.

3. Ensure **one and only one** `extract.error` log per failure (avoid duplicates).

**Failure Modes**

* None; this replaces ad-hoc error messages.

**Telemetry**

* `extract.error` includes: `error_code`, `message`, `details` (JSON), and common fields.

**Tests**

* Parameterized tests mapping each policy violation to its `error_code`.
* Assert that a single failure produces exactly one `extract.error`.

---

## 4.3 Libarchive version fingerprint

**Goal**
Link odd behaviors to the underlying libarchive build.

**Config & Data**

* In-process cache of `libarchive.version` (string), captured once.

**Implementation Steps**

1. At process startup (or first extraction), read libarchive’s runtime version string.
2. Emit a **one-time** structured log:

   * `type="extract.libarchive.info"`, fields `version`, `build_flags` (if available), `platform`, `python`, `pid`.
3. Add `libarchive_version` to **every** extraction’s common fields.

**Failure Modes**

* If reading version fails, log once with `W_LIBARCHIVE_VERSION_UNAVAILABLE` and continue.

**Telemetry**

* As above.

**Tests**

* Unit test: stub version provider → assert one log record and presence of `libarchive_version` in subsequent events.

---

## 4.4 Per-archive audit record

**Goal**
A tamper-evident, machine-readable, deterministic record of what was extracted.

**Config & Data**

* `extract.manifest.emit: bool` (default **true**)
* `extract.manifest.filename: str` (default `".extract.audit.json"`)
* `manifest.schema_version: "1.0"` (string)

**Implementation Steps**

1. Build a manifest in memory during extraction with the following **canonical structure**:

   ```text
   {
     schema_version: "1.0",
     run_id, archive_path, archive_sha256, libarchive_version,
     format, filters,
     policy: { // all knobs materialized, including limits & modes
       encapsulate, allowed_formats, allowed_filters, overwrite, casefold_policy, timestamps, max_depth, size_limits, ratio_limits, include_globs, …
     },
     metrics: { entries_total, entries_included, entries_skipped, bytes_declared, bytes_written, ratio_total, duration_ms, space: {available_bytes, needed_bytes, margin} },
     entries: [
       { path_rel, scan_index, size, sha256, mtime }
       // one per successfully written regular file, in chosen order mode
     ]
   }
   ```

2. After success, write it **atomically** to `encapsulated_root / .extract.audit.json` (temp + rename; 0644).
3. Optional: compute a **manifest SHA-256** over the canonical JSON (no whitespace) and include it as `manifest_sha256`.

**Failure Modes**

* `W_MANIFEST_WRITE` (non-fatal; extraction remains successful)

**Telemetry**

* `extract.audit.emitted`: `true`, `path`, `entries` (count), `bytes` (JSON length)

**Tests**

* Assert determinism: two runs on identical inputs produce byte-identical audit JSON in deterministic modes.
* Schema validation test (simple JSON Schema) for required fields and types.

---

# 5) Developer Experience & Extensibility

## 5.1 Policy object (plug-in ready)

**Goal**
Split policy from mechanism so we can evolve behavior without forking extraction code.

**Config & Data**

* Internal `ExtractionPolicy` (not public API), frozen at extraction start.
* **Hooks** (call signatures defined in design doc, not code):

  * `allow_formats(format, filters) -> bool`
  * `limits() -> Limits` (depth, size, ratio, timeouts, etc.)
  * `encapsulation() -> {enabled: bool, name: "sha256" | "basename"}`
  * `path_normalizer(path: str) -> NormalizedPath` (Unicode normalization, separators, casefold policy)
  * `post_write_hook(path_rel, size, sha256, metadata) -> None`

**Implementation Steps**

1. A **Policy Builder** translates user config into an `ExtractionPolicy` instance once per archive.
2. The extraction engine depends only on the policy interface (no direct reads from config).
3. Provide **two** policy presets for tests: `StrictPolicy` (defaults described in your hardening plans) and `LenientPolicy` (for curated, trusted archives).

**Failure Modes**

* Misconfigured policy → throw `ConfigError` early with `E_POLICY_INVALID`.

**Telemetry**

* Emit `policy.name`, `policy.hash` (hash of normalized policy dict) at `extract.start`.

**Tests**

* Swap in `LenientPolicy` to validate overridable behaviors (e.g., allow plain HTTP, looser depth).
* Contract tests: every hook is invoked in the expected order and can veto extraction.

---

## 5.2 Probe API

**Goal**
List contents safely without writing files.

**Config & Data**

* Non-public `probe_archive(archive_path) -> List[EntryMeta]`
* `EntryMeta = { path_norm, type, size_declared, format, filter, crc_declared (optional), scan_index }`

**Implementation Steps**

1. Reuse **the same pre-scan** as extraction: normalization, policy checks (formats/filters, names, traversal), include/exclude filters, and counting/size accumulation.
2. **Never** write files: skip bodies via libarchive’s header skip, and do not allocate dirfd.
3. Return `EntryMeta` for **included** entries only; expose enough metadata to allow client decisions (e.g., dry-run UI, quota checks).

**Failure Modes**

* Same policy errors as extraction; return or raise consistently (choose raise, logged via `extract.error` with `mode="probe"`).

**Telemetry**

* `extract.pre_scan.done` with `mode="probe"` and all pre-scan metrics.

**Tests**

* Probe vs full extraction on the same archive yields the same included set and sizes.
* Probe on disallowed format raises `E_UNSUPPORTED_FORMAT`.

---

## 5.3 Idempotence mode

**Goal**
Define overwrite semantics for repeated runs.

**Config & Data**

* `extract.overwrite: "reject" | "replace" | "keep_existing"` (default **"reject"**)
* Works in tandem with **Duplicate Entry Policy** (from correctness plan).
* Requires per-file existence checks relative to the dirfd.

**Implementation Steps**

1. At target open time, check if the final path exists:

   * `"reject"` → abort with `E_OVERWRITE_FILE`.
   * `"replace"` → allow atomic replace (rename over existing).
   * `"keep_existing"` → **skip** writing this file, but still record it in audit as `skipped_existing=true`.
2. Emit a per-file decision (counted at `extract.done`): `overwrite.replaced`, `overwrite.skipped_existing`.

**Failure Modes**

* Conflicts with duplicate entry policy (e.g., `last_wins` + `reject`): resolve precedence and document (recommended: duplicate policy applies to **in-archive** duplicates; overwrite applies to **on-disk** existing files).

**Telemetry**

* `overwrite.mode`, `overwrite.replaced`, `overwrite.skipped_existing`

**Tests**

* Pre-create a subset of files and exercise all three modes; assert behaviors and audit entries.

---

# 6) Cross-Platform Edge Cases

## 6.1 Windows reserved names & trailing spaces/dots

**Goal**
Reject non-portable names *before* extraction on NTFS and keep behavior predictable on non-Windows.

**Config & Data**

* Catalog of reserved names (case-insensitive): `CON`, `PRN`, `AUX`, `NUL`, `COM1..COM9`, `LPT1..LPT9`.
* Trailing space/dot restriction on components.

**Implementation Steps**

1. In the **path_normalizer** hook (policy), after Unicode normalization:

   * If OS is Windows **or** `extract.portability.strict=true`:

     * Reject any component equal (case-insensitive) to a reserved name.
     * Reject components with trailing spaces or dots.
2. Emit `E_PORTABILITY` on violation.

**Failure Modes**

* `E_PORTABILITY` with `component` and `reason`.

**Telemetry**

* `portability.strict=true|false`, count of rejected entries.

**Tests**

* Names `con`, `NUL`, `file .ttl` (note trailing dot) → rejected in pre-scan.

---

## 6.2 Very long paths on Windows

**Goal**
Support long paths when OS allows, while still enforcing internal limits.

**Config & Data**

* `extract.windows.long_paths: bool` (default **true**)
* Internal `extract.max_path_len` still enforced (e.g., 4096 bytes UTF-8).

**Implementation Steps**

1. If on Windows and `long_paths=true`, prefix the absolute **encapsulation root** with `\\?\` when opening the root dirfd and performing `openat` equivalents.
2. Continue to enforce internal `max_path_len` and component length limits to avoid resource abuse.
3. Document that long paths require appropriate OS policy (group policy/registry); if OS refuses, map to `E_PORTABILITY`.

**Failure Modes**

* `E_PORTABILITY` (OS refused long path handling)

**Telemetry**

* `windows.long_paths=true|false`, `path_len_max`, `path_len_violation_count`.

**Tests**

* On Windows runner with long paths enabled, extract a deep path that would exceed 260 chars; assert success.
* With `long_paths=false`, the same archive fails with `E_PATH_LEN`.

---

## 6.3 NFD/NFC macOS behavior

**Goal**
Avoid duplicates/ghost files due to normalization differences.

**Config & Data**

* `extract.unicode.normalize="NFC"` (default)
* `extract.casefold_collision_policy="reject"` (default)

**Implementation Steps**

1. Ensure path normalization to **NFC** occurs before casefold collision checking.
2. On macOS, consider emitting a diagnostic field `unicode.filesystem="HFS+NFD"|"APFS"` if discoverable, but always write **NFC** names to disk.

**Failure Modes**

* Collision after normalization → `E_CASEFOLD_COLLISION`

**Telemetry**

* `unicode.form="NFC"`, `unicode.collisions`

**Tests**

* Archive with both `e\u0301.ttl` (NFD) and `é.ttl` (NFC) → rejected for collision under default policy.

---

# 7) Testing & QA Enhancements

## 7.1 Property-based tests (Hypothesis)

**Goal**
Systematically explore weird inputs and prove invariants.

**Config & Data**

* Hypothesis settings: deadline disabled for I/O ops, example counts tuned for CI.

**Implementation Steps**

1. **Generators**

   * Path components: random Unicode (include combining marks, bidi, zero-width), random case mixes, backslashes/forward slashes.
   * Depth/length generators respecting upper bounds; occasionally exceed bounds to assert rejection.
   * Entry types: regular, symlink, device (flagged), huge declared sizes vs small bodies, etc.
2. **Properties**

   * “Safe extraction never writes outside root”: after run, enumerate files; assert all under encapsulation root.
   * “Policy is monotonic”: tightening any limit never increases number of extracted files.
   * “Determinism”: with deterministic modes, two runs produce identical audit JSON.

**Failure Modes**

* N/A; failing tests surface invariants violations.

**Telemetry**

* Capture seed for any failing example.

**Tests**

* Add a test module `test_extract_property_based.py` with at least three properties as above.

---

## 7.2 Corpus of adversarial archives

**Goal**
Permanent regression shield for known tricky cases.

**Config & Data**

* Directory `tests/data/extract_corpus/` with small hand-crafted archives and metadata readme.

**Implementation Steps**

1. Include samples:

   * Path traversal (`../`), absolute paths, Windows reserved names, long paths, NFD/NFC dupes, symlinks, devices, global/per-entry bombs, unknown formats (7z/RAR), malformed headers, truncated bodies.
2. Tests walk the corpus and assert the expected code for each (`expected_error_code` stored next to the sample).

**Failure Modes**

* N/A.

**Telemetry**

* Report coverage of the corpus (how many samples exercised).

**Tests**

* Corpus runner test that parametrizes over all archives and checks expected outcomes.

---

## 7.3 Performance test

**Goal**
Detect throughput regressions early.

**Config & Data**

* Marked as `@slow` or separate CI job; set a **p95 wall-time budget** per CI class (documented in the test).

**Implementation Steps**

1. Generate or store a medium archive (~10k entries, mixed sizes, ~1 GB uncompressed).
2. Run extraction with defaults and capture wall time.
3. Assert p95 < budget (e.g., 60–120 s depending on CI hardware).
4. Emit perf telemetry to logs for trend tracking (optional).

**Failure Modes**

* Test failure only (doesn’t affect runtime behavior).

**Telemetry**

* `perf.bytes_declared`, `perf.duration_ms`, `perf.buffer_size_bytes`, `perf.hash.mode`.

**Tests**

* `test_extract_performance_medium_archive.py` with clear skip markers for low-resource CI.

---

## 7.4 Flaky detector

**Goal**
Catch non-determinism and I/O flakes.

**Config & Data**

* CI matrix step that runs extraction tests **twice** with different seeds and toggles.

**Implementation Steps**

1. In CI, run the extraction suite twice:

   * Pass 1: `extract.encapsulate=true` (default).
   * Pass 2: `extract.encapsulate=false` (if supported by policy) to exercise pathing.
2. Collect wall times; if variance > 20% on the same job class, emit a **warning** (do not fail the build).
3. Fail the build if any audit JSON differs between identical runs in deterministic modes.

**Failure Modes**

* Build warning for perf variance; failure for determinism violation.

**Telemetry**

* `ci.flaky_detector: {variance_pct, deterministic_ok}`

**Tests**

* CI workflow files only; no code tests.

---

# Final Acceptance Snapshot (for this tranche)

* [ ] Structured events and fields shipped (`extract.start`, `pre_scan.done`, `extract.start`, `extract.done`, `extract.error`, `audit.emitted`), with common fields and metrics populated.
* [ ] Centralized error helper emits a single `extract.error` with standardized `error_code` and rich `details`.
* [ ] Libarchive version logged once per process and included on every run’s common fields.
* [ ] Audit JSON emitted atomically with stable schema/version; deterministic under deterministic modes.
* [ ] `ExtractionPolicy` abstraction in place; engine depends on the policy interface only; strict & lenient presets exist for tests.
* [ ] Non-public `probe_archive` reuses pre-scan and returns `EntryMeta` for included entries.
* [ ] Idempotence modes implemented: `"reject" | "replace" | "keep_existing"` with counters and audit annotations.
* [ ] Windows portability checks (reserved names, trailing dot/space) and long-path support handled; NFC normalization + casefold collision enforced (macOS included).
* [ ] Property-based tests, adversarial corpus, perf test (budgeted), and CI flaky detector added.

This gives you **top-tier visibility**, **clean extensibility**, and **rock-solid cross-platform behavior**, all aligned with your “minimal bespoke, maximal clarity” direction.
