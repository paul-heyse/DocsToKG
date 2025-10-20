# Correctness & Integrity — Full Implementation Plan (Code-free, Agent-Ready)

Assumptions: you’ve already migrated `extract_archive_safe` to libarchive; Safety & Policy Hardening and Throughput plans are in place (encapsulation, dirfd/openat, atomic writes, space budgeting, hashing). This plan adds **verifiable correctness**, **reproducibility**, and **deterministic outputs** without changing the public signature or call-sites.

We expand items **19–24** and add two cross-cutting integrity features (duplicate entry policy, provenance manifest). Each item includes **Goal**, **Config & Data**, **Implementation Steps**, **Failure Modes**, **Telemetry**, and **Tests**.

---

## 19) CRC / Integrity Verification (per-entry & archive-level)

**Goal**
Detect silent corruption and guarantee that bytes written match the archive’s declared integrity when available.

**Config & Data**

* `extract.integrity.verify: bool` (default **true**)
* `extract.integrity.fail_on_mismatch: bool` (default **true**)
* `extract.hash.enable: bool` (from throughput plan; default **true**)
* For each entry (libarchive header): declared CRC (ZIP) or equivalent; declared uncompressed size (most formats)

**Implementation Steps**

1. **Pre-scan capture**

   * Record for each entry: `name_norm`, `type`, `size_declared`, `crc_declared` (if present), and `format` (ZIP/TAR/…)
   * Record archive-level metadata (format, filters).

2. **Extraction stage verification**

   * **ZIP**: rely on libarchive’s CRC enforcement *and* independently compare your computed digest/CRC if available.
   * **Non-ZIP**: verify by comparing `size_written` to `size_declared` and your computed digest (e.g., SHA-256) against a reference, if you maintain one.
   * Treat a **CRC mismatch** or **short write** as a fatal error for the whole archive when `fail_on_mismatch=true`.

3. **End-of-archive sanity checks**

   * Sum of `size_written` over regular files equals `bytes_written` reported in telemetry.
   * No entries with declared sizes remain unhandled.

**Failure Modes**

* `E_CRC_MISMATCH` (ZIP CRC or computed vs declared)
* `E_SIZE_MISMATCH` (written size ≠ declared size)
* `E_SHORT_READ` (premature EOF from archive)

**Telemetry**

* `integrity.entries_checked`, `integrity.crc_mismatches`, `integrity.size_mismatches`
* `integrity.archive_format`, `integrity.filters`

**Tests**

* ZIP with one file manually CRC-tampered → `E_CRC_MISMATCH`
* Archive with truncated entry (declared size > actual) → `E_SIZE_MISMATCH`
* Clean archives (ZIP/TAR.*) → all checks pass

---

## 20) Timestamp Policy (deterministic & reproducible mtimes)

**Goal**
Deliver predictable file mtimes for reproducible downstream pipelines while preserving useful provenance when desired.

**Config & Data**

* `extract.timestamps.mode: "preserve" | "normalize" | "source_date_epoch"` (default **preserve**)
* `extract.timestamps.normalize_to: "archive_mtime" | "now"` (used when `mode="normalize"`)
* Respect `SOURCE_DATE_EPOCH` env var when `mode="source_date_epoch"`
* Option: `extract.timestamps.preserve_dir_mtime: bool` (default **false**)
* Entry header timestamps (mtime/ctime/atime if present), archive file’s own mtime

**Implementation Steps**

1. Choose **target mtime** per file:

   * `preserve`: use entry mtime if present; else fallback to archive mtime
   * `normalize`: force to `archive_mtime` (or `now`) for all files
   * `source_date_epoch`: force to integer value from env for all files
2. Apply timestamps **after** rename (final path) via the root dirfd to avoid TOCTOU.
3. Optionally set directory mtimes to the **max** mtime of contained files (only if `preserve_dir_mtime=true`).
4. Record the policy and the chosen value in telemetry once per archive.

**Failure Modes**

* None fatal; if setting time fails, log `W_SET_MTIME` and continue.

**Telemetry**

* `timestamps.mode`, `timestamps.value` (epoch for normalize/source_date_epoch)

**Tests**

* Archive with diverse entry mtimes: verify `preserve` vs `normalize` vs `source_date_epoch` outcomes
* Repeated extraction yields identical mtimes under deterministic modes

---

## 21) Unicode Normalization of Filenames (cross-platform stability)

**Goal**
Ensure path names are stable across filesystems with different Unicode compositions and avoid duplicate collisions.

**Config & Data**

* `extract.unicode.normalize: "NFC" | "NFD"` (default **NFC**)
* `extract.unicode.on_decode_error: "reject" | "replace"` (default **reject**)
* Input encodings (ZIP: UTF-8 flag vs CP437 fallback; TAR: POSIX UTF-8)

**Implementation Steps**

1. Let libarchive decode raw names; if libarchive exposes encoding flags (ZIP UTF-8), trust them and avoid re-decoding.
2. Immediately **normalize** decoded names to configured form (NFC) **before**:

   * path traversal checks
   * include/exclude filtering
   * case-fold collision detection
3. If decoding fails or yields ill-formed sequences and `on_decode_error="reject"`, fail entry/whole archive per your policy; otherwise replace invalid bytes (U+FFFD) and log a warning.
4. Persist **normalized** names to disk; never write the non-normalized original.

**Failure Modes**

* `E_UNICODE_DECODE` (when `reject`)
* `E_CASEFOLD_COLLISION` (from Safety plan; caused by normalization)

**Telemetry**

* `unicode.form`, `unicode.decode_errors`, `unicode.replacements`

**Tests**

* Archive with NFD names on macOS expectation → normalized to NFC
* Mixed ASCII + combining marks; ensure duplicates detected post-normalization
* Invalid byte sequences in names → `E_UNICODE_DECODE` (reject) or replaced (warn)

---

## 22) Top-Level Directory Enforcement (structure correctness)

**Goal**
Provide a predictable top-level structure for consumers that expect a single root folder.

**Config & Data**

* `extract.structure.require_single_top: bool` (default **false**)
* `extract.structure.on_violation: "encapsulate" | "reject"` (default **encapsulate**)
* Pre-scan: set of top-level components `top_components = {first_path_segment(name_norm)}` for included entries

**Implementation Steps**

1. During **pre-scan after filtering**, compute `top_components`.
2. If `require_single_top` is true and `len(top_components) != 1`:

   * If `on_violation="encapsulate"` → ensure the **encapsulation root** is enabled and proceed (everything goes under the encapsulated dir; structure thereby becomes single-root)
   * If `reject` → abort with `E_MULTI_TOP`
3. If already single-top, ensure that the top dir is **not** absolute and passes name policies; otherwise treat it like any entry for normalization & checks.

**Failure Modes**

* `E_MULTI_TOP` (when `reject`)

**Telemetry**

* `structure.top_components_count`, `structure.policy`, `encapsulated=true|false`

**Tests**

* Archive with multiple top-level folders: `encapsulate` → success; `reject` → failure
* Archive with a single top-level → accepted as is

---

## 23) Format Allow-List (trusted formats & filters only)

**Goal**
Restrict extraction to a vetted set of archive formats and compression filters; fail fast on unknown/rare formats.

**Config & Data**

* `extract.allowed_formats: List[str]` (default e.g., `["zip","tar","ustar","pax","gnutar"]`)
* `extract.allowed_filters: List[str]` (default e.g., `["gzip","bzip2","xz","zstd","none"]`)
* libarchive exposes detected `format_name` and `filter_stack` per archive

**Implementation Steps**

1. Before pre-scan, query libarchive for the **detected format** and **filter chain**.
2. Compare against allow-lists; if any item is disallowed → abort with `E_FORMAT_NOT_ALLOWED`.
3. Log the detected format & filters once per archive; reuse in telemetry and manifest.

**Failure Modes**

* `E_FORMAT_NOT_ALLOWED` (format or filter not in allow-list)

**Telemetry**

* `format=<name>`, `filters=[...]`, `format.allowed=true|false`

**Tests**

* Known good formats (ZIP/TAR.GZ/XZ/ZSTD) → allowed
* 7z or RAR → rejected with `E_FORMAT_NOT_ALLOWED`
* TAR with an unexpected filter → rejected

---

## 24) Entry Ordering Determinism (stable outputs)

**Goal**
Ensure deterministic result ordering and consistent behavior across runs and platforms.

**Config & Data**

* `extract.order.mode: "header" | "path_asc"` (default **header**)

  * `"header"` preserves the archive’s internal order
  * `"path_asc"` produces a stable lexicographic order independent of archive creation tool

**Implementation Steps**

1. In pre-scan, assign a **monotonic index** to each included entry (`scan_index`).
2. Build the `results: List[Path]` after extraction using either:

   * `header`: sort by `scan_index`
   * `path_asc`: sort by normalized relative path
3. Persist the chosen order in the provenance manifest (see below).

**Failure Modes**

* None (ordering only).

**Telemetry**

* `order.mode`, `order.count`

**Tests**

* Archive with interleaved directory/file entries: verify `header` vs `path_asc` outcomes
* Repeated extraction yields identical order per mode

---

## Cross-Cutting Integrity Add-Ons

### A) Duplicate Entry Policy (same normalized path appears multiple times)

**Goal**
Define one behavior for duplicates to avoid ambiguous results.

**Config & Data**

* `extract.duplicate_policy: "reject" | "first_wins" | "last_wins"` (default **reject**)
* Applied **after** normalization and case-fold/collision checks

**Implementation Steps**

1. During pre-scan, if a normalized path is encountered again:

   * `reject` → abort with `E_DUP_ENTRY` (report both indices)
   * `first_wins` → mark later duplicates as **skipped** (do not extract; count as skipped)
   * `last_wins` → later duplicate replaces earlier; ensure overwrite policy is compatible (still atomic via temp + rename)
2. Reflect the decision in telemetry and in the manifest.

**Failure Modes**

* `E_DUP_ENTRY` (when `reject`)

**Telemetry**

* `duplicates.detected`, `duplicates.policy`, `duplicates.skipped` or `replaced`

**Tests**

* Archive with two files named `same.ttl` → exercise all three policies; verify final content and logs

---

### B) Provenance Manifest (deterministic, auditable output)

**Goal**
Emit a machine-readable record of what was extracted to enable end-to-end verification.

**Config & Data**

* `extract.manifest.emit: bool` (default **true**)
* `extract.manifest.filename: str` (default `".extract.audit.json"`)
* Manifest schema (stable keys):

  * `archive_path`, `archive_sha256`
  * `format`, `filters`, `timestamps.mode/value`
  * `policy` (safety + throughput knobs materialized)
  * `entries: [ { path_rel, size, sha256, mtime, scan_index } ]`
  * `metrics` (entries counts/sizes, ratios, durations)

**Implementation Steps**

1. Build the manifest as you extract (append entries as they complete).
2. Write to the **encapsulation root**; atomic temp + rename; include a **schema_version**.
3. Optionally include a **manifest SHA-256** envelope for tamper evidence (hash of the canonical JSON).

**Failure Modes**

* `W_MANIFEST_WRITE` (non-fatal warning)

**Telemetry**

* `manifest.emitted=true|false`, `manifest.path`, `manifest.entries`

**Tests**

* Validate JSON structure and determinism across repeated runs
* Verify entry counts, sizes, digests match the actual files on disk

---

## Execution Order (recommended for incremental delivery)

1. **Unicode normalization (21)** → **Duplicate entry policy (A)** → **Ordering (24)**
2. **Format allow-list (23)**
3. **Timestamp policy (20)**
4. **CRC / integrity verification (19)**
5. **Provenance manifest (B)**
6. **Top-level directory enforcement (22)**

This order front-loads determinism and correctness of names/structure, then enforces format trust, then locks in time/digest integrity and auditability.

---

## End-to-End Acceptance (quick checklist)

* [ ] Names decoded & normalized (NFC), collisions checked, duplicate policy applied
* [ ] Allowed formats & filters enforced (others rejected)
* [ ] Deterministic ordering per configured mode
* [ ] Timestamps set per policy (preserve/normalize/SDE)
* [ ] Per-entry integrity verified (CRC/size), archive sanity checks passed
* [ ] Provenance manifest emitted with stable schema and accurate metrics
* [ ] Comprehensive tests added: corruption, timestamps, Unicode & duplicates, format gating, ordering modes, manifest determinism

This plan yields **verifiable outputs**, **reproducible builds**, and a **clean audit trail**, turning extraction into a trustworthy, testable foundation for the rest of your Ontology pipeline.
