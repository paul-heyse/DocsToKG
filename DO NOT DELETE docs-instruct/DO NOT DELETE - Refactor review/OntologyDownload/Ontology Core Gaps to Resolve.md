Absolutely — here’s a **best-in-class, code-free, agent-ready implementation plan** to add **libarchive-c**–backed, secure extraction to **`src/DocsToKG/OntologyDownload`**. It keeps the public surface stable, replaces format-specific code with a hardened two-phase pipeline, and wires telemetry, settings, and tests so regressions can’t hide.

---

# North-Star (what “done” means)

* **Public API unchanged**: `extract_archive_safe(archive_path: Path, destination: Path, *, logger: Optional[logging.Logger]=None) -> list[Path]` remains the *only* entry for archive extraction. Callers and imports don’t change.
* **libarchive-c** performs *all* format detection and payload streaming. We do **two passes**:

  1. **Pre-scan** (headers only): validate every entry (type, size, path), compute totals and guard ratios, and build **exact target paths**.
  2. **Extract**: write only if pre-scan passes; use atomic per-file writes with fsync and dirfsync.
* **Security defaults**: block symlinks/hardlinks/devices/FIFOs/sockets; block traversal and absolute paths; enforce path depth/length; normalize filenames to NFC and check case-fold collisions; strip setuid/setgid/sticky; deny non-allowlisted formats/filters; zip-bomb guards (global + per-entry).
* **Observability**: emit `extract.start | extract.pre_scan | extract.done | extract.error` events with detailed metrics (entries, bytes, ratios, duration, format, filters, policy).
* **Settings-driven behavior**: all limits and format policies come from `ExtractionSettings` (Pydantic v2), validated and included in the run’s `config_hash`.
* **Tests**: comprehensive unit/component/chaos suite (traversal, broken archives, extreme paths, zip-bombs, resume interactions, Windows/macOS edge cases).

---

# Files & Ownership

```
src/DocsToKG/OntologyDownload/
  extraction.py              # NEW: authoritative home of extraction pipeline (may house helpers)
  ontology_download.py       # keep public extract_archive_safe() wrapper here (retains signature)
  settings.py                # ensure ExtractionSettings has knobs described below
  policy/                    # if present: place path policy helpers here; else keep near extraction.py
  observability/events.py    # emit structured events (if already present elsewhere, reuse)
tests/ontology_download/
  test_extract_pre_scan.py
  test_extract_security.py
  test_extract_formats.py
  test_extract_windows_mac.py
  test_extract_zipbomb.py
  test_extract_resume_interplay.py
  test_extract_audit_manifest.py
```

> If you already have `io_safe.py` or `net.py` that housed the old extraction logic, mark their extraction helpers **deprecated**, re-export the new function to keep imports working for one release, and delete in the next.

---

# Dependencies

* Add **`libarchive-c`** (Python bindings for libarchive) to runtime deps.
* No other new runtime deps required. We will rely on system libarchive — document minimum supported version in README.

---

# Public API & Invariants

**Keep**:

```python
def extract_archive_safe(
    archive_path: Path,
    destination: Path,
    *,
    logger: Optional[logging.Logger] = None
) -> list[Path]:
    ...
```

**Invariants**:

* Returns a **list of `Path` objects for regular files actually extracted**, in **deterministic order** (configurable: header order or path ascending).
* Never writes outside `destination` (or the configured encapsulation root beneath it).
* Partial failures leave **no** partially visible files (temp+rename discipline) and **no** directory traversal side effects.
* On any policy violation or corruption, raises `ConfigError/PolicyError` and emits one `extract.error` with a precise `error_code`.

---

# Configuration (Pydantic `ExtractionSettings`)

Add/confirm these **strict, validated** fields (some may exist already; align names):

**Formats & filters**

* `allowed_formats: set[str]` — default: `{"zip","tar","ustar","pax","gnutar"}`
* `allowed_filters: set[str]` — default: `{"none","gzip","bzip2","xz","zstd"}`
* `require_single_top: bool` — default: `False`
* `on_multi_top_violation: Literal["encapsulate","reject"]` — default: `"encapsulate"`

**Security & structure**

* `encapsulate: bool` — default: `True`
* `encapsulation_name: Literal["sha256","basename"]` — default: `"sha256"`
* `unicode_form: Literal["NFC","NFD"]` — default: `"NFC"`
* `casefold_collision_policy: Literal["reject","allow"]` — default: `"reject"`
* `max_depth: int` — default: `32`
* `max_components_len: int` — default: `240` (bytes, UTF-8)
* `max_path_len: int` — default: `4096`
* `windows_portability_strict: bool` — default: `True`

**Bomb & size limits**

* `max_total_ratio: float` — default: `10.0` (uncompressed / compressed)
* `max_entry_ratio: float` — default: `100.0` (per entry, when available)
* `max_entries: int` — default: `50000`
* `max_file_size_bytes: int` — default: `2_147_483_648` (2 GiB)

**Throughput & durability**

* `preallocate: bool` — default: `True`
* `copy_buffer_min: int` — default: `65536`
* `copy_buffer_max: int` — default: `1_048_576`
* `group_fsync: int` — default: `32`  (dir fsync every N files)
* `timestamps_mode: Literal["preserve","normalize","source_date_epoch"]` — default: `"preserve"`
* `timestamps_normalize_to: Literal["archive_mtime","now"]` — default: `"archive_mtime"`

**Integrity & selection**

* `include_globs: list[str]` — default: `[]`
* `exclude_globs: list[str]` — default: `[]`
* `hash_enable: bool` — default: `True`
* `hash_algorithms: list[str]` — default: `["sha256"]`

All these feed the **`config_hash`** published in events/audit.

---

# Architecture & Flow

## 1) Two-Phase Extraction

### Phase A: **Pre-scan**

* Open with `libarchive.file_reader(archive_path)`.
* Iterate **headers only**:

  * Collect for each entry: `name_raw`, `type`, `size_declared`, `format`, `filters`.
  * Reject early on **entry type** not in `{regular, directory}`.
  * Normalize `name_raw` to **NFC**, split into components, enforce:

    * **No absolute** paths or drive letters.
    * No `..` segments; no backslashes (convert if you choose, but recommend reject).
    * Depth ≤ `max_depth`, per-segment bytes ≤ `max_components_len`, full path bytes ≤ `max_path_len`.
    * **Windows reserved names** and trailing spaces/dots rejection under `windows_portability_strict`.
  * Compute **encapsulated_root** once (if `encapsulate=true`):
    `destination / f"{sha256(archive)[:12]}.d"` or `destination / f"{archive_stem}.d"`.
  * Build **sanitized target path**: `encapsulated_root / normalized_relpath` (or `destination/…` if not encapsulating); store **dirfd-relative** mapping for Phase B.
  * Track **case-fold set** of normalized paths; if collision and policy=`reject` → error.
  * Enforce **include/exclude globs** now (post normalization) to define the **final extraction set**; directories still created as needed.
  * Accumulate:

    * `entries_total`, `entries_included`, `bytes_declared` (sum for included regular files).
    * `format_name`, `filters` (libarchive reported).
* **Bomb guard**:

  * `compressed = archive_path.stat().st_size`.
  * Reject if `bytes_declared / compressed > max_total_ratio` (allow small compressed values with safe guard; treat `0` compressed size as “unknown”, skip ratio).
* **Format/filters allow-list**:

  * Reject if detected format or any filter is not in allowed sets.
* **Single-top policy**:

  * If `require_single_top=true` and `top_components_count != 1`:

    * If `on_multi_top_violation="encapsulate"`: already satisfied by encapsulation root.
    * Else `reject`.

Emit `extract.pre_scan` with metrics (counts, bytes, ratio_total (if available), format, filters, encapsulated_root).

### Phase B: **Extract**

* Create `encapsulated_root` (if needed) and hold **dir FD** (`os.open(path, O_DIRECTORY)`).
* For each **included regular file** (in deterministic order):

  * Create parent directories with **dirfd-relative** calls.
  * Open a **temp file** in the *final* parent dir: `.<name>.tmp-<pid>-<counter>` with `O_CREAT|O_EXCL|O_NOFOLLOW`.
  * If `preallocate=true` and `size_declared` known → **posix_fallocate** to reduce fragmentation/ENOSPC surprises.
  * Stream libarchive file blocks into the temp FD using a **single reusable copy buffer** sized by a small heuristic (choose between `copy_buffer_min/max` based on size distribution computed in Phase A).
  * If `hash_enable`, update digest(s) **inline** during writes (hashlib in C; acceptable overhead).
  * On finishing writes: `fsync(file)`, **rename** temp → final name atomically, set **mtime** per `timestamps_mode`, and every `group_fsync` files: `fsync(parent_dir_fd)`.
  * **Permissions**: set to `0644` for files, `0755` for directories, masked by process umask; strip `suid/sgid/sticky`.
* If any write fails: remove the temp, emit `extract.error` and raise; **do not proceed** to the next file.
* On success: produce a `List[Path]` of **regular files** in chosen order (default: **header order**; secondary option: **path_asc** via a setting).

Emit `extract.done` with `entries_*`, `bytes_declared`, `bytes_written`, `ratio_total`, `duration_ms`, `format`, `filters`.

---

# Security & Policy (defense in depth)

* **Entry type allow-list**: only `regular` and `directory`. Reject others with `E_SPECIAL_TYPE`.
* **Path traversal**: forbid absolute & `..`; enforce via our *own* canonicalization **and** libarchive `SECURE_*` flags (see note below).
* **Libarchive secure flags** (if using “extract” helpers):

  * `SECURE_NODOTDOT`, `SECURE_NOABSOLUTEPATHS`, `SECURE_SYMLINKS`. Even though we stream manually, keep these if we call any built-ins.
* **Per-entry ratio**: when compressed size per entry is available (ZIP), compute `uncompressed / compressed` → reject if > `max_entry_ratio` (`E_ENTRY_RATIO`), else warn if unknown.
* **Zip-bomb total**: `bytes_declared / compressed > max_total_ratio` → `E_BOMB_RATIO`.
* **Per-file cap**: `size_declared` or **bytes written** > `max_file_size_bytes` → `E_FILE_SIZE`/`E_FILE_SIZE_STREAM`.
* **Unicode normalization** to NFC before policy & collision checks.
* **Windows reserved names & trailing spaces/dots**: reject early in pre-scan if `windows_portability_strict`.

---

# Telemetry & Audit

* **Events** (all include `run_id`, `config_hash`, `archive`, `format`, `filters`):

  * `extract.start` → start time, archive path, policy snapshot (selected fields).
  * `extract.pre_scan` → `entries_total`, `entries_included`, `bytes_declared`, `ratio_total?`, `max_depth_observed`, `encapsulated_root`.
  * `extract.done` → `entries_included`, `bytes_written`, `duration_ms`, `ratio_total`, `errors=0`.
  * `extract.error` → `error_code`, `message`, `details{entry, reason, limit, observed}`, `errors=1`.
* **Audit JSON** (optional but recommended):

  * `.extract.audit.json` written atomically in the **encapsulated root**:

    * `schema_version`, `run_id`, `archive_sha256`, `format`, `filters`, `policy` (materialized), `metrics` (as above), and `entries: [{path_rel, size, sha256, mtime, scan_index}]`.
  * Useful for post-hoc verification and DuckDB ingestion.

---

# Tests (what to add)

## Unit

* **Name normalization & policy**: NFC normalization, depth/length limits, reserved names, case-fold collision detection.
* **Format & filters**: allow-list decisions; unknown formats rejected (`E_FORMAT_NOT_ALLOWED`).

## Component

* **Traversal**: archives containing `../evil` and `/abs/path` → reject before writes; verify **no files** created.
* **Symlink/hardlink/device/FIFO**: reject on pre-scan; assert error codes.
* **Zip-bomb**: small compressed file expanding > `max_total_ratio` → reject; event contains ratio.
* **Per-entry ratio**: crafted ZIP entry with extreme ratio → reject if available; unknown ratio path logs “unavailable”.
* **Include/exclude**: extraction set respects globs and sidesteps non-ontology files.
* **Permissions**: suid/sgid stripped; perms 0644/0755 set under umask 022.
* **Timestamps**: `preserve | normalize | source_date_epoch` behaviors.

## Cross-platform

* **Windows**: reserved names (`con`, `NUL`, `LPT1`), trailing dot/space → reject; long path support if you implement `\\?\` mode.
* **macOS**: NFD → NFC normalization; collisions between NFD/NFC pairs rejected when policy=`reject`.

## Chaos / Recovery

* **Early close**: archive truncated → libarchive error → **no partial files**; error emitted.
* **Kill mid-extract**: simulate exception during write; assert only the **temp file** is removed and no partial final; rerun works cleanly.

## Performance sanity

* Pre-scan 10k entries in < target (budget depends on runner); streaming block sized appropriately; hashing overhead < 10% of write.

---

# Implementation Steps (PR sequence)

### PR-A: Wire dep + skeleton

* Add `libarchive-c` to deps.
* Create `extraction.py` with the two-phase skeleton (no body writes yet).
* Port settings (& defaults) into `ExtractionSettings` and compute `config_hash` impacts.
* Add `extract.start`/`pre_scan` events with basic fields.

### PR-B: Pre-scan with policy gates

* Implement header iteration; normalized paths; depth/length checks; include/exclude; case-fold collision; Windows portability; format/filter allow-list; single-top policy.
* Compute `bytes_declared`, `ratio_total`, `top_components` set.
* Emit `extract.pre_scan` metrics.
* Tests: traversal, reserved names, allow-list formats, include/exclude.

### PR-C: Extraction streaming + durability

* Implement dirfd/openat, temp files with `O_NOFOLLOW|O_EXCL`, optional preallocation, single reusable buffer; fsync(file) + rename + periodic dirfsync; perms & timestamps.
* Add per-file cap and unknown-size streaming cap.
* Emit `extract.done` and error paths.
* Tests: happy path zips/tars, permission/timestamps, unexpected EOF.

### PR-D: Bomb guards & per-entry ratio

* Implement global & per-entry ratio checks; error codes `E_BOMB_RATIO` / `E_ENTRY_RATIO`.
* Expand tests: global bomb, per-entry bomb, unknown ratio path.

### PR-E: Audit manifest + docs

* Write `.extract.audit.json` atomically with schema; document policy knobs in README and Settings doc.
* Add tests: manifest determinism and schema compliance.

---

# Error Taxonomy (use existing catalog; add if missing)

* `E_TRAVERSAL`, `E_SPECIAL_TYPE`, `E_DEPTH`, `E_SEGMENT_LEN`, `E_PATH_LEN`, `E_PORTABILITY`,
* `E_FORMAT_NOT_ALLOWED`, `E_BOMB_RATIO`, `E_ENTRY_RATIO`, `E_FILE_SIZE`, `E_FILE_SIZE_STREAM`,
* `E_EXTRACT_CORRUPT` (libarchive read error), `E_EXTRACT_IO` (write/rename/fallocate errors).

All raised via a central helper that also emits `extract.error`.

---

# Performance & Durability Notes

* **Two-phase design** avoids partial writes on policy failure and keeps stats/ratios cheap.
* **Preallocation** reduces fragmentation and mid-stream ENOSPC surprises; if unsupported, continue without failing the run (log `W_PREALLOC_UNSUPPORTED`).
* **Buffer sizing**: if >50% of total bytes in large files, use a larger buffer (up to `copy_buffer_max`); else a small default (64 KiB) is fine.
* **Atomicity**: per-file atomic via rename; directory durability ensured by periodic dirfsync (tunable per settings).

---

# Documentation Updates

* README section “Secure Extraction (libarchive)”:

  * Supported formats/filters; policy defaults; examples (“strict defaults”, “partner mode”).
* Settings doc: table of all knobs with defaults.
* Developer note: how to add a new format (usually not allowed; prefer keep allow-list tight).

---

# Definition of Done (acceptance)

* [ ] `extract_archive_safe(...)` implementation delegates to libarchive two-phase pipeline; signature unchanged; imports unchanged.
* [ ] Security gates enforced: traversal, type allow-list, Windows portability, Unicode NFC, case-fold collisions.
* [ ] Zip-bomb global & per-entry guards active; per-file cap active; include/exclude effective.
* [ ] Atomic write discipline (temp → fsync → rename; periodic dirfsync) verified; no partials remain on failures.
* [ ] Events emitted (`extract.start`, `extract.pre_scan`, `extract.done`, `extract.error`) with `run_id` and `config_hash`.
* [ ] `.extract.audit.json` written atomically; deterministic; validated by tests.
* [ ] Cross-platform tests (Windows/macOS/Linux) pass targeted suites; chaos tests confirm safe recovery.
* [ ] Performance budgets (pre-scan & streaming) meet CI thresholds.

---

This plan replaces bespoke archive handling with a hardened, observable, and configurable pipeline that’s **safer**, **simpler to maintain**, and **far easier to diagnose** when things go sideways — while keeping your public API and callers **exactly** as they are today.
