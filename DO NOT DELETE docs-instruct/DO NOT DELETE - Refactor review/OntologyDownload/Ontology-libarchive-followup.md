Awesome — here’s a **best-in-class, highly specific, step-by-step implementation plan** to land **Secure Archive Extraction** in `src/DocsToKG/OntologyDownload`. It preserves your public surface, replaces ad-hoc `zipfile`/`tarfile` flows with a hardened **libarchive** two-phase pipeline, and wires in settings, events, tests, and budgets so regressions can’t hide.

---

# 0) Scope & Definition of Done

**Goal:** one authoritative `extract_archive_safe(archive_path, destination, *, logger=None) -> list[Path]` that:

* Uses **libarchive-c** for all formats/filters, in **two phases**:

  * **Phase A (pre-scan)**: read headers **only**; normalize paths; enforce **policy gates**; compute totals/ratios.
  * **Phase B (extract)**: stream regular files to **temp files** under the final parent dir → **fsync(file)** → **rename** → periodic **dirfsync**.
* Enforces defense-in-depth: traversal block, **regular files only**, **zip-bomb** (global & per-entry) guards, **NFC** normalization, case-fold collision detection, Windows portability, per-file size & entry-count caps, include/exclude globs, deterministic ordering.
* Emits structured **events**: `extract.start | extract.pre_scan | extract.done | extract.error` (with `run_id`, `config_hash`, metrics).
* (Optional but recommended) Writes deterministic **`.extract.audit.json`** under the encapsulation root.
* Cross-platform green (Linux/macOS/Windows). No partial final files on failure.

---

# 1) Files & Ownership (exact paths)

```
src/DocsToKG/OntologyDownload/
  extraction.py                # NEW: authoritative implementation
  policy/path_policy.py        # NEW: path normalization & checks (if not already present)
  policy/errors.py             # ensure E_* codes exist (reuse if present)
  observability/events.py      # existing or add tiny emit() shim
  settings.py                  # extend ExtractionSettings (see §2)
  ontology_download.py         # PUBLIC WRAPPER stays here; delegates to extraction.py
tests/ontology_download/
  test_extract_prescan_policy.py
  test_extract_security_types.py
  test_extract_zip_bomb.py
  test_extract_per_entry_ratio.py
  test_extract_include_exclude.py
  test_extract_timestamps_perms.py
  test_extract_cross_platform.py
  test_extract_audit_manifest.py
```

**Housekeeping (remove or re-export):**

* If you have prior helpers (e.g., in `io_safe.py` or `net.py`) that do extraction, mark them **deprecated**, re-export the new function for one release, then delete.

---

# 2) Settings (strict Pydantic v2; all feed `config_hash`)

In `settings.py`, add/confirm:

```text
ExtractionSettings:
  # Formats & filters
  allowed_formats: set[str] = {"zip","tar","ustar","pax","gnutar"}
  allowed_filters: set[str] = {"none","gzip","bzip2","xz","zstd"}
  require_single_top: bool = False
  on_multi_top_violation: "encapsulate" | "reject" = "encapsulate"

  # Structure & portability
  encapsulate: bool = True
  encapsulation_name: "sha256" | "basename" = "sha256"
  unicode_form: "NFC" | "NFD" = "NFC"
  casefold_collision_policy: "reject" | "allow" = "reject"
  max_depth: int = 32
  max_components_len: int = 240     # bytes UTF-8
  max_path_len: int = 4096          # bytes UTF-8
  windows_portability_strict: bool = True

  # Bomb/size/entries
  max_total_ratio: float = 10.0     # sum(uncompressed)/archive_size
  max_entry_ratio: float = 100.0    # per-entry when available
  max_entries: int = 50000
  max_file_size_bytes: int = 2_147_483_648  # 2 GiB

  # Throughput/durability
  preallocate: bool = True
  copy_buffer_min: int = 65536
  copy_buffer_max: int = 1_048_576
  group_fsync: int = 32

  # Timestamps/perms
  timestamps_mode: "preserve" | "normalize" | "source_date_epoch" = "preserve"
  timestamps_normalize_to: "archive_mtime" | "now" = "archive_mtime"

  # Filtering & hashing
  include_globs: list[str] = []
  exclude_globs: list[str] = []
  hash_enable: bool = True
  hash_algorithms: list[str] = ["sha256"]
```

**Validation rules:**

* All numbers must be >0 where appropriate; ratios ≥ 1.0.
* `include_globs`/`exclude_globs` pre-compile to a predicate; filenames must be normalized **before** matching.
* `allowed_formats/filters` are lower-cased; reject unknown strings early with a helpful message (list allowed).

---

# 3) Public wrapper (stable API)

In `ontology_download.py`:

```text
def extract_archive_safe(archive_path: Path, destination: Path, *, logger: Optional[logging.Logger]=None) -> list[Path]:
  # 1) load settings + build policy snapshot + config_hash
  # 2) delegate to extraction.extract_archive_safe_impl(...)
  # 3) return list[Path]
```

No call-site changes across the repo.

---

# 4) Core implementation (extraction.py)

## 4.1 Top-level structure

```
extract_archive_safe_impl(archive_path, destination, settings, logger=None) -> list[Path]
  emit extract.start {archive, dest, policy_key_fields...}
  t0 = monotonic()

  # Phase A: PRE-SCAN
  prescan = _pre_scan(archive_path, destination, settings)

  emit extract.pre_scan {entries_total, entries_included, bytes_declared, ratio_total?, format, filters, encapsulated_root}

  # Phase B: EXTRACT (only if prescan passed)
  results = _extract_streaming(archive_path, prescan, settings)

  emit extract.done {entries_included, bytes_written, ratio_total?, duration_ms}
  return results
```

On any failure, call `events.emit("extract.error", …)` once and raise the mapped exception.

---

## 4.2 Phase A — PRE-SCAN (headers only)

**Open & iterate:**

* Use `libarchive.file_reader(str(archive_path))`.
* For each header:

  * `entry.type` must be in `{regular, directory}`; otherwise **reject** `E_SPECIAL_TYPE`.
  * `name_raw = entry.pathname` → **normalize**:

    1. Ensure **POSIX** separators (`/`); if backslashes present → reject (safer) or convert then validate.
    2. Unicode **NFC** normalization.
    3. **Reject**: absolute paths (start with `/`), drive letters (`^[A-Za-z]:`), any `..` segments.
    4. Enforce `max_depth`, `max_components_len` per segment (bytes), `max_path_len` (bytes).
    5. If `windows_portability_strict`:

       * Reject Windows **reserved names** (`CON`, `PRN`, `AUX`, `NUL`, `COM1..9`, `LPT1..9`) at any component (case-insensitive).
       * Reject trailing **space** or **dot** on any component.
    6. Build `relpath_norm` (POSIX, NFC).
  * **Case-fold collision**:

    * Maintain a set of `relpath_norm.casefold()`. If collision (same case-fold, different original) and policy=`reject` → `E_CASEFOLD_COLLISION`.
  * Accumulate **declared size** for **regular files only**: `size_declared = entry.size or 0`.
  * (Optional) capture per-entry **compressed size** if libarchive exposes it (ZIP) for per-entry ratio later.

**Encapsulation:**

* If `settings.encapsulate`:

  * Compute once:

    * if `sha256`: `enc_root = destination / f"{sha256(archive_bytes)[:12]}.d"`
    * else `basename`: `enc_root = destination / f"{archive_path.stem}.d"`
  * All target paths are `enc_root / relpath_norm`.
* Else: target is `destination / relpath_norm`.

**Include/exclude filtering:**

* Compute final **extraction set** by applying compiled globs on **normalized `relpath_norm`** (dirs still created even if files excluded).
* `entries_total` counts all headers; `entries_included` counts regular files included.

**Format & filters allow-list:**

* Read archive **format name** (e.g., `zip`, `tar`, `ustar`, `pax`, `gnutar`) and **filter names** (e.g., `gzip`, `xz`, `zstd`).
* Reject any not in `allowed_*` with `E_FORMAT_NOT_ALLOWED`.

**Single-top policy:**

* Compute top-level components of **included** files; if `require_single_top` and count != 1:

  * if `on_multi_top_violation="encapsulate"`: already contained by `enc_root` → OK
  * else `E_MULTI_TOP`

**Zip-bomb (global) guard:**

* `compressed = archive_path.stat().st_size`
* `ratio_total = bytes_declared / compressed` (skip if `compressed == 0`)
* If `ratio_total > max_total_ratio` → `E_BOMB_RATIO`

**Return:**

* A `PreScanPlan` dataclass (internal) with:

  * `encapsulated_root: Path`
  * `targets: list[TargetEntry]` where each `TargetEntry = {relpath_norm, abs_path, type, size_declared, scan_index, include: bool}`
  * `entries_total`, `entries_included`, `bytes_declared`, `ratio_total?`, `format`, `filters`

---

## 4.3 Phase B — EXTRACT (streaming, atomic)

**Setup:**

* `enc_root.mkdir(parents=True, exist_ok=True)` (if encapsulating) and open **dir FD**: `dirfd = os.open(enc_root or destination, O_DIRECTORY)`.
* Choose a **single reusable buffer**:

  * If >50% of **bytes_declared** are in files >4 MiB → `buffer = min(copy_buffer_max, 1 MiB)`
  * else `buffer = max(copy_buffer_min, 64 KiB)`

**Per-target loop (deterministic order by `scan_index` unless path_asc configured):**

* For **directories**: ensure existence with `mkdirat(dirfd, parent)` (dirfd-relative).
* For **regular, included** files:

  1. Build parent dirs with `mkdirat` chain; enforce no symlink in path via `O_NOFOLLOW` on directory opens where possible.
  2. **Temp file** path: `.name.tmp-<pid>-<counter>` within the **final parent dir**.
  3. Open with flags: `O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW` (on Windows, emulate via `FILE_FLAG_OPEN_REPARSE_POINT` equivalents if practical; otherwise rely on earlier path policy).
  4. If `preallocate` and `size_declared > 0`: `posix_fallocate(fd, 0, size_declared)`; on failure, **warn** once and continue.
  5. Stream entry data via libarchive → write with buffer; **hash** inline if `hash_enable` (sha256 etc.). Guard:

     * If bytes exceed `max_file_size_bytes` → remove temp; `E_FILE_SIZE_STREAM`.
  6. `fsync(file)` → `renameat(dirfd, tmp, dirfd, final_name)` → every `group_fsync` files `fsync(dirfd)`.
  7. Set final **mtime** per `timestamps_mode`:

     * `preserve`: use entry mtime if present; else archive mtime.
     * `normalize`: `archive_mtime` or now.
     * `source_date_epoch`: from env var if set.
  8. Set **perms** explicitly to `0644` for files (masked by umask).
* Track `bytes_written` as sum of final file sizes.

**Error handling:**

* If libarchive read fails (truncated/invalid) → **remove temp**, emit `extract.error` `E_EXTRACT_CORRUPT`.
* On any write/rename error → remove temp, emit `E_EXTRACT_IO`.

---

# 5) Security Gates → Error Codes (single catalog)

Use/extend a central error catalog (e.g., `policy/errors.py`). Map gates → codes:

* Path & portability: `E_TRAVERSAL`, `E_DEPTH`, `E_SEGMENT_LEN`, `E_PATH_LEN`, `E_PORTABILITY`, `E_CASEFOLD_COLLISION`
* Entry types: `E_SPECIAL_TYPE`
* Format/filters: `E_FORMAT_NOT_ALLOWED`
* Bomb/size: `E_BOMB_RATIO`, `E_ENTRY_RATIO`, `E_FILE_SIZE`, `E_ENTRY_BUDGET`
* Extraction errors: `E_EXTRACT_CORRUPT` (read), `E_EXTRACT_IO` (write/rename)

**Raising:** one helper `raise_with_event(code, details)` that 1) emits `extract.error` with `{error_code, details}`, 2) raises the right exception (`PolicyError`, `ConfigError`, or `IOError`).

---

# 6) Observability (events & audit)

**Event envelope:** all include `{ts, run_id, config_hash, context{app_version, os, python, libarchive_version}, ids{}}`.

* `extract.start` → `{archive, destination, policy_snapshot: {encapsulate, limits…}}`
* `extract.pre_scan` → `{entries_total, entries_included, bytes_declared, ratio_total?, max_depth_observed, encapsulated_root, format, filters}`
* `extract.done` → `{entries_included, bytes_written, ratio_total?, duration_ms, format, filters}`
* `extract.error` → `{error_code, message, details{entry?, reason, limit, observed}}`

**Audit JSON** (optional): write `enc_root/.extract.audit.json` atomically (temp → rename):

```json
{
  "schema_version": "1.0",
  "run_id": "…",
  "libarchive_version": "x.y.z",
  "archive_path": "…",
  "archive_sha256": "…",
  "format": "tar",
  "filters": ["gzip"],
  "policy": { "...materialized settings..." },
  "metrics": {
    "entries_total": 437, "entries_included": 281,
    "bytes_declared": 128734662, "bytes_written": 93455672,
    "ratio_total": 6.12, "duration_ms": 1843
  },
  "entries": [
    { "path_rel":"data/obo/core.ttl","scan_index":12,"size":38291,"sha256":"…","mtime":"2025-10-21T01:23:45Z" }
  ]
}
```

---

# 7) Tests (file-level, explicit behaviors)

## 7.1 Unit

* `test_extract_prescan_policy.py`:

  * Absolute paths, `..`, backslashes → `E_TRAVERSAL`
  * Depth/segment/path length limits → respective errors
  * Windows reserved names / trailing dot/space → `E_PORTABILITY`
  * Case-fold collision (`A.ttl` vs `a.ttl`) → `E_CASEFOLD_COLLISION` when policy=`reject`

* `test_extract_security_types.py`:

  * Symlink/hardlink/device/FIFO/socket → `E_SPECIAL_TYPE` (rejected at pre-scan)

* `test_extract_zip_bomb.py`:

  * Small `.zip` expanding beyond `max_total_ratio` → `E_BOMB_RATIO`

* `test_extract_per_entry_ratio.py`:

  * ZIP entry with extreme ratio → `E_ENTRY_RATIO` if compressed size known

* `test_extract_include_exclude.py`:

  * Include only `*.ttl|*.rdf|*.owl|*.obo`; non-ontology files skipped

* `test_extract_timestamps_perms.py`:

  * `preserve | normalize | source_date_epoch` applied correctly; perms `0644/0755`

## 7.2 Component / Integration

* `test_extract_cross_platform.py`:

  * macOS: NFD → NFC normalization + collision check
  * Windows (runner): reserved names fail

* `test_extract_audit_manifest.py`:

  * Audit JSON exists, schema minimal compliance, deterministic ordering (header vs path_asc mode)

* Corruption scenarios:

  * Truncated tar/zip → `E_EXTRACT_CORRUPT`; **no partial finals**; temps cleaned
  * Write failure injection (mock `os.rename` failure) → `E_EXTRACT_IO`; temps removed

**Performance sanity (marks=slow):**

* Pre-scan 10k headers < budget (CI-specific); streaming 500 MB with inline hashing overhead <10%.

---

# 8) CI & Budgets

* PR lane: unit + component suites; no slow tests.
* Nightly: add slow & cross-platform (Windows/macOS runners).
* Budgets (tune for your runner class):

  * Pre-scan 10k headers: **≤ 250 ms**
  * Stream 500 MB (compressible): **≥ 150 MiB/s** (inline hash overhead <10%)

---

# 9) Migration Steps (in order, small PRs)

**PR-E1 — Shell & settings**

* Add `extraction.py` skeleton; extend `ExtractionSettings`; wire `extract.start`/`pre_scan` event emission.

**PR-E2 — Pre-scan gates**

* Implement path normalization & all policy gates; compute metrics; tests for traversal/portability/allow-lists.

**PR-E3 — Streaming & durability**

* Implement temp-write → fsync → rename → dirfsync; perms & timestamps; `extract.done`; tests for happy paths & I/O errors.

**PR-E4 — Bomb guards & per-entry**

* Implement ratio checks; tests for total & per-entry ratio.

**PR-E5 — Audit & docs**

* Write `.extract.audit.json` atomically; add tests; update README Settings table; add “Secure Extraction” doc page.

**PR-E6 — Cleanup**

* Remove or deprecate old `zipfile`/`tarfile` code; ensure a single `extract_archive_safe` remains.

---

# 10) Developer Aids

* Add `ONTODOC_EXTRACT_DEBUG=1` mode that:

  * Logs sanitized path mapping (raw → normalized → final)
  * Dumps the policy snapshot & chosen buffer size
  * Prints libarchive `format/filters` once

* Add `ontofetch extract --dry-run <archive> --dest <dir>` to run **pre-scan only** and print a formatted plan (count, bytes, top components, filtered entries).

---

# 11) Quick QA Checklist (paste into PR)

* [ ] One authoritative `extract_archive_safe` delegates to libarchive two-phase pipeline; signature unchanged.
* [ ] Path policy enforces traversal block, NFC normalization, case-fold collision, Windows reserved names, length/depth caps.
* [ ] Only regular files extracted; link/device types rejected.
* [ ] Global & per-entry bomb guards active; per-file & entry count caps enforced.
* [ ] Atomic write discipline (temp → fsync → rename → periodic dirfsync) verified; no partial finals on failure.
* [ ] Deterministic ordering (header order default) and optional path_asc.
* [ ] Events (`extract.start|pre_scan|done|error`) carry `run_id` & `config_hash` with metrics.
* [ ] Optional audit JSON written atomically; schema-compliant & deterministic.
* [ ] Linux/macOS/Windows tests green; performance sanity budgets met.
* [ ] Old `zipfile/tarfile` code deleted or re-exported with deprecation notice.

---

This plan gives you a **hardened, observable, deterministic** extraction pipeline that matches the standard we set everywhere else in the project: **typed settings**, **evented instrumentation**, **atomic boundaries**, and **proof-grade tests**.
