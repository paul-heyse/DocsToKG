Awesome — here’s a **surgical, code-free implementation plan** for **PR 4** to **switch `extract_archive_safe` to libarchive** while **retaining the exact public signature and behavior** (including logging and return type). I’ve grounded the steps in your repo’s current shape where `extract_archive_safe(archive_path: Path, destination: Path, *, logger: Optional[logging.Logger]=None) -> List[Path]` is the centralized entry point used by validators and pipeline code.

---

# North-star (what “done” looks like)

* `extract_archive_safe` continues to accept `(archive_path, destination, *, logger=None)` and returns **`List[Path]` of extracted files**; call sites and imports **do not change**.
* Implementation uses **libarchive-c** (Python bindings to libarchive) for **reading and extracting**, eliminating direct `zipfile`/`tarfile` use. ([PyPI][1])
* Security guarantees are **equal or stronger** than current: path traversal prevention, explicit policy on symlinks/hardlinks/devices, and **zip-bomb guard (~10:1 uncompressed:compressed)** remain in force.   ([OpenSSF Best Practices Working Group][2])

---

# Design decisions (explicit)

1. **Two-phase extraction** to avoid partial writes:

   * **Phase A (pre-scan):** iterate headers with libarchive (no writes) to validate each entry (type, path, size), compute total uncompressed size for bomb detection, and build the **exact destination path list**. Uses `archive_read_next_header` semantics via libarchive-c’s `file_reader`. ([manpages.debian.org][3])
   * **Phase B (extract):** perform the actual extraction **only if** Phase A passes all checks. Prefer libarchive’s secure extract flags or manual streaming write under our sanitized paths (see § 5). ([PyPI][1])

2. **Security posture (default-deny on risky features):**

   * **Reject** entries that are **symlinks, hardlinks, device nodes, FIFOs, sockets** (consistent with your “safe extraction” intent). If needed later, add a config flag to allow symlinks but **never** allow them to escape `destination`.
   * **Path traversal** protections: both **our own canonicalization check** and libarchive’s **SECURE_NODOTDOT/SECURE_NOABSOLUTEPATHS** remain active. ([PyPI][1])
   * **Zip-bomb guard:** keep your **~10:1** ratio rule (sum of uncompressed sizes ÷ on-disk archive size). Pre-scan yields sizes without writing any content.  ([OpenSSF Best Practices Working Group][2])

3. **Behavioral compatibility:**

   * Preserve your **logging fields** (e.g., stage=`"extract"`, archive path, file count).
   * Preserve **ordering** of returned paths (header order) and **destination tree shape** used by validators.

---

# Step-by-step implementation

## 0) Dependencies & module boundaries

* Add/confirm runtime dep: **`libarchive-c`** (import as `libarchive`). Document that it wraps the system **libarchive** (supports zip, tar.* etc.). ([PyPI][4])
* Identify the **authoritative definition** of `extract_archive_safe` (currently centralized; previously moved out of per-format helpers). Keep the **public import/export** stable from the package facade.

## 1) Replace format-specific branches with libarchive detection

* Remove bespoke suffix detection for routing (`.zip`, `.tar.gz`, …) **from the core path**; libarchive auto-detects formats/filters. You may keep suffix checks **only** to improve error messages. (Your earlier centralization handled suffixes explicitly; libarchive lets us simplify.)  ([PyPI][1])

## 2) Pre-scan phase (no writes)

* Open archive with **`libarchive.file_reader(archive_path)`**. For **each entry header**:

  * **Collect**: original entry pathname, entry type, declared size. (libarchive exposes entry attributes; Python binding mirrors these.) ([PyPI][1])
  * **Reject** if type ∈ {symlink, hardlink, char/block device, fifo, socket}.
  * **Normalize and validate path**:

    * Disallow absolute paths and drive letters.
    * Canonicalize (resolve `.`/`..`), **recompose** under `destination`, and **verify containment**: `normalized_target.is_relative_to(destination)` (conceptually).
    * Disallow empty components, backslashes in posix context, and NULs (if any).
  * **Accumulate** `total_uncompressed += entry.size` (skip unknown sizes by conservative upper-bound policy: treat as violating threshold or do a streaming one-pass with rolling limit).
  * **Skip body** via libarchive’s “read-data-skip” semantics to keep pre-scan fast; you’re only iterating headers in this pass. (Libarchive’s `archive_read_next_header` + skip is the intended pattern.) ([manpages.debian.org][3])
* **Bomb check**: obtain `compressed = archive_path.stat().st_size`; compute ratio and **reject** if `total_uncompressed / compressed > threshold` (~10). Log metrics on rejection.

## 3) Prepare destination & staging

* Ensure `destination` exists.
* Decide on **write strategy**:

  * **A. Libarchive “extract” functions**: change CWD to `destination` and call `libarchive.extract_file(archive_path, flags=SECURE_*)`, then rebuild the file list from the **pre-scan mapping** (preferred → **minimal bespoke code**). ([PyPI][1])
  * **B. Manual streaming** with `file_reader`: create directories, stream file blocks to the **pre-validated target paths**. Choose this if you require atomic per-file creation semantics or extra policy (e.g., sanitizing modes).
* Either way, ensure **NO owner preservation** (do not chown), keep default umask; you may preserve **mtime** if important for validators.

## 4) Extraction phase

* If **strategy A** (library extract):

  * Use default secure flags (PyPI notes these are applied automatically when `flags=None`: **SECURE_NODOTDOT**, **SECURE_NOABSOLUTEPATHS**, **SECURE_SYMLINKS**). Consider adding **NO_OVERWRITE** if you want idempotence. ([PyPI][1])
  * Reconstruct the **returned `List[Path]`** using the sanitized mapping built in pre-scan (filter non-regular files).
* If **strategy B** (manual streaming):

  * For **directories**, ensure they exist.
  * For **regular files**, stream blocks to `target_path` and fsync on close; forbid overwrites unless explicitly allowed; append to the results list.
  * If any write fails, **stop** and best-effort remove any files you wrote (documented partial rollback behavior).

## 5) Symlinks, hardlinks, devices policy

* **Default**: reject these **in pre-scan** so extraction never attempts to create them. Log a clear policy message (`"entry type not permitted"`).
* If you later need symlink support for trusted sources, gate it behind a config flag and still enforce **containment** (symlink target must resolve inside `destination`).

## 6) Logging & telemetry (unchanged keys)

* On success, log `"extracted {zip|tar|auto} archive"` with `extra={"stage": "extract", "archive": str(archive_path), "files": len(results)}` as before. On rejection, log reason (`"path_traversal"`, `"bomb_ratio"`, `"entry_type_disallowed"`). (Your prior logs used `stage="extract"`; keep that for continuity.)

## 7) Error semantics

* Unsupported/undetected format → **ConfigError** with detected format hint (keep message shape).
* Path traversal or policy violation → **ConfigError** (“safe extraction policy”).
* Libarchive errors (corruption) → **ConfigError** wrapping the underlying message.

## 8) Test plan (augment; no caller edits)

Create/verify tests that target the function behavior (no code changes required in callers):

* **Traversal**: archive with `../evil` and absolute `/etc/passwd` → rejected before any writes (assert no files created).
* **Bomb**: tiny `.zip` inflating past threshold → rejected; log includes ratio.
* **Symlink/Hardlink**: entry is symlink/hardlink → rejected (policy).
* **Device/FIFO**: entries of those types → rejected.
* **Windows drive letters**: `C:\…` style member names → rejected (treated as absolute).
* **Happy path (zip/tar.*)**: normal archives extract; returned `List[Path]` matches header order and expected subpaths.
* **SevenZip/other** (optional): verify libarchive’s auto-detection supports formats you care about; otherwise assert “unsupported format” message is clear. ([PyPI][1])
* **Idempotence**: re-extract same archive → either no overwrite or allowed overwrite matches prior behavior; assert consistent results set.
* **Telemetry**: check `stage="extract"` log and final `files` count.

## 9) Cleanup & deletions (after green)

* Delete or deprecate **format-specific helpers** now replaced by libarchive in this path (`extract_zip_safe`, `extract_tar_safe`, compression-ratio duplications). These were previously noted as candidates after centralization.
* Remove direct imports of `zipfile`/`tarfile` from modules that only used them for extraction; keep where still needed elsewhere.

---

# Acceptance checklist (agent can tick as they implement)

* [ ] The **only** extraction entry point used by validators/pipeline is `extract_archive_safe(archive_path, destination, *, logger=None) -> List[Path]`.
* [ ] Implementation uses **libarchive-c**; no direct `zipfile`/`tarfile` calls remain in this function. ([PyPI][1])
* [ ] **Pre-scan** validates entry types and sanitized target paths; **bomb ratio** enforced (~10:1).
* [ ] Extraction honors **SECURE_NODOTDOT**, **SECURE_NOABSOLUTEPATHS**, **SECURE_SYMLINKS** (via flags or our own checks). ([PyPI][1])
* [ ] Returned `List[Path]` and **stage="extract"** logging match prior behavior.
* [ ] Tests cover traversal, bomb, disallowed entry types, normal formats, and telemetry.
* [ ] Former helper functions and unused imports are removed (or deprecated) across the package.

---

## Notes for reviewers

* libarchive auto-detects **formats and compression filters** and exposes entry headers first, which is ideal for our pre-scan → extract pattern and lets us keep the **“no partial writes on rejection”** guarantee. ([manpages.debian.org][3])
* The **secure-by-default flags** in libarchive-c match our policy stance and dramatically reduce bespoke edge-case code you previously maintained. ([PyPI][1])

If you want, I can convert this into a ready-to-paste PR description with the above acceptance boxes and link to the specific tests to add/modify.

[1]: https://pypi.org/project/libarchive-c/?utm_source=chatgpt.com "libarchive-c · PyPI"
[2]: https://best.openssf.org/Secure-Coding-Guide-for-Python/CWE-664/CWE-409/?utm_source=chatgpt.com "CWE-409: Improper Handling of Highly Compressed Data (Data Amplification) | OpenSSF Best Practices Working Group"
[3]: https://manpages.debian.org/bookworm/libarchive-dev/archive_read_header.3.en.html?utm_source=chatgpt.com "archive_read_header(3) — libarchive-dev — Debian bookworm — Debian Manpages"
[4]: https://pypi.org/project/libarchive-c/2.3/?utm_source=chatgpt.com "libarchive-c · PyPI"
