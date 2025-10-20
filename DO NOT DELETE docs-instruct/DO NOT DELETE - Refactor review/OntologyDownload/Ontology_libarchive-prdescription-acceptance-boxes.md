**Title:** OntologyDownload: switch `extract_archive_safe` to libarchive APIs (signature unchanged)

---

## Summary

This PR replaces the internal implementation of `extract_archive_safe(archive_path, destination, *, logger=None) -> List[Path]` with **libarchive-c** while **keeping the public signature, return type, logging keys, and call-sites unchanged**. The new implementation strengthens safety (path traversal, link/device handling) and reduces bespoke code by delegating format/codec detection and streaming to libarchive.

---

## Motivation

* Consolidate multiple format-specific code paths (zip/tar/*.gz/*.xz/…) into one robust, battle-tested reader.
* Enforce safe extraction uniformly (no path traversal, no absolute paths, no links/devices, zip-bomb guard).
* Reduce maintenance surface and align with our “minimize bespoke code” objective.

---

## Scope (what changes)

* **`extract_archive_safe`** now uses **libarchive** for:

  * Format autodetection and streaming reads.
  * A two-phase flow: **pre-scan** (validate entries & compute uncompressed size) then **extract**.
* Security posture remains at least as strict as before: default-deny on symlinks/hardlinks/devices/FIFOs/sockets; path traversal protections; zip-bomb ratio guard (~10:1) enforced at the archive level.

**Not changing**

* Function signature and import surface.
* Structured log keys (e.g., `stage="extract"`).
* Return type and semantics: `List[Path]` of the files actually extracted.

---

## Design (high-level behavior)

1. **Pre-scan (no writes):**

   * Iterate headers only; collect:

     * normalized, destination-anchored target path for each entry,
     * entry type,
     * declared uncompressed size (accumulate `total_uncompressed`).
   * Reject early if:

     * entry is a link/device/FIFO/socket,
     * path is absolute, contains `..`, drive letters, or escapes `destination`,
     * zip-bomb rule violated: `total_uncompressed / archive_size > threshold` (≈10).

2. **Extract (writes):**

   * Proceed only if pre-scan passes.
   * Use libarchive’s secure extract behavior (and/or manual streaming) to write:

     * create directories as needed,
     * write regular files to the pre-validated targets,
     * do not preserve owner; default umask applies; optionally preserve mtime if current behavior expects it.
   * Build and return the `List[Path]` in header order for regular files.

3. **Security defaults:**

   * **On**: no absolute paths, no `..`, no links/devices, destination containment enforced.
   * **On**: zip-bomb guard (ratio) using pre-scan sizes vs on-disk archive size.
   * **Idempotence**: by default do not overwrite existing files unless current behavior allowed it (matched to prior semantics).

4. **Errors & logging:**

   * Unsupported/corrupt archives → `ConfigError` with a clear message.
   * Policy violations (traversal, links, bomb ratio) → `ConfigError` (“safe extraction policy …”).
   * Logs retain `stage="extract"` and include `archive`, `files` (count on success), and reason on failure.

---

## Dependency

* Runtime: **`libarchive-c`** (Python bindings). The underlying system libarchive handles ZIP/TAR/* compression filters.
* Remove direct `zipfile`/`tarfile` imports from this function’s module (keep elsewhere if still needed).

---

## Public API

* **No changes**. Existing imports and call-sites continue to work:

  * `extract_archive_safe(archive_path: Path, destination: Path, *, logger: Optional[logging.Logger] = None) -> List[Path]`

---

## Test Plan (new/updated tests)

> Create a focused suite for extraction. If a similar suite already exists, migrate/rename accordingly.

**New file**: `tests/ontology_download/test_extract_archive_safe.py`

Add the following tests (one test function per bullet):

1. **Happy path — ZIP**

   * Small ZIP with nested dirs and files.
   * Assert: returned `List[Path]` matches expected relative layout and header order; all files exist; `stage="extract"` log present with `files` count.

2. **Happy path — TAR.GZ**

   * Same assertions as ZIP; ensures filter (gzip) and format (tar) are autodetected.

3. **Traversal — parent directory**

   * Archive containing `../evil.txt`.
   * Assert: **rejected**; no files extracted; error message contains “path traversal” or similar; logs include failure reason.

4. **Traversal — absolute path**

   * Archive containing `/etc/passwd` or `C:\windows\system32\foo`.
   * Assert: **rejected**; no files extracted; message indicates absolute path blocked.

5. **Disallowed entry types — symlink**

   * Archive with a symlink entry.
   * Assert: **rejected** with a clear policy message; no writes performed.

6. **Disallowed entry types — hardlink/device/FIFO/socket**

   * One test each or a parameterized test.
   * Assert: **rejected**; no writes performed.

7. **Zip-bomb ratio guard**

   * Tiny archive that expands well beyond the ratio threshold (≈10:1).
   * Assert: **rejected**; message mentions uncompressed/compressed ratio; no writes performed.

8. **Windows path components**

   * Archive member `dir\subdir\file.txt`.
   * Assert: normalized to POSIX path under `destination`; files extracted; or rejected if policy forbids backslashes. Match prior behavior.

9. **Idempotence (re-extract)**

   * Extract same archive twice.
   * Assert: behavior matches prior semantics (no duplicate writes; either skip or overwrite exactly as before); returned `List[Path]` consistent.

10. **Telemetry**

* Capture logs; assert `stage="extract"`, `archive` path in record, and `files` count equals number of regular files extracted on success.

> **Fixtures/helpers**
>
> * Use in-test archive builders (zip/tar) to create the inputs.
> * Use a temporary directory for `destination`.
> * For the bomb test, generate a highly compressible payload.

---

## Acceptance Checklist

* [ ] `extract_archive_safe` uses **libarchive** (no direct `zipfile`/`tarfile` calls in this function).
* [ ] **Two-phase** flow implemented: pre-scan (validation & sizing) then extract.
* [ ] **Path traversal** and **absolute path** attempts are blocked before any writes.
* [ ] **Symlinks, hardlinks, devices, FIFOs, sockets** are rejected by policy.
* [ ] **Zip-bomb guard** enforced at roughly **10:1** (uncompressed/compressed); rejection leaves no partial files.
* [ ] **Destination containment** verified for every member; canonicalization preserves header order for the results list.
* [ ] **Idempotence** matches prior behavior (no unintended overwrites; consistent results).
* [ ] **Logging keys** unchanged (`stage="extract"`; `archive`, `files` on success; clear reason on failure).
* [ ] **Return type** remains `List[Path]` of regular files extracted, in header order.
* [ ] New test suite `tests/ontology_download/test_extract_archive_safe.py` added and **all tests pass**.
* [ ] Removed/deprecated format-specific extraction helpers now superseded by libarchive in this path (if any).
* [ ] Documentation comment on `extract_archive_safe` updated to describe security posture and behavior.

---

## Notes for Reviewers

* libarchive autodetects formats and compression filters, which lets us collapse multiple bespoke branches safely.
* We intentionally **reject** links/devices and absolute/parent-traversing paths to preserve “safe extraction” guarantees across all formats.
* The ratio-based bomb guard happens at **pre-scan** time so we never partially write in failure cases.
