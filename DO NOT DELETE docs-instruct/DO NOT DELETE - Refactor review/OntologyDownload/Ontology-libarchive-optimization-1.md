# Safety & Policy Hardening — Full Implementation Plan (Code-free, Agent-Ready)

Below are **ten** hardening topics expanded into a concrete, code-free plan. For each we specify **Goal**, **Config & Data**, **Implementation Steps**, **Failure Modes**, **Telemetry**, and **Tests**. Assume this is applied to the already-libarchive-backed `extract_archive_safe(archive_path, destination, *, logger=None) -> List[Path]`.

---

## 1) Single-root encapsulation (“anti-tarbomb”)

**Goal**
Guarantee that every extraction lands in **one deterministic subdirectory** under `destination`, enabling atomic cleanup and preventing file scatter.

**Config & Data**

* `extract.encapsulate: bool` (default: **true**).
* `extract.encapsulation_name: "sha256" | "basename"` (default: **sha256**).
* `encapsulated_root: Path` computed once per archive.

**Implementation Steps**

1. Pre-flight: compute `archive_sha256` and `archive_basename` (no need to load full archive to compute hash—stream once or reuse existing metadata if available).
2. Choose `encapsulated_root`:

   * If `sha256`: `destination / f"{archive_sha256[:12]}.d"`.
   * If `basename`: `destination / f"{archive_basename}.d"`.
3. Create `encapsulated_root` with restricted perms (e.g., 0755) and **hold a directory FD** to it (used later for `openat`-style operations).
4. **All extracted paths are relative to this root**, never directly to `destination`.

**Failure Modes**

* Root already exists and `extract.overwrite="reject"` → fail with `E_OVERWRITE_ROOT`.
* Cannot create root (permissions / ENOSPC) → fail with `E_ROOT_CREATE`.

**Telemetry**

* `extract.encapsulated=true`, `root=<path>`, `naming_policy=<sha256|basename>`.

**Tests**

* Extraction yields files only under `<destination>/<root>`.
* Re-extract with same archive + `encapsulate=true` → identical root; policy for pre-existing root honored.

---

## 2) DirFD + “openat” semantics (race-free pathing)

**Goal**
Eliminate symlink races/TOCTOU by doing **all filesystem ops relative to a directory FD** of the root, with “no-follow” and exclusive creation.

**Config & Data**

* No new user config; internal requirement.
* `root_fd` (int) for `encapsulated_root`.

**Implementation Steps**

1. Open `encapsulated_root` directory **once**, store `root_fd`.
2. During extraction, resolve each normalized relative path into parent + leaf components.
3. Create directories using relative ops (e.g., `mkdirat(root_fd, ...)`).
4. Open files for writing with flags equivalent to `O_CREAT|O_EXCL|O_NOFOLLOW` via the `root_fd`; write to a `.tmp` name, then `renameat` into place.
5. After all writes in a directory, `fsync` the **directory FD**.

**Failure Modes**

* Attempt to follow a symlink in parents → kernel rejects due to `O_NOFOLLOW` at open time; map to `E_TRAVERSAL`.
* Pre-existing file with overwrite policy “reject” → `E_OVERWRITE_FILE`.

**Telemetry**

* `extract.dirfd=true`, `file_open_flags=["O_NOFOLLOW","O_EXCL"]`.

**Tests**

* Archive with a path that becomes a symlink mid-extraction (inject race in tests) → still safe.
* Archive trying to escape via a symlinked intermediate directory → rejected.

---

## 3) Symlink & hardlink defense-in-depth

**Goal**
Never create links; belt-and-suspenders checks prevent link abuse.

**Config & Data**

* `extract.allow_symlinks: bool` (default: **false**).
* `extract.allow_hardlinks: bool` (default: **false**).

**Implementation Steps**

1. **Pre-scan**: if entry type is symlink or hardlink → **reject** with reason.
2. **Extraction**: regardless of pre-scan, open target with `O_NOFOLLOW`; after write/rename, `fstat` and ensure **regular file**.
3. If you ever enable `allow_symlinks` for trusted sources:

   * Resolve link **target** as if extracted; verify resolved absolute path stays within root; reject otherwise.

**Failure Modes**

* Link entries present → `E_LINK_TYPE`.
* Post-write `fstat` not regular → `E_POSTTYPE`.

**Telemetry**

* Count rejected link entries; log sample names.

**Tests**

* Archive with symlink and hardlink entries → rejected in pre-scan.
* If `allow_symlinks=true`, link pointing outside root → rejected.

---

## 4) Device/FIFO/socket quarantine

**Goal**
Forbid creation of special files entirely.

**Config & Data**

* No new config (or `extract.allow_special=false` default).

**Implementation Steps**

1. **Pre-scan**: reject entries of types device (char/block), FIFO, socket.
2. Ensure libarchive flags do not auto-create such entries even if pre-scan missed.

**Failure Modes**

* Encounter such entries → `E_SPECIAL_TYPE`.

**Telemetry**

* `special_rejected=<count>`, sample names.

**Tests**

* Archives containing each special type → rejected; zero filesystem writes.

---

## 5) Case-fold collision detection

**Goal**
Avoid duplicate/overwriting outcomes on case-insensitive filesystems.

**Config & Data**

* `extract.casefold_collision_policy: "reject"|"allow"` (default: **reject**).

**Implementation Steps**

1. During **pre-scan**, maintain a set of `casefold(normalized_relpath)`.
2. If a new entry’s casefolded path already exists:

   * `reject` → fail with `E_CASEFOLD_COLLISION`.
   * `allow` → proceed; note potential overwrite risk (discouraged).

**Failure Modes**

* Duplicate after casefold → `E_CASEFOLD_COLLISION`.

**Telemetry**

* `casefold_collisions=<count>`.

**Tests**

* Add `A.ttl` and `a.ttl` to archive; on casefold-insensitive expectation → rejected.

---

## 6) Component & path constraints

**Goal**
Prevent path abuse via extreme depth/length.

**Config & Data**

* `extract.max_depth` (default: 32).
* `extract.max_components_len` (default: 240 bytes per path component after UTF-8 encode).
* `extract.max_path_len` (default: 4096 bytes after UTF-8 encode).
* `extract.normalize_unicode: "NFC"|"NFD"` (default: **NFC**; applied before limits).

**Implementation Steps**

1. Normalize each relative path to **NFC** prior to validation.
2. Split into components and enforce:

   * Depth ≤ `max_depth`.
   * Each component length ≤ `max_components_len`.
   * Full encoded length ≤ `max_path_len`.
3. Reject on first violation; record which limit failed.

**Failure Modes**

* Too deep → `E_DEPTH`.
* Component too long → `E_SEGMENT_LEN`.
* Full path too long → `E_PATH_LEN`.

**Telemetry**

* Include maximum observed depth/lengths; log offending sample.

**Tests**

* Construct paths breaching each limit → rejected with correct code.
* Mix of Unicode combining sequences → normalized to NFC first, then limits applied.

---

## 7) Entry count & inode budget

**Goal**
Prevent DoS via millions of tiny files.

**Config & Data**

* `extract.max_entries` (default: 50,000).
* `extract.max_total_inodes` (alias of `max_entries`, documented both ways).

**Implementation Steps**

1. **Pre-scan**: count *extractable* entries (post policy filters).
2. If count > `max_entries` → reject immediately.

**Failure Modes**

* Exceeds count → `E_ENTRY_BUDGET`.

**Telemetry**

* `entries_declared`, `entries_allowed`.

**Tests**

* Archive with `max_entries+1` tiny regular files → rejected during pre-scan.

---

## 8) Per-file size guard

**Goal**
Prevent a single giant file from exhausting disk.

**Config & Data**

* `extract.max_file_size_bytes` (default: 2 GiB; configurable).

**Implementation Steps**

1. During **pre-scan**, use the declared uncompressed size (when available).
2. If size unknown, enforce during streaming:

   * Stop when threshold exceeded; delete partial temp file; mark as failed.

**Failure Modes**

* Declared size exceeds threshold → `E_FILE_SIZE`.
* Streaming exceeds threshold → `E_FILE_SIZE_STREAM`.

**Telemetry**

* `file_size_max_bytes`, sample offending entry.

**Tests**

* Entry with declared size just over limit → rejected.
* Entry with unknown size that surpasses limit during write → aborted and cleaned.

---

## 9) Per-entry compression ratio guard

**Goal**
Catch localized bombs even when global ratio passes.

**Config & Data**

* `extract.max_entry_ratio` (default: 100:1).
* Availability depends on format (ZIP typically has both sizes).

**Implementation Steps**

1. If libarchive exposes compressed and uncompressed sizes per entry:

   * Compute `ratio = uncompressed / max(1, compressed)`.
   * If `ratio > max_entry_ratio` → reject archive.
2. If compressed size is unavailable:

   * Fall back to global ratio only; log `entry_ratio_unavailable=true`.

**Failure Modes**

* Entry ratio exceeded → `E_ENTRY_RATIO`.

**Telemetry**

* `max_entry_ratio`, `offending_entry`, `entry_ratio`.

**Tests**

* ZIP entry with 0.5 KB compressed → 100 MB uncompressed (ratio > 200:1) → rejected.
* TAR.GZ where per-entry compressed size not provided → `entry_ratio_unavailable`.

---

## 10) Explicit default permissions

**Goal**
Avoid inheriting risky modes; predictable perms across platforms.

**Config & Data**

* `extract.preserve_permissions: bool` (default: **false**).
* `extract.dir_mode` (default: 0755), `extract.file_mode` (default: 0644).

**Implementation Steps**

1. When `preserve_permissions=false`, **ignore** stored mode bits that include setuid/setgid/sticky.
2. Apply `dir_mode` to created directories and `file_mode` to created files (after write & before rename), masked by process umask.
3. When `preserve_permissions=true`, still **strip setuid/setgid/sticky** for safety unless an explicit privileged mode is enabled by config (discouraged).

**Failure Modes**

* None expected; policy enforcement.

**Telemetry**

* `preserve_permissions=false|true`, `modes_applied`.

**Tests**

* Archive with files marked 0777 and setuid bits → resulting files are 0644; no setuid.
* Directories created with 0755.

---

# Cross-cutting Implementation Notes

* **Ordering & determinism**: Always record `results` in **header order** (libarchive read sequence).
* **Atomicity**: Per-file atomic via temp + rename; directory atomicity out of scope but mitigated by **encapsulation root**.
* **Cleanup on failure**: If any policy fails **before** extraction, nothing is written. If failure occurs **during** extraction, remove *only* the in-progress temp file and keep previously completed files; report `partial=true` in telemetry.
* **Space budgeting**: Combine with throughput optimization—use pre-scan `total_uncompressed` and compare to `statfs(destination)` free bytes with safety margin (e.g., 1.1×); fail with `E_SPACE`.

---

# Telemetry & Error Taxonomy (consistent across items)

* **Common keys**: `stage="extract"`, `archive=<path>`, `format`, `filters`, `entries_total`, `entries_allowed`, `bytes_declared`, `bytes_written`, `ratio_total`, `duration_ms`, `encapsulated_root`, `dirfd=true`.
* **Error codes** (string constants in error messages/logs):

  * `E_TRAVERSAL`, `E_LINK_TYPE`, `E_SPECIAL_TYPE`, `E_CASEFOLD_COLLISION`, `E_DEPTH`, `E_SEGMENT_LEN`, `E_PATH_LEN`, `E_ENTRY_BUDGET`, `E_FILE_SIZE`, `E_FILE_SIZE_STREAM`, `E_ENTRY_RATIO`, `E_OVERWRITE_ROOT`, `E_OVERWRITE_FILE`, `E_SPACE`.

---

# Test Plan (coverage linked to each policy)

Create/expand `tests/ontology_download/test_extract_archive_policy.py`:

* **Encapsulation**: verify root subdir and contained outputs; re-extract behavior.
* **DirFD/openat**: simulate symlink race in parent during extraction → safe.
* **Symlink/Hardlink**: presence → rejected; if allowed in future, outside-root target → rejected.
* **Special files**: device/FIFO/socket → rejected.
* **Casefold collisions**: `A.ttl`/`a.ttl` → policy enforced.
* **Limits**: depth, component length, path length → each rejection.
* **Entry budget**: exceed `max_entries` → rejected.
* **Per-file size**: declared + streaming overflows → rejected, cleanup verified.
* **Per-entry ratio**: ZIP with extreme ratio → rejected.
* **Permissions**: setuid bits stripped, file/dir modes applied.
* **Space check**: simulate low space → `E_SPACE` before writes.
* **Telemetry**: assert presence of keys and codes for both success and failure.

---

# Execution Order (to implement incrementally, safely)

1. **Encapsulation root + dirfd/openat** (foundation for everything else).
2. **Link/special-file rejection** (pre-scan).
3. **Path normalization + casefold collision + path limits** (pre-scan).
4. **Entry budget + per-file size guard + per-entry ratio + global ratio** (pre-scan; streaming guard for unknown sizes).
5. **Explicit permissions** (extraction).
6. **Space budgeting + metrics** (pre-scan + pre-extract).
7. **Full telemetry + error taxonomy** (finish).

This plan keeps the surface area minimal, maximizes safety by default, and yields a predictable, diagnosable extraction pipeline that’s robust across platforms and hostile inputs.
