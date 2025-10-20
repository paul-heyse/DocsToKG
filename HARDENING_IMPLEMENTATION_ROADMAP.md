# Archive Extraction Safety & Policy Hardening — Implementation Roadmap

**Status:** 🚀 Starting Phase 1
**Date:** October 2025
**Scope:** Full implementation of 10 hardening policies for `extract_archive_safe()`

---

## Overview

This roadmap details the implementation of comprehensive safety hardening for libarchive-based archive extraction. We're implementing 10 interconnected policies that provide defense-in-depth against both accidental misconfiguration and intentional abuse.

### The 10 Policies

| # | Policy | Priority | Foundation | Status |
|---|--------|----------|-----------|--------|
| 1 | Single-root encapsulation | ⭐⭐⭐ | - | 🔜 |
| 2 | DirFD + openat semantics | ⭐⭐⭐ | #1 | 🔜 |
| 3 | Symlink & hardlink defense | ⭐⭐⭐ | #2 | 🔜 |
| 4 | Device/FIFO/socket quarantine | ⭐⭐⭐ | #2 | 🔜 |
| 5 | Case-fold collision detection | ⭐⭐ | #2 | 🔜 |
| 6 | Component & path constraints | ⭐⭐⭐ | #2 | 🔜 |
| 7 | Entry count & inode budget | ⭐⭐⭐ | - | 🔜 |
| 8 | Per-file size guard | ⭐⭐⭐ | #7 | 🔜 |
| 9 | Per-entry compression ratio | ⭐⭐ | #7 | 🔜 |
| 10 | Explicit default permissions | ⭐⭐ | - | 🔜 |

---

## Phase 1: Foundation (Encapsulation + DirFD)

### 1.1 Single-Root Encapsulation

**Goal:** Guarantee all files land in one deterministic subdirectory

**Config Changes:**

```python
class ExtractionPolicyConfig:
    encapsulate: bool = True  # default: enabled
    encapsulation_name: str = "sha256"  # "sha256" | "basename"
```

**Implementation:**

1. Pre-compute `archive_sha256` via streaming (one pass)
2. Generate `encapsulated_root = destination / f"{digest[:12]}.d"`
3. Create root with restricted perms (0755)
4. All extracted paths relative to this root
5. Hold directory FD for openat ops

**Telemetry:**

- `extract.encapsulated=true`
- `root=<path>`
- `naming_policy=<sha256|basename>`

**Tests:**

- Files land only under root
- Re-extract yields identical root
- Overwrite policies honored

---

### 1.2 DirFD + OpenAt Semantics

**Goal:** Eliminate TOCTOU/symlink races

**Implementation:**

1. Open `encapsulated_root` once, store `root_fd`
2. Use `mkdirat(root_fd, ...)` for directories
3. Create files with `O_CREAT|O_EXCL|O_NOFOLLOW` via root_fd
4. Atomic rename: temp file → final path
5. fsync directory FD after writes

**Failure Modes:**

- Symlink in parent path → `E_TRAVERSAL`
- Pre-existing file (reject policy) → `E_OVERWRITE_FILE`

**Telemetry:**

- `extract.dirfd=true`
- `file_open_flags=["O_NOFOLLOW","O_EXCL"]`

---

## Phase 2: Pre-Scan Security (Links, Specials, Paths)

### 2.1 Symlink & Hardlink Defense

**Config:**

```python
allow_symlinks: bool = False
allow_hardlinks: bool = False
```

**Checks:**

- Pre-scan: reject if type is link
- Extraction: open with O_NOFOLLOW, fstat after write
- If allowed: resolve target, verify within root

**Error:** `E_LINK_TYPE`

---

### 2.2 Device/FIFO/Socket Quarantine

**Implementation:**

- Pre-scan: reject char/block devices, FIFO, sockets
- Error: `E_SPECIAL_TYPE`

---

### 2.3 Path Normalization & Constraints

**Config:**

```python
max_depth: int = 32
max_components_len: int = 240  # bytes after UTF-8
max_path_len: int = 4096  # bytes after UTF-8
normalize_unicode: str = "NFC"  # "NFC" | "NFD"
```

**Checks:**

- Normalize to NFC before validation
- Check depth, component length, total length
- Errors: `E_DEPTH`, `E_SEGMENT_LEN`, `E_PATH_LEN`

---

### 2.4 Case-Fold Collision Detection

**Config:**

```python
casefold_collision_policy: str = "reject"  # "reject" | "allow"
```

**Implementation:**

- Track set of casefolded paths during pre-scan
- Detect duplicates after casefold
- Error: `E_CASEFOLD_COLLISION`

---

## Phase 3: Resource Budgets

### 3.1 Entry Count & Inode Budget

**Config:**

```python
max_entries: int = 50_000
```

**Implementation:**

- Pre-scan: count extractable entries
- Fail if count > threshold
- Error: `E_ENTRY_BUDGET`

---

### 3.2 Per-File Size Guard

**Config:**

```python
max_file_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB
```

**Implementation:**

- Pre-scan: check declared size (if available)
- Streaming: enforce during write, abort if exceeded
- Clean up partial temp file on overflow
- Errors: `E_FILE_SIZE`, `E_FILE_SIZE_STREAM`

---

### 3.3 Per-Entry Compression Ratio

**Config:**

```python
max_entry_ratio: float = 100.0  # 100:1
```

**Implementation:**

- If libarchive provides per-entry sizes: compute ratio
- Reject if exceeds threshold
- Error: `E_ENTRY_RATIO`

---

## Phase 4: Permissions & Space

### 4.1 Explicit Default Permissions

**Config:**

```python
preserve_permissions: bool = False
dir_mode: int = 0o755
file_mode: int = 0o644
```

**Implementation:**

- Strip setuid/setgid/sticky bits
- Apply dir_mode to directories
- Apply file_mode to files (before rename)
- Respect process umask

---

### 4.2 Space Budgeting

**Implementation:**

- Pre-scan: compute `total_uncompressed`
- Check against `statfs(destination).f_bavail` with safety margin (1.1×)
- Error: `E_SPACE`

---

## File Structure

### New/Modified Files

```
src/DocsToKG/OntologyDownload/io/
├── filesystem.py              (Updated: extract_archive_safe)
├── extraction_policy.py        (New: policy config & validation)
├── extraction_telemetry.py     (New: error codes & metrics)
└── extraction_constraints.py   (New: validators for each policy)

tests/ontology_download/
└── test_extract_archive_policy.py  (New: comprehensive suite)
```

---

## Configuration Evolution

### Current (Minimal)

```python
extract_archive_safe(archive_path, destination, *, logger=None)
```

### Phase 1 (Encapsulation + DirFD)

```python
extract_archive_safe(
    archive_path,
    destination,
    *,
    logger=None,
    extraction_policy=ExtractionPolicy()  # defaults: safe
)
```

### Final (All 10 Policies)

```python
class ExtractionPolicy:
    # Encapsulation
    encapsulate: bool = True
    encapsulation_name: str = "sha256"

    # Constraints
    max_depth: int = 32
    max_components_len: int = 240
    max_path_len: int = 4096
    normalize_unicode: str = "NFC"

    # Links & Special Files
    allow_symlinks: bool = False
    allow_hardlinks: bool = False

    # Budgets
    max_entries: int = 50_000
    max_file_size_bytes: int = 2 * 1024**3
    max_entry_ratio: float = 100.0
    casefold_collision_policy: str = "reject"

    # Permissions
    preserve_permissions: bool = False
    dir_mode: int = 0o755
    file_mode: int = 0o644
```

---

## Implementation Phases

### ✅ Phase 0: Foundation (DONE)

- ✅ Removed legacy format-specific functions
- ✅ Unified to libarchive-based extraction
- ✅ Two-phase pre-scan + extract pattern established

### 🔜 Phase 1: Encapsulation + DirFD

- Create extraction_policy.py (config dataclass)
- Implement single-root encapsulation
- Add DirFD + openat semantics
- Add telemetry infrastructure

### 🔜 Phase 2: Pre-Scan Hardening

- Link & special file rejection
- Path normalization & constraints
- Case-fold collision detection

### 🔜 Phase 3: Resource Budgeting

- Entry count guard
- Per-file size guard
- Per-entry ratio guard

### 🔜 Phase 4: Permissions & Finalization

- Explicit permissions enforcement
- Space budgeting
- Full telemetry + error taxonomy
- Comprehensive test suite

---

## Error Taxonomy

All error codes follow pattern: `E_<POLICY>_<TYPE>`

```
E_TRAVERSAL              Path escapes root
E_LINK_TYPE              Symlink/hardlink entry
E_SPECIAL_TYPE           Device/FIFO/socket
E_CASEFOLD_COLLISION     Duplicate after casefolding
E_DEPTH                  Path exceeds max depth
E_SEGMENT_LEN            Component too long
E_PATH_LEN               Full path too long
E_ENTRY_BUDGET           Too many entries
E_FILE_SIZE              Declared size exceeds limit
E_FILE_SIZE_STREAM       Streamed file exceeds limit
E_ENTRY_RATIO            Per-entry compression ratio exceeded
E_OVERWRITE_ROOT         Encapsulation root exists (reject policy)
E_OVERWRITE_FILE         File exists (reject policy)
E_SPACE                  Insufficient disk space
```

---

## Telemetry Keys (Common)

```
stage="extract"
archive=<path>
format=<auto-detected>
filters=<compression>
entries_total=<count>
entries_allowed=<count>
bytes_declared=<size>
bytes_written=<size>
ratio_total=<ratio>
duration_ms=<elapsed>
encapsulated_root=<path>
dirfd=true
policies_applied=["encapsulation","dirfd","link_defense",...]
error_code=<E_*>
partial=false|true
```

---

## Testing Strategy

### Test File: `test_extract_archive_policy.py`

| Policy | Test Case | Assertions |
|--------|-----------|-----------|
| 1 | Encapsulation root created | Files under root, re-extract consistent |
| 2 | DirFD race safety | Symlink in parent rejected at open |
| 3 | Link rejection | Symlink/hardlink → E_LINK_TYPE |
| 4 | Special rejection | Device/FIFO/socket → E_SPECIAL_TYPE |
| 5 | Casefold collision | `A.txt`/`a.txt` → E_CASEFOLD_COLLISION |
| 6 | Path constraints | Each limit breached → specific E_* |
| 7 | Entry budget | Exceed max_entries → E_ENTRY_BUDGET |
| 8 | File size | Declared/streamed overflow → E_FILE_SIZE |
| 9 | Entry ratio | Extreme ratio → E_ENTRY_RATIO |
| 10 | Permissions | setuid stripped, modes applied |

---

## Success Criteria

✅ **Phase 1 Complete When:**

- [ ] Encapsulation root created correctly
- [ ] DirFD prevents TOCTOU races
- [ ] Telemetry captures encapsulation_root path
- [ ] Tests verify root behavior

✅ **All Phases Complete When:**

- [ ] All 10 policies configurable
- [ ] All error codes defined & emitted
- [ ] Telemetry complete for all paths
- [ ] 50+ test cases covering all scenarios
- [ ] Documentation comprehensive
- [ ] No breaking changes to existing API

---

## Notes

- **Backward Compat:** All policies have sensible defaults; existing calls work unchanged
- **Incremental:** Can implement in phases; each phase is independent after foundation
- **Testing:** Each policy has dedicated test before proceeding to next
- **Telemetry:** Every error path and success path logged with complete context
- **Safety:** Default-deny posture; opt-in for any risky features
