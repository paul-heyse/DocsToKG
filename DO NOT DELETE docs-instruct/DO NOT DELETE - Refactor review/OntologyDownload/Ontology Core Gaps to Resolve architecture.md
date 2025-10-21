Absolutely — here are two “architecture cards” for the **libarchive-based secure extraction** you’re adding to `src/DocsToKG/OntologyDownload`. They’re designed to make the design pop at a glance and double as onboarding docs for engineers and agents.

---

# 1) Big-Picture Architecture

## 1.1 Component Map (who talks to whom)

```
┌──────────────────────────────────────────────────────────────────┐
│        Callers (planner/validators/CLI)                          │
│        - ontology_download.extract_archive_safe(...)             │
└───────────────┬──────────────────────────────────────────────────┘
                │ (stable API)
                ▼
┌──────────────────────────────────────────────────────────────────┐
│   Extraction Pipeline (extraction.py)                            │
│   - Phase A: PRE-SCAN (libarchive headers only)                  │
│   - Phase B: EXTRACT (stream files to disk)                      │
│   - Policy gates (path, type, ratios, portability)               │
│   - Atomic write discipline (temp → fsync → rename → dirfsync)   │
│   - Timestamps & perms normalization                             │
└───────┬───────────────────────┬──────────────────┬────────────────┘
        │                       │                  │
        │                       │                  │
┌───────▼────────┐     ┌────────▼─────────┐   ┌────▼────────────────┐
│ Extraction     │     │ Observability    │   │ Settings            │
│ Policy (strict │     │ (events.emit)    │   │ (Pydantic v2)       │
│ defaults)      │     │ extract.* events │   │ ExtractionSettings   │
└───────┬────────┘     └────────┬─────────┘   └────────────┬────────┘
        │ (gates: OK/ERROR)      │ (start/pre_scan/done)               │
        │                        │ (error with E_* code)               │
        │                        │                                     │
        ▼                        ▼                                     ▼
┌────────────────┐     ┌────────────────────────┐          ┌─────────────────────┐
│ Filesystem     │     │ Audit JSON (optional)  │          │ Config Hash         │
│ <dest>/<root>/ │     │ .extract.audit.json    │          │ for provenance      │
└────────────────┘     └────────────────────────┘          └─────────────────────┘
```

**Key properties**

* **Two-phase** design blocks writes until the archive passes policy.
* **Libarchive** does format/codec detection and efficient streaming; we enforce policy and atomicity.
* **Events** and **audit** make runs testable, debuggable, and reproducible.
* **Settings** define limits; their normalized materialization contributes to a **config_hash** reused in all events.

---

## 1.2 Two-Phase Flow (happy path)

```
extract_archive_safe(archive, dest):
  emit extract.start
  Phase A: PRE-SCAN
    open libarchive.reader(archive)
    for each header:
      normalize path (NFC)
      enforce path policy (no abs/.., depth/length, portability)
      enforce entry type allowlist (regular/dir only)
      apply include/exclude filters
      record target path under encapsulated_root
      accumulate sizes & metrics
    enforce format/filter allow-list
    enforce zip-bomb guards (global ratio)
    enforce single-top policy (or encapsulate)
    emit extract.pre_scan(metrics)

  Phase B: EXTRACT
    mkdir encapsulated_root; open dir FD
    for each included file (deterministic order):
      mkdir parents via dirfd/openat
      open temp file (O_CREAT|O_EXCL|O_NOFOLLOW)
      (optional) posix_fallocate(size)
      stream blocks → write; hash inline (sha256)
      fsync(file); rename(temp → final); set mtime
      every N files → fsync(parent dir)
    emit extract.done(metrics, duration)
    return [final regular file paths]
```

**On any policy failure or I/O error:**
Clean up the active temp, emit **extract.error** (with precise `E_*`), and raise.

---

# 2) Policy & Security at a Glance

## 2.1 “Gates” and Errors (defense-in-depth)

```
Path Gate
  - Reject abs paths, '..', backslashes
  - Enforce depth/segment/path length
  - Normalize Unicode (NFC); detect case-fold collisions
  - Windows reserved names & trailing space/dot
  ERRORS: E_TRAVERSAL, E_DEPTH, E_SEGMENT_LEN, E_PATH_LEN, E_PORTABILITY, E_CASEFOLD_COLLISION

Entry Gate
  - Allow only regular files & directories
  - Deny symlink/hardlink/device/FIFO/socket
  ERRORS: E_SPECIAL_TYPE

Format/Filter Gate
  - Allow only {zip, tar, ustar, pax, gnutar} and {none,gzip,bzip2,xz,zstd}
  ERRORS: E_FORMAT_NOT_ALLOWED

Bomb/Size Gate
  - Global ratio: sum(uncompressed)/archive_size ≤ max_total_ratio
  - Per entry ratio (when available): uncompressed/entry_compressed ≤ max_entry_ratio
  - Size caps: file ≤ max_file_size_bytes; entries ≤ max_entries
  ERRORS: E_BOMB_RATIO, E_ENTRY_RATIO, E_FILE_SIZE, E_ENTRY_BUDGET

Durability Gate
  - temp → fsync → rename → dirfsync (atomicity)
  - perms normalization (0644 files / 0755 dirs)
```

**Encapsulation root** (optional but recommended):
All outputs live under `<dest>/<sha12>.d/` or `<dest>/<archive_basename>.d/` to 1) avoid tarbombs, 2) simplify GC, 3) make cleanups atomic by directory.

---

## 2.2 State Machine per File (Phase B)

```
PLAN → (excluded?) ──► SKIP
  │
  ├─► PREPARE_PARENTS (dirfd/openat)
  │
  ├─► OPEN_TEMP (O_CREAT|O_EXCL|O_NOFOLLOW)
  │
  ├─► (optional) PREALLOCATE
  │
  ├─► STREAM_WRITE (buffer, hash inline)
  │         ├── if write error → CLEANUP_TEMP → ERROR(E_EXTRACT_IO)
  │         └── if bytes > cap → CLEANUP_TEMP → ERROR(E_FILE_SIZE_STREAM)
  │
  ├─► FSYNC_FILE
  │
  ├─► RENAME_TEMP_TO_FINAL
  │
  ├─► SET_MTIME (per timestamps policy)
  │
  └─► DIR_FSYNC (every group_fsync)
       → DONE
```

---

# 3) Data Contracts (so tests & tools align)

## 3.1 Sanitized Target (internal, Phase A output)

```
SanitizedTarget {
  relpath_norm: str      # NFC path under encapsulated_root
  type: "file"|"dir"
  size_declared: int|None
  include: bool          # after include/exclude evaluation
  scan_index: int        # deterministic order
}
```

## 3.2 Metrics (events & audit)

```
ExtractionMetrics {
  entries_total: int
  entries_included: int
  bytes_declared: int
  bytes_written: int
  ratio_total: float|null     # if compressed size known
  max_depth_observed: int
  format: str                 # libarchive reported
  filters: [str]
  encapsulated_root: str
  duration_ms: int
}
```

## 3.3 Audit JSON (optional but recommended; deterministic)

```
{
  "schema_version": "1.0",
  "run_id": "...",
  "libarchive_version": "x.y.z",
  "archive_path": "...",
  "archive_sha256": "…",
  "format": "tar",
  "filters": ["gzip"],
  "policy": { …ExtractionSettings materialized… },
  "metrics": { …ExtractionMetrics… },
  "entries": [
    { "path_rel": "data/obo/core.ttl",
      "scan_index": 12,
      "size": 38291,
      "sha256": "…",
      "mtime": "2025-10-20T01:23:45Z" }
    …
  ]
}
```

---

# 4) Observability (“answers, not just logs”)

## 4.1 Event Grammar (namespaced & consistent)

* `extract.start`
  `{archive, dest, format=?, filters=?, policy_snapshot={encapsulate,…}}`

* `extract.pre_scan`
  `{entries_total, entries_included, bytes_declared, ratio_total?, max_depth_observed, encapsulated_root}`

* `extract.done`
  `{entries_included, bytes_written, ratio_total?, duration_ms, format, filters}`

* `extract.error`
  `{error_code, message, details={entry?, reason, limit, observed}}`

**All events** include `{ts, run_id, config_hash, context{app_version, python, os, libarchive_version}}`.

## 4.2 Monday-Morning Queries (DuckDB or CLI)

* **Safety heatmap**: `SELECT payload.error_code, COUNT(*) FROM events WHERE type='extract.error' GROUP BY 1 ORDER BY 2 DESC;`
* **Zip-bomb sentinels**: `SELECT ts, payload.ratio_total FROM events WHERE type='extract.done' AND payload.ratio_total > 10.0;`
* **Throughput**: `SELECT approx_quantile(payload.duration_ms,0.95) FROM events WHERE type='extract.done';`
* **Format mix**: `SELECT payload.format, COUNT(*) FROM events WHERE type='extract.done' GROUP BY 1;`

---

# 5) Ordering, Performance, and Durability

## 5.1 Deterministic Ordering (choose 1)

* **Header order** (default): `scan_index` preserves libarchive header sequence.
* **Path ascending**: stable lexicographic order by normalized relative path.
  Pick once; expose as a setting only if you truly need both.

## 5.2 Performance knobs

* **Pre-scan**: O(headers) with minimal allocations; pre-compute **size mix** to choose buffer size:

  * Mostly small files → 64 KiB buffer
  * Many large files → up to 1 MiB buffer (configurable)
* **Preallocation**: `posix_fallocate` where supported; ignore failure gracefully.
* **Fsynced durability**: `fsync(file)` before `rename`; `fsync(dir)` every `group_fsync` files (defaults to 32).
  This yields a **crash-consistent** directory image without excessive stalls.

---

# 6) Platform Notes

* **Windows**: if you need long paths, adopt `\\?\` only when the OS policy supports it, but still enforce your internal `max_path_len`. Always block reserved names (`CON`, `NUL`, …) and trailing dot/space.
* **macOS**: normalize input names to **NFC**; detect NFD/NFC collisions via case-fold comparison.
* **POSIX**: dirfd/openat + `O_NOFOLLOW|O_EXCL` prevents symlink races.

---

# 7) How It Fits Existing Modules

* **Signature stays the same**, so callers don’t change: `extract_archive_safe(archive, dest, logger=None)`.
* **Settings**: all limits come from `ExtractionSettings` (Pydantic v2) built by your traced settings loader; any change updates the **config_hash** surfaced in events/audit.
* **Validators** (SHACL/ROBOT/etc.) consume the **extracted files**; the audit JSON gives them optional provenance and file lists if needed.
* **Doctor/Prune** can treat the encapsulation root as a unit — delete/GC is now trivial and safe.

---

# 8) Quick Reference: Error Map

```
E_TRAVERSAL            absolute/.. paths or backslashes
E_DEPTH                exceeds max_depth
E_SEGMENT_LEN          a component exceeds max_components_len
E_PATH_LEN             full path exceeds max_path_len
E_PORTABILITY          Windows reserved names or trailing dot/space
E_CASEFOLD_COLLISION   duplicate after case-fold normalization
E_SPECIAL_TYPE         symlink/hardlink/device/fifo/socket
E_FORMAT_NOT_ALLOWED   unknown/blocked format or filter
E_BOMB_RATIO           sum(uncompressed)/archive_size > max_total_ratio
E_ENTRY_RATIO          per-entry ratio > max_entry_ratio
E_FILE_SIZE            declared file size exceeds max
E_FILE_SIZE_STREAM     stream grew beyond max during write
E_EXTRACT_CORRUPT      libarchive read failure (truncated, invalid)
E_EXTRACT_IO           write/rename/preallocate failure
```

Each error is emitted once via `extract.error` and raised as `PolicyError/ConfigError/IOError` with a compact `details` dict.

---

## TL;DR (for engineers skimming)

* **Two-phase**: pre-scan (decide) → extract (write).
* **Hardened**: no symlinks/devices, no traversal, zip-bomb guards, NFC normalization, Windows portability.
* **Atomic**: temp → fsync → rename → dirfsync.
* **Observable**: `extract.*` events + deterministic audit JSON; **config_hash** everywhere.
* **Config-driven**: all limits live in `ExtractionSettings` (Pydantic) and feed the hash.

If you’d like, I can convert these cards into a single “**Secure Extraction**” page (with linkable anchors and small code snippets) you can keep under `docs/architecture/`.
