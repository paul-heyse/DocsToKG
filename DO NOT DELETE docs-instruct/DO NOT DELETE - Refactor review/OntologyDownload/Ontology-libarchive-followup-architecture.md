Love it — here’s a crisp, visual **architecture pack** for the secure-extraction subsystem you’re adding to `src/DocsToKG/OntologyDownload`. It’s designed to make the flow, boundaries, and contracts obvious at a glance.

---

# 1) System context (who talks to whom)

```
┌────────────────────────────────────────────────────────────────────────┐
│  Callers                                                               │
│  - planner / CLI: ontology_download.extract_archive_safe(...)          │
│  - ingest boundary orchestrator (writes DB after FS success)           │
└───────────────┬────────────────────────────────────────────────────────┘
                │ (stable API)
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Extraction Pipeline (extraction.py)                                   │
│  ┌───────────────────────────────┐   ┌───────────────────────────────┐ │
│  │ Phase A: PRE-SCAN (headers)   │ → │ Phase B: EXTRACT (streaming)  │ │
│  └───────────────────────────────┘   └───────────────────────────────┘ │
│    • libarchive.file_reader            • dirfd/openat + O_NOFOLLOW     │
│    • path policy gates                 • temp → fsync → rename → dirfsync
│    • format/filter allow-list          • timestamps & perms normalize   │
│    • zip-bomb guards (global/per-entry)• inline hashing (sha256…)       │
└────────────┬───────────────────────────────────────┬───────────────────┘
             │                                       │
             │ events.emit(...)                      │ results: [Path]
             ▼                                       ▼
   Observability (events.*)                  Ingest boundary (caller)
   extract.start / pre_scan / done / error   → DuckDB upserts (extracted_files)
             │                                  → STORAGE.put_bytes(audit.json, opt.)
             │                                  → set_latest_version() later
             ▼
       JSON logs (stdout) + DuckDB events (flush at boundary)
```

**Key properties**

* **Two-phase**: no disk writes until the archive passes policy.
* **Atomic** per-file writes: temp → `fsync` → `rename`; periodic parent `fsync`.
* **Observable**: events carry `run_id` + `config_hash`; optional audit JSON is deterministic.
* **DB commit happens outside** the extractor (ingest boundary), *after* the filesystem is consistent.

---

# 2) Phase A vs Phase B (zoom-in)

```
Phase A: PRE-SCAN (no writes)
1) open libarchive.reader(archive)
2) for each header:
   - normalize relpath (POSIX, NFC)
   - gates: no abs/.., depth/segment/path caps, Windows portability
   - entry type allow-list: regular/dir only
   - case-fold collision check
   - include/exclude globs
   - accumulate size_declared for included regular files
3) detect format & filters → allowed?
4) single-top policy → encapsulate or reject
5) zip-bomb guard: bytes_declared / archive_size ≤ max_total_ratio?
→ build PreScanPlan (encapsulated_root, target list, metrics)
→ emit extract.pre_scan

Phase B: EXTRACT (streaming)
1) mkdir(encapsulated_root); open dir FD
2) for each included regular file (deterministic order):
   - mkdir parents via dirfd
   - open temp (O_CREAT|O_EXCL|O_NOFOLLOW) in final parent
   - posix_fallocate (if size known & enabled)
   - stream libarchive blocks → write buffer → hash inline
   - guard max_file_size_bytes during stream
   - fsync(file) → rename(temp→final) → periodic fsync(dir)
   - set mtime (preserve|normalize|SDE), set perms (0644)
3) sum bytes_written, duration
→ emit extract.done
→ return [final paths]
```

---

# 3) Policy gates → error taxonomy

```
Path & portability
  - abs path / '..' / backslashes        → E_TRAVERSAL
  - depth > max_depth                     → E_DEPTH
  - segment bytes > max_components_len    → E_SEGMENT_LEN
  - full path bytes > max_path_len        → E_PATH_LEN
  - Win reserved names / trailing dot/spc → E_PORTABILITY
  - case-fold collision (NFC)             → E_CASEFOLD_COLLISION

Entry types
  - symlink / hardlink / device / fifo / socket → E_SPECIAL_TYPE

Format & filters
  - not in {zip, tar, ustar, pax, gnutar} or filters not allowed → E_FORMAT_NOT_ALLOWED

Zip-bomb & size
  - sum(uncompressed)/archive_size > max_total_ratio → E_BOMB_RATIO
  - per-entry ratio > max_entry_ratio (when avail)   → E_ENTRY_RATIO
  - file > max_file_size_bytes (declared)            → E_FILE_SIZE
  - entries > max_entries                            → E_ENTRY_BUDGET

I/O & corruption
  - libarchive read failure / truncated              → E_EXTRACT_CORRUPT
  - write/rename/fallocate failure                   → E_EXTRACT_IO
```

**One road for errors**
A helper raises with an `E_*` code **and** emits a single `extract.error {error_code, details}`.

---

# 4) Sequence (end-to-end with IDs & events)

```
caller.extract_archive_safe(archive, dest)
  │
  ├─ settings = load() → ExtractionSettings → config_hash
  │
  ├─ events.emit("extract.start",{archive,dest,policy_snapshot,config_hash})
  │
  ├─ plan = PRE-SCAN
  │      └─ events.emit("extract.pre_scan",{entries_total,included,bytes_declared,ratio_total?,format,filters,encapsulated_root})
  │
  ├─ paths = EXTRACT (stream + atomic writes)
  │      └─ events.emit("extract.done",{included,bytes_written,ratio_total?,duration_ms,format,filters})
  │
  └─ return paths
      (caller boundary now upserts DuckDB + optionally STORAGE.put_bytes(audit.json) + later set_latest)
```

All events carry `{run_id, config_hash, context{app_version, os, python, libarchive_version}}`.

---

# 5) Data contracts (internal types)

```
PreScanPlan
  encapsulated_root: Path
  entries_total: int
  entries_included: int
  bytes_declared: int
  ratio_total: float | null
  format: str
  filters: list[str]
  targets: list[TargetEntry]

TargetEntry
  relpath_norm: str    # POSIX, NFC
  abs_path: Path
  type: "file"|"dir"
  size_declared: int|None
  scan_index: int
  include: bool

ExtractionMetrics (for events/audit)
  entries_total, entries_included
  bytes_declared, bytes_written
  ratio_total?, max_depth_observed
  format, filters, encapsulated_root, duration_ms
```

**Audit (optional)**: deterministic `.extract.audit.json` placed under encapsulated root with the metrics and `entries[{path_rel, scan_index, size, sha256, mtime}]`.

---

# 6) Per-file state machine

```
PLAN → (excluded?) ──► SKIP
  │
  ├─► PREPARE_PARENTS (dirfd, mkdirat chain)
  │
  ├─► OPEN_TEMP (O_CREAT|O_EXCL|O_NOFOLLOW)
  │
  ├─► (optional) PREALLOCATE (posix_fallocate)
  │
  ├─► STREAM_WRITE
  │      - write buffer
  │      - hash inline (sha256…)
  │      - enforce max_file_size_bytes
  │      - on libarchive error: CLEANUP_TEMP → E_EXTRACT_CORRUPT
  │      - on write error: CLEANUP_TEMP → E_EXTRACT_IO
  │
  ├─► FSYNC(file)
  │
  ├─► RENAME(temp → final)
  │
  ├─► SET_MTIME (policy)
  │
  └─► DIR_FSYNC (every group_fsync) → DONE
```

---

# 7) Ordering & performance knobs

* **Ordering**: default **header order**; optionally **path_asc** (single setting).
* **Buffer sizing**: choose 64 KiB–1 MiB based on pre-scan size mix.
* **Preallocation**: `posix_fallocate` when size known; warn but continue if unsupported.
* **Budget sanity** (per CI class):

  * Pre-scan 10k headers ≤ ~250 ms
  * Stream 500 MiB compressible ≥ ~150 MiB/s; hashing overhead < 10%

---

# 8) Cross-boundary choreography (where DB & “LATEST” live)

```
FS success (extract done)  ──►  DuckDB upsert (extracted_files rows)  ──►  STORAGE.put_bytes(audit.json) (opt.)
                                                                                      │
                                                                                      └─ later boundary: set_latest_version()
```

* **Never** commit DB before FS success (avoids torn states).
* **LATEST** is **authoritative in DuckDB** (mirror JSON optional for humans/ops).

---

# 9) “What to watch” (SLOs & guardrails)

* **Safety**: `extract.error` heatmap by `error_code` (E_TRAVERSAL, E_BOMB_RATIO, …)
* **Latency**: p95 `duration_ms` of `extract.done` (by format/filter)
* **Zip-bomb sentinels**: any `ratio_total > max_total_ratio` in `extract.done`
* **Throughput**: MiB/s (bytes_written / duration) trend over time
* **Portability**: count of `E_PORTABILITY` on Windows/macOS runners

---

# 10) File locations (for quick orientation)

```
src/DocsToKG/OntologyDownload/
  extraction.py               # two-phase pipeline (libarchive)
  policy/path_policy.py       # path normalization & checks
  settings.py                 # ExtractionSettings (strict)
  observability/events.py     # emit extract.* events
  ontology_download.py        # public wrapper (signature unchanged)

tests/ontology_download/
  test_extract_*              # traversal, zipbomb, per-entry, include/exclude, perms, timestamps, audit
  (plus cross-platform suites in CI)
```

---

If you’d like, I can turn this into a one-page **“Secure Extraction”** doc (with link anchors, tiny code snippets, and the error-code table) so engineers and agents can onboard in minutes.
