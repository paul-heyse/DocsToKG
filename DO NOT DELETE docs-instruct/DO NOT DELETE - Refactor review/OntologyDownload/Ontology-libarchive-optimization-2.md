# Throughput & I/O Efficiency — Full Implementation Plan (Code-free, Agent-Ready)

Assume we already use libarchive and the hardened safety policy from your previous step. Below are **eight** throughput optimizations expanded into concrete, high-detail steps an AI agent can implement without guesswork. Each item includes **Goal**, **Config & Data**, **Implementation Steps**, **Failure Modes**, **Telemetry**, and **Tests**.

---

## 11) Two-phase extraction with precise I/O & space budgeting

**Goal**
Guarantee we won’t start writing unless the target filesystem has enough free space for the **worst-case** extraction footprint.

**Config & Data**

* `extract.space_safety_margin`: float (default **1.10** = 10% headroom).
* `extract.space_source`: `"statfs"` (default) | `"quota"` (future hook).
* `pre_scan.bytes_uncompressed_total` (from libarchive headers; fall back to conservative estimates for unknowns).

**Implementation Steps**

1. In **pre-scan**, accumulate `bytes_uncompressed_total` from entry headers; for entries with unknown size, add a conservative placeholder (e.g., `max_file_size_bytes` if you enforce it; otherwise a capped default).
2. Query available bytes on the destination filesystem (`statfs`/equivalent) **for the encapsulated root**.
3. Compute `needed = bytes_uncompressed_total × space_safety_margin`.
4. If `available < needed`, **abort before extraction** with a clear “insufficient space” error.
5. If you enable **selective extraction globs** (see §17), recompute `bytes_uncompressed_total` using only **included** entries.

**Failure Modes**

* `E_SPACE`: not enough free space (report `needed`, `available`, and `margin`).

**Telemetry**

* `space.available_bytes`, `space.needed_bytes`, `space.margin`, `pre_scan.bytes_uncompressed_total`, `entries_included`.

**Tests**

* Mount/point destination at a tmpfs with a small limit; assert pre-scan aborts with `E_SPACE`.
* With include globs that reduce footprint, assert the same archive now proceeds.

---

## 12) File preallocation to reduce fragmentation & mid-stream ENOSPC

**Goal**
Minimize fragmentation and reduce the chance of ENOSPC after partial writes by preallocating files with known sizes.

**Config & Data**

* `extract.preallocate: bool` (default **true**).
* `extract.preallocate_strategy`: `"full"` (default) | `"none"`.
* For each entry: `size_declared`.

**Implementation Steps**

1. For entries with a **known size** and `preallocate=true`, preallocate the temp file to `size_declared` **before** streaming any data.
2. For entries with **unknown size**: skip preallocation; rely on per-file size guard (§18) and global space budget (§11).
3. Ensure preallocation happens **on the same filesystem** as the final rename target (i.e., create the temp file under the final parent dir).
4. If preallocation fails due to quota/ENOSPC, abort this file and the extraction with an actionable error.

**Failure Modes**

* `E_PREALLOC`: preallocation failure (report size, path, errno).

**Telemetry**

* `prealloc.attempted_files`, `prealloc.bytes_reserved`, `prealloc.failures`.

**Tests**

* Simulate low free space: large preallocation should fail cleanly before any data writes.
* Validate that final file fragmentation (optional metric) is reduced compared to a non-preallocated run.

---

## 13) Direct-to-FD streaming (minimize copies & Python overhead)

**Goal**
Minimize per-block overhead by writing libarchive’s decompressed output **directly to file descriptors** using a single reusable buffer.

**Config & Data**

* `extract.copy_buffer_min`: bytes (default **64 KiB**).
* `extract.copy_buffer_max`: bytes (default **1 MiB**).
* `extract.fd_write_mode`: always buffered by kernel page cache (no O_DIRECT).

**Implementation Steps**

1. Maintain one **reusable copy buffer** sized by the adaptive heuristic (§15).
2. For each regular file entry, **open a temp file FD** beneath the root dirfd; stream libarchive blocks into this FD using the reusable buffer until EOF.
3. Avoid extra Python-level buffering layers; rely on OS page cache.
4. After the last block, **fsync the file**, then rename atomically.

**Failure Modes**

* `E_WRITE`: short or failed write; abort the file, remove temp, propagate error.

**Telemetry**

* `io.buffer_size_bytes`, `io.bytes_written`, `io.blocks_written`.

**Tests**

* Assert identical file contents against a known checksum set.
* Benchmark test (optional): compare wall-time vs a naive read-into-bytes approach.

---

## 14) Atomic per-file writes with robust fsync discipline

**Goal**
Ensure no partially visible files and durable directory updates.

**Config & Data**

* `extract.atomic: bool` (default **true**).
* `extract.group_fsync: int` (default **32**) — fsync the parent directory every N files to amortize cost.

**Implementation Steps**

1. For each file, write to a temp name inside the **final parent directory**: `.<name>.tmp-<pid>-<counter>`.
2. On completion, **fsync the file**, then **rename** onto the final name.
3. Keep a counter; after every `group_fsync` renames, **fsync the parent directory** using the parent dirfd.
4. On **any** failure mid-file, close & delete temp; do **not** leave temp files behind.

**Failure Modes**

* `E_FSYNC_FILE` or `E_FSYNC_DIR` (rare); emit clear reason and halt.

**Telemetry**

* `atomic.renames`, `atomic.dirfsyncs`, `atomic.group_size`.

**Tests**

* Inject a failure after write but before rename; assert no visible partial file.
* Inject crash after rename but before dir fsync; re-run extraction to check idempotence.

---

## 15) Adaptive block size selection (match file size distribution)

**Goal**
Balance syscall overhead and memory use by sizing the copy buffer to the archive’s workload.

**Config & Data**

* `extract.copy_buffer_min` / `extract.copy_buffer_max` (see §13).
* Heuristic thresholds:

  * Small files: `< 64 KiB`
  * Medium: `< 4 MiB`
  * Large: `≥ 4 MiB`

**Implementation Steps**

1. During pre-scan, compute **entry size distribution** (count & sum per bucket).
2. Choose `copy_buffer_size` using a simple heuristic:

   * If **>50%** of bytes are in **large** files → set to `min(1 MiB, copy_buffer_max)`.
   * Else if **>50%** of files (by count) are **small** → set to `max(64 KiB, copy_buffer_min)`.
   * Otherwise use **256 KiB** (middle ground).
3. Reuse this size for all file writes in the run (single buffer).

**Failure Modes**

* None; heuristic only affects performance.

**Telemetry**

* `io.buffer_size_bytes`, `io.size_mix = {small_pct, medium_pct, large_pct}`.

**Tests**

* Construct archives biased to many tiny files vs a few huge files; assert buffer size selection follows heuristic.

---

## 16) Inline or parallel hashing pipeline (content integrity + throughput)

**Goal**
Compute per-file digests **without extra read passes**, and optionally overlap hashing with I/O when beneficial.

**Config & Data**

* `extract.hash.enable: bool` (default **true**).
* `extract.hash.algorithms: ["sha256"]` (default) | add others as needed.
* `extract.hash.mode`: `"inline"` (default) | `"parallel"`.
* `extract.hash.parallel_threads`: int (default **min(2, CPU)**).

**Implementation Steps**

1. **Inline mode (default):** update digests **in the same loop** that writes blocks to disk (hashlib in C releases the GIL; this is often fast enough).
2. **Parallel mode (optional):**

   * Create a bounded **ring buffer** per file (e.g., 2–4 chunks).
   * Writer thread pushes chunks; a hasher thread consumes and updates digests.
   * Use backpressure: writer waits if the buffer is full to avoid unbounded RAM.
   * On error, drain/close the buffer and clean up temp file.
3. Persist digests in your manifest or return structure as today.

**Failure Modes**

* `E_HASH`: hasher thread error; abort file, remove temp.

**Telemetry**

* `hash.mode`, `hash.bytes_hashed`, `hash.algorithms`, `hash.threads` (if parallel).

**Tests**

* Verify digests match reference values for multiple files.
* For parallel mode, ensure blocked backpressure works (no memory blow-up) and throughput improves on multi-core machines.

---

## 17) Selective extraction (include/ignore patterns)

**Goal**
Skip I/O for irrelevant files (e.g., images or binaries) when you only need ontology sources.

**Config & Data**

* `extract.include_globs`: list (e.g., `["*.ttl","*.rdf","*.owl","*.obo"]`).
* `extract.exclude_globs`: list (optional).
* `extract.report_skipped: bool` (default **true**).

**Implementation Steps**

1. During pre-scan, **match paths** against include/exclude globs **after normalization & NFC**.
2. Build the **final extraction set** = included − excluded; recompute `bytes_uncompressed_total` and entry counts based on this set.
3. During extraction, **skip bodies** of excluded entries (libarchive header skip).
4. Emit counts: `entries_skipped`, `bytes_skipped_declared`.

**Failure Modes**

* None; policy choice. Ensure at least one file remains; otherwise treat as success with empty result set.

**Telemetry**

* `filter.include_count`, `filter.exclude_count`, `filter.skipped_bytes`.

**Tests**

* Archive mixing ontology and non-ontology files; assert only the desired types are written.
* With include list empty → result set empty but the run succeeds gracefully.

---

## 18) CPU guard (soft wall-time per archive)

**Goal**
Prevent pathological archives (e.g., highly compressed with slow codecs) from monopolizing CPU.

**Config & Data**

* `extract.max_wall_time_seconds`: int (default **120**).
* `extract.cpu_guard_action`: `"abort"` (default) | `"warn"`.

**Implementation Steps**

1. Start a wall-time timer at the beginning of extraction.
2. Periodically (e.g., after every N files or M bytes) check elapsed wall time.
3. If elapsed > max:

   * If `abort` → stop extraction, cleanup current temp file, return an actionable error (`E_TIMEOUT`).
   * If `warn` → log once and continue.
4. Ensure the guard integrates with the hashing pipeline and backpressure (no deadlocks on abort).

**Failure Modes**

* `E_TIMEOUT` on slow or adversarial inputs.

**Telemetry**

* `cpu_guard.max_seconds`, `cpu_guard.elapsed_seconds`, `cpu_guard.action`.

**Tests**

* Create an archive with a very slow-to-decompress entry (e.g., nested compression or synthetic). Assert abort at ~configured threshold and no partial outputs.

---

## Cross-cutting: Execution Order & Interactions

**Recommended implementation order** (each step delivers independent value):

1. **Atomic writes & fsync discipline** (§14)
2. **Direct-to-FD streaming** (§13)
3. **Two-phase budgeting** (§11)
4. **Preallocation** (§12)
5. **Adaptive buffer heuristic** (§15)
6. **Selective extraction** (§17)
7. **Inline hashing** (baseline) → optional **parallel hashing** (§16)
8. **CPU guard** (§18)

**Interactions to watch:**

* **Preallocation + Space budget**: preallocation should only run after the global `E_SPACE` check; still handle per-file ENOSPC.
* **Selective extraction** lowers the required `needed_bytes`; re-compute budget **after** filtering.
* **Parallel hashing** adds concurrency; ensure error propagation cleans up temp files and releases buffers.
* **Adaptive buffer** is global; keep it simple and deterministic for reproducibility across runs.

---

## End-to-end Telemetry Additions (across items)

Emit once per archive:

* `extract.format`, `extract.filters`, `io.buffer_size_bytes`, `prealloc.bytes_reserved`, `pre_scan.bytes_uncompressed_total`, `write.bytes_written`, `filter.entries_included`, `filter.entries_skipped`, `space.available_bytes`, `space.needed_bytes`, `hash.mode`, `duration_ms`, `cpu_guard.elapsed_seconds`.

---

## QA / Test Matrix (beyond unit tests)

* **Throughput benchmarks**:

  * Many tiny files (10k × 1–8 KiB), few huge files (4 × 1–4 GiB), mixed workload.
  * Measure wall-time with/without preallocation; with small vs large buffers; inline vs parallel hashing.

* **Robustness drills**:

  * Kill the process between temp write and rename (simulate); on restart, ensure temp leftovers (if any) are pruned by your higher-level cleanup, or that no temps remain due to proper exception handling.
  * Toggle `group_fsync` between 1 and large values; validate durability vs latency trade-off in benchmarks.

* **Platform checks**:

  * Linux (ext4/xfs), macOS (APFS), Windows (NTFS) — verify atomicity, fsync semantics, and performance variance.
  * Network filesystems (optional): document caveats if fsync/rename guarantees differ.

---

These optimizations keep your implementation **fast, durable, and predictable** while preserving the strict safety posture you’ve already designed. They also remain faithful to your “minimal bespoke” philosophy: most complexity is policy/config and disciplined use of standard OS semantics, not custom algorithms.
