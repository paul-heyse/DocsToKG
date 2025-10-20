Here’s a **surgical, narrative-only implementation plan** for **PR-2: Replace bespoke JSONL & file locks with libraries**. It is structured so an AI programming agent can execute it step-by-step without guessing. No code is included—only precise instructions, file targets, invariants, and acceptance checks.

---

# Scope & intent

**Goals**

1. **Standardize file locking** across DocParsing on `filelock.FileLock`, eliminating custom “.lock” sentinels and spin-loops. Today you maintain a hand-rolled `acquire_lock()` with busy-wait, PID checking, and mkdir retries.
2. **Replace bespoke JSONL readers/writers** with a small, library-backed I/O surface:

   * **Read (streaming)**: `iter_jsonl` / `iter_jsonl_batches` (keep names and behavior) but implement using a JSONL library and delete the private `_iter_jsonl_records` loop. Your current readers expose `skip_invalid` and `max_errors`; keep these semantics.
   * **Write (append/atomic)**: unify on a single `jsonl_append_iter(..., atomic=True)` and the existing `atomic_write` (retain; it already ensures parent dirs).
3. **Simplify telemetry appends** by injecting a lock-aware writer, removing the private `_acquire_lock_for()` indirection (`StageTelemetry` currently acquires its own lock and appends).

**Non-goals (deferred)**

* No change to file formats, manifest field names, or directory layout.
* No change to CLI shape or environment variables.

---

# Current duplication & pain points (inventory)

* **Locking**: multiple, slightly different versions of “lockfile with busy-wait” exist (`_common.acquire_lock`, later a `core/concurrency.acquire_lock`) and are used by writers and telemetry paths. This leads to drift and edge cases.
* **JSONL reading**: a private `_iter_jsonl_records` with custom error budget and slicing drives the public readers; it’s duplicated/reached via wrappers and patched in places like `jsonl_load` (now deprecated).
* **Writes/atomicity**: you recently hardened `jsonl_append_iter` to do single-shot append+fsync and changed manifest appends to default `atomic=True`; keep this but move locking to a single, library-backed place.
* **Parent dir creation**: your `atomic_write` now creates parents—this is correct; retain it as a foundational primitive.

---

# Target design (the “after” state)

1. **One locking primitive** used everywhere: `FileLock(path.with_suffix(".lock"))` as a context manager, replacing all bespoke “.lock” file loops. Callers never implement their own retry/timeouts; the lock context owns this behavior.
2. **One JSONL read surface**:

   * `iter_jsonl(path, *, start=None, end=None, skip_invalid=False, max_errors=10)`
   * `iter_jsonl_batches(paths, batch_size, *, skip_invalid=False, max_errors=10)`
     The public signatures remain, but the internals no longer call `_iter_jsonl_records` (delete it). Maintain `skip_invalid` and `max_errors` semantics.
3. **One JSONL write surface**:

   * `jsonl_append_iter(path, rows, *, atomic=True)` remains the only append API for manifests/telemetry. The implementation already enforces atomic single-shot writes; keep it.
   * `atomic_write(path)` stays for full-file writes; do not re-implement; it already ensures parent dirs.
4. **Telemetry**:

   * `StageTelemetry` no longer calls its own `_acquire_lock_for(path)`; instead it **accepts a writer** dependency: `writer(path, rows)` that internally handles locking + append. Default this to a `FileLock` + `jsonl_append_iter(..., atomic=True)` combo; remove `_acquire_lock_for`. Today the sink appends by invoking a private lock and `jsonl_append_iter`; consolidate this.

---

# Step-by-step implementation plan

## 0) Pre-flight

* Branch: `codex/pr2-jsonl-and-locks`.
* Add runtime deps to your main extras: **`filelock`** and **`jsonlines`** (plus optional **`msgspec`** behind an extra like `docparsing-fastio`).
* Grep the tree for:

  * `acquire_lock(` (both old `_common` and `core/concurrency`) and `_acquire_lock_for(` in telemetry.
  * `_iter_jsonl_records` and any direct JSON line loops.

## 1) Locking: centralize on FileLock

**Files:**

* `src/DocsToKG/DocParsing/core/concurrency.py` (or create it if not present),
* remove/replace usages in: `DocParsing/io.py`, `DocParsing/telemetry.py`, and any stage loops that roll their own locks.

**Actions:**

1. **Retire** `_common.acquire_lock` and any sentinel-based `acquire_lock` in `core/concurrency.py`. Replace all usages with a single helper `with FileLock(path.with_suffix(".lock")):` semantics; callers do not implement retry loops or PID checks anymore. Today’s custom implementations show busy-wait and PID probing; delete them.
2. **Remove `_acquire_lock_for()` from telemetry** (see Step 3—telemetry will inject a writer dependency). Current docs list it as a private API; post-change it should not exist.
3. **Tests**:

   * Keep your existing nested directory acquisition test but point it at the new lock (semantics should still pass: lock file appears and disappears).

**Acceptance for Step 1**

* No remaining references to sentinel-based `acquire_lock` functions.
* Telemetry no longer references `_acquire_lock_for`.

## 2) JSONL read path: replace bespoke iterator with a library

**Files:** `src/DocsToKG/DocParsing/io.py`

**Actions:**

1. **Delete** the private `_iter_jsonl_records(...)` and its call sites. The public `iter_jsonl` should be backed by the library reader but must preserve your flags:

   * `skip_invalid`: swallow line-level parse errors when `True`, else raise on first error and stop. Your current reader exposes both `skip_invalid` and a `max_errors` budget—keep both.
   * `max_errors`: stop after N decode errors even when `skip_invalid=True`.
   * `start`/`end`: preserve current write-time index slicing behavior (row-number based, not byte offsets), as the public signature already supports it.
2. **Keep** `iter_jsonl_batches` but change its implementation to rely on the new `iter_jsonl`; preserve batch size validation and buffering semantics (you already validate `batch_size > 0`).
3. **Deprecation path**: `jsonl_load` must remain deprecated and internally call `iter_jsonl` while emitting a warning (you already started doing this); do not reintroduce eager loads in new call sites.
4. **Manifest helpers**: keep `manifest_append` calling `jsonl_append_iter(..., atomic=True)`; do not add ad-hoc loops. You’ve already wired `atomic=True` in a recent change; make that the non-optional default for telemetry/manifest writes (see Step 3).

**Acceptance for Step 2**

* `_iter_jsonl_records` no longer exists.
* All JSONL reads go through `iter_jsonl`/`iter_jsonl_batches` and pass the existing tests for `skip_invalid`, `max_errors`, `start`, `end` (or equivalent new tests).

## 3) JSONL write path & telemetry: lock-aware writer injection

**Files:**

* `src/DocsToKG/DocParsing/telemetry.py`
* `src/DocsToKG/DocParsing/io.py` (only for write calls—implementation remains your hardened atomic append)

**Actions:**

1. **Define a writer dependency** for `StageTelemetry` (constructor arg): a callable `writer(path, rows)` that appends one or more rows to a JSONL file **under a lock** and with **atomic append**. The default implementation should:

   * Acquire `FileLock(path.with_suffix(".lock"))`.
   * Call `jsonl_append_iter(path, rows, atomic=True)` (you already changed telemetry to pass `atomic=True`).
2. **Remove `_acquire_lock_for()`** and `_append_payload()`’s reliance on it. `_append_payload()` should now just delegate to the injected writer. Current docs show `_acquire_lock_for` and `_append_payload`; the latter already appends via `jsonl_append_iter`.
3. **Ensure parent dir creation** remains guaranteed via your existing primitives (`jsonl_append_iter` ensures parent dir on write, or rely on `atomic_write` for whole-file writes).
4. **Tests**:

   * Concurrency: simulate two concurrent writers appending to the same manifest; assert there is no interleaving corruption and both rows appear exactly once.
   * Interruption safety: you already have a test for atomic failure preserving manifest contents—retain it (or adapt paths).

**Acceptance for Step 3**

* `StageTelemetry` has no private file-lock helper; writes route through the injected writer.
* Telemetry append paths always use atomic appends.

---

# Cross-cutting edits

1. **Delete list (post-migration)**

   * `DocsToKG.DocParsing._common.acquire_lock` and any similarly named function in `core/concurrency.py`.
   * `DocsToKG.DocParsing.io._iter_jsonl_records` and any direct JSON parse loops in writers.
   * `DocsToKG.DocParsing.telemetry._acquire_lock_for`.

2. **Keep list**

   * `atomic_write(path)`—unchanged and used for full-file writes; it already creates parent dirs.
   * `jsonl_append_iter(..., atomic=True)`—unchanged (recent hardening to single-shot append+fsync is correct).

3. **Docs / API reference**

   * Update API docs: remove references to `_acquire_lock_for` and `_iter_jsonl_records`; ensure the telemetry page lists the injected writer behavior instead. Current docs enumerate those private functions.
   * Mark `jsonl_load` as deprecated in docs (you’ve already added a deprecation string).

---

# Risk notes & mitigations

* **Behavioral parity of readers**: Your readers currently implement `start`/`end` slicing, `skip_invalid`, and `max_errors`. When replacing internals with a library, preserve these flags (implement index slicing around the library iterator as needed).
* **Atomicity expectations**: You recently changed manifest appends to default `atomic=True`. Ensure all telemetry and manifest paths pass `atomic=True` so the same guarantees remain.
* **Lock contention**: `FileLock` semantics differ from your busy-wait loop; set reasonable default timeouts (library default is usually fine). Remove PID-probing logic (no longer needed).

---

# Test plan (what to run / (re)write)

1. **Lock creation in nested dirs** (exists today) still passes under `FileLock`.
2. **Atomic append interruption safety** (exists): simulate `os.write` failure and assert prior rows remain intact.
3. **Reader parity**:

   * `skip_invalid=True` continues past bad lines; `skip_invalid=False` raises on first bad line.
   * `max_errors` stops after N decode failures.
   * `start`/`end` bounds respected.
4. **Telemetry writer injection**:

   * Default writer is used when none provided.
   * Custom writer can be injected in tests (e.g., to capture calls).
   * No `_acquire_lock_for` attribute remains on telemetry.

---

# Acceptance criteria (definition of “done”)

* **No references** to `DocsToKG.DocParsing._common.acquire_lock` or `DocsToKG.DocParsing.telemetry._acquire_lock_for` anywhere in the tree.
* **No `_iter_jsonl_records`** symbol in `DocParsing/io.py`; `iter_jsonl`/`iter_jsonl_batches` provide the entire streaming read API.
* **All manifest/telemetry appends** route through `jsonl_append_iter(..., atomic=True)` behind a `FileLock`.
* **`atomic_write` retained** as the canonical whole-file write with parent creation guarantees.
* **Docs updated**: private helpers removed; `jsonl_load` documented as deprecated.

---

# Work breakdown (small, reviewable commits)

1. **Commit A — Add deps & adapters**

   * Add `filelock` and `jsonlines` to dependencies; introduce a tiny internal adapter in `io.py` to wrap the library reader and preserve `skip_invalid`, `max_errors`, `start`, `end`.

2. **Commit B — Replace bespoke locking**

   * Remove `_common.acquire_lock` and replace all usages with `FileLock` (centralized in `core/concurrency.py` if you keep a helper). Update imports in all call sites.

3. **Commit C — Remove `_iter_jsonl_records`**

   * Delete the private iterator and refactor `iter_jsonl`/`iter_jsonl_batches` to use the adapter around the library. Ensure `jsonl_load` still warns and delegates to `iter_jsonl`.

4. **Commit D — Telemetry writer injection**

   * Add `writer` dependency to `StageTelemetry` (defaulting to lock+atomic append). Remove `_acquire_lock_for` and make `_append_payload` delegate to the writer. Update tests.

5. **Commit E — Cleanups & docs**

   * Delete any dead code paths in IO/telemetry; update Sphinx/markdown API references to remove private helpers and to reflect the new telemetry write flow.

---

# Rollback plan

* If the library reader triggers unforeseen issues, revert to the previous `iter_jsonl` implementation by restoring `_iter_jsonl_records` from Git history and temporarily keeping the `jsonlines` dependency for future re-attempts.
* If `FileLock` introduces contention problems, you can temporarily shim a `concurrency.acquire_lock` that wraps `FileLock` but accepts the old timeout value, then revisit backoff parameters.

---

# Why this reduces code & risk (immediate wins)

* **Deletes** custom lock PIDs/loops in at least two places and a private JSONL iterator that re-implements a well-solved problem.
* **Keeps** your hardened atomic append and directory-safe `atomic_write` unchanged, so integrity guarantees remain.
* **Simplifies telemetry** to a single injected writer (clean seams for testing), rather than mixing locking and append logic inside the class.

---

If you want next, I can draft **PR titles + commit messages**, plus a **CI lint** that forbids any new references to `_iter_jsonl_records` or `_acquire_lock_for`, and flags “.lock” sentinel writes in diffs.
