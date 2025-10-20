# DuckDB Implementation Plan for `OntologyDownload` (code-free, agent-ready)

Below is a **detailed, end-to-end plan** to adopt **DuckDB** as the *local catalog* for ontology artifacts (versions, archives, extracted files, validations, provenance, “latest” pointer, pruning). Binary payloads still live on the **filesystem** for streaming/atomicity; DuckDB stores **metadata + lineage**. No public call-site changes are required beyond wiring calls at the natural pipeline boundaries.

---

## 0) Goals, Non-Goals, and Operating Model

**Goals**

* Single-node, zero-ops, transactional catalog of: versions, artifacts, extracted files, validations, and events.
* Fast queries (counts, deltas, pruning, audits) with **idempotent** re-runs.
* Atomic boundaries that keep **DB and filesystem consistent**.

**Non-Goals**

* Not a replacement for your RDF/SPARQL engine. DuckDB only catalogs metadata.
* Not a blob store; bytes remain on disk (atomic rename, CAS optional).

**Operating Model**

* **One writer at a time** (process lock), many readers.
* DB file: `<root>/.catalog/ontofetch.duckdb`.

---

## 1) Configuration & Bootstrapping

**Settings (Pydantic)**

* `ONTOFETCH_DB_PATH` (default: `<root>/.catalog/ontofetch.duckdb`)
* `ONTOFETCH_DB_THREADS` (default: `min(8, cpu_count)`)
* `ONTOFETCH_DB_READONLY` (bool, default `false`)
* `ONTOFETCH_DB_WLOCK` (bool, default `true`) → enable a *file lock* `<db>.lock` for writer serialization
* `ONTOFETCH_DB_PARQUET_EVENTS` (bool, default `false`) → store events as Parquet and attach on demand (optional)

**Boot Sequence**

1. Create `<root>/.catalog/` if absent.
2. Acquire **exclusive file lock** if write-mode.
3. Open DuckDB connection; set:

   * `PRAGMA threads = ONTOFETCH_DB_THREADS`
   * `PRAGMA memory_limit = 'auto'` (default), tweak later if needed
4. Run **migrations** (Section 8). If DB is new, create schema.

**Acceptance**

* First run creates file; subsequent read-only runs succeed without lock.

---

## 2) Filesystem Layout (unchanged for blobs)

* Root: `…/ontologies/`
* Version tree: `…/ontologies/<service>/<version>/…`
* Optional CAS: `…/ontologies/.cas/sha256/<aa>/<bb>/<digest>`
* Audit file per extraction: `…/<service>/<version>/.extract.audit.json`
* Latest marker (FS): `…/ontologies/LATEST.json` (kept in sync with DB pointer)

---

## 3) Catalog Schema (DuckDB)

> DuckDB supports `PRIMARY KEY`/`UNIQUE`. **Foreign-key enforcement is limited**; enforce referential integrity in code + periodic checks.

**3.1 `schema_version`**

* `version` TEXT PRIMARY KEY
* `applied_at` TIMESTAMP

**3.2 `versions`**

* `version_id` TEXT PRIMARY KEY
* `service` TEXT NOT NULL
* `created_at` TIMESTAMP NOT NULL
* `plan_hash` TEXT NULL
* **Computed views** later for deltas & stats.

**3.3 `artifacts`** (one row per downloaded archive)

* `artifact_id` TEXT PRIMARY KEY        — sha256 of archive bytes
* `version_id` TEXT NOT NULL            — references `versions.version_id` (check in code)
* `service` TEXT NOT NULL
* `source_url` TEXT NOT NULL            — normalized URL
* `etag` TEXT NULL
* `last_modified` TIMESTAMP NULL
* `content_type` TEXT NULL
* `size_bytes` BIGINT NOT NULL
* `fs_relpath` TEXT NOT NULL            — path under root to the archive
* `status` TEXT NOT NULL CHECK (status IN ('fresh','cached','failed'))
* UNIQUE (`version_id`,`fs_relpath`)

**3.4 `extracted_files`** (one per regular file after extraction)

* `file_id` TEXT PRIMARY KEY            — sha256 of file bytes
* `artifact_id` TEXT NOT NULL
* `version_id` TEXT NOT NULL
* `relpath_in_version` TEXT NOT NULL
* `format` TEXT NOT NULL                — rdf|ttl|owl|obo|other
* `size_bytes` BIGINT NOT NULL
* `mtime` TIMESTAMP NULL
* `cas_relpath` TEXT NULL
* UNIQUE (`version_id`,`relpath_in_version`)

**3.5 `validations`** (RDF/SHACL/ROBOT/Arelle, etc.)

* `validation_id` TEXT PRIMARY KEY      — `sha256(file_id + validator + run_at)` or ULID
* `file_id` TEXT NOT NULL
* `validator` TEXT NOT NULL
* `passed` BOOLEAN NOT NULL
* `details_json` JSON NULL              — compact message or summary
* `run_at` TIMESTAMP NOT NULL
* INDEX on (`file_id`), (`validator`, `run_at`)

**3.6 `latest_pointer`**

* `slot` TEXT PRIMARY KEY DEFAULT 'default'
* `version_id` TEXT NOT NULL
* `updated_at` TIMESTAMP NOT NULL
* `by` TEXT NULL

**3.7 `events`** (optional; for observability rollups)

* `run_id` TEXT NOT NULL
* `ts` TIMESTAMP NOT NULL
* `type` TEXT NOT NULL                  — extract.start|pre_scan.done|extract.done|extract.error|…
* `level` TEXT NOT NULL                 — INFO|WARN|ERROR
* `payload` JSON NOT NULL

> If `ONTOFETCH_DB_PARQUET_EVENTS=true`, events go to Parquet files under `…/.catalog/events/YYYY/MM/DD/*.parquet` and are **attached** ad hoc in queries.

---

## 4) Transaction Boundaries (DB ↔ FS atomic choreography)

> Treat each boundary as a **mini two-phase commit**: write FS, then commit DB.

**4.1 After successful *download* (archive file renamed into place)**

* Begin TX
* Upsert `versions` (if not present)
* Upsert `artifacts` with status = 'fresh' | 'cached', size, etag, fs path, etc.
* Commit

**4.2 After successful *extraction***

* Begin TX
* Bulk insert `extracted_files` rows for this `artifact_id` & `version_id`
* (Optional) Upsert *file CAS mapping* if you maintain CAS hardlinks
* Commit

**4.3 After *validations***

* Begin TX
* Insert rows into `validations` (batch)
* Commit

**4.4 When *setting latest***

* Begin TX
* Upsert `latest_pointer`
* Write `LATEST.json` on FS via temp + rename

  * If FS write fails → **ROLLBACK**
  * If FS write succeeds → **COMMIT**

**Recovery/Doctor**

* On startup or via `doctor` command:

  * Scan for **incomplete boundary** symptoms (e.g., archive on FS without `artifacts` row; `extracted_files` missing for a known artifact; latest mismatch between DB and JSON)
  * Offer **fix** actions: reconcile DB from FS (or vice versa), quarantining ambiguous cases

---

## 5) Idempotence, Dedup & Overwrite Semantics

* Identity:

  * `artifact_id = sha256(archive)`
  * `file_id = sha256(file bytes)`
* Re-runs:

  * If same `artifact_id` already present → **skip archive write**, update `status` if needed, ensure `artifacts` row consistent.
  * Extraction: if (`version_id`,`relpath_in_version`) exists:

    * Respect **overwrite policy** (`reject`|`replace`|`keep_existing`) and reflect outcome:

      * `replace` → update row (`size_bytes`, `mtime`, `file_id`)
      * `keep_existing` → skip insert; optionally record a no-op event

---

## 6) Prune & Garbage Collection

**Goal**: remove **orphaned blobs** safely.

**Dry-run Algorithm**

1. Live set A (from DB):

   * `artifact_id` in `artifacts` (status != 'failed')
   * `file_id` in `extracted_files`
2. Disk set B:

   * CAS digests on disk (if using CAS)
   * Version files under each `<service>/<version>` tree
3. Compute `(B − A)` by path or digest
4. Report counts, bytes, and sample paths

**Apply**

* Delete in **small batches** with progress logs
* For version trees, refuse to delete if the owning `version_id` is **current latest** unless `--force`
* Post-delete: re-compute a small sample to confirm deletion

**DB Cleanup**

* Optionally annotate deleted rows or keep DB as the source of truth and only remove FS files (recommended: keep DB rows for audit).

---

## 7) Query API Surface (internal facades)

> Keep return shapes stable; do not leak SQL to callers.

* `db_latest_version() -> str | None`
* `db_set_latest(version_id: str, by: str | None) -> None`
* `db_upsert_version(version_id, service, plan_hash?) -> None`
* `db_upsert_artifact(artifact_meta) -> None`
* `db_insert_extracted_files(files: Iterable[FileMeta]) -> None` (batched)
* `db_insert_validations(entries: Iterable[ValidationMeta]) -> None`
* `db_version_stats(version_id) -> VersionStats`
* `db_list_versions(limit?, service?) -> List[VersionRow]`
* `db_list_files(version_id, format?) -> List[FileRow]`
* `db_where_is(file_id) -> LocalPath` (CAS or version tree)
* `db_prune_plan() -> PruneReport` (returns orphans list/bytes for dry-run)

---

## 8) Migrations & Schema Versioning

* Maintain SQL migration files `migrations/0001_init.sql`, `0002_add_validations.sql`, …
* On boot:

  * Read current `schema_version`
  * Apply *forward* migrations in a transaction
  * Insert new `schema_version` row
* Add **compatibility views** for renamed columns to keep readers stable across releases.

---

## 9) Observability & Events

* Emit **structured events** to both:

  * Logs (`extract.*` events already planned)
  * DB `events` table **or** Parquet logs (configurable)
* Minimum events to write to DB: `extract.start`, `pre_scan.done`, `extract.done`, `extract.error`, `audit.emitted`
* Provide ad-hoc SQL examples:

  * “Top N versions by bytes_written”
  * “Validation failure rates by validator”
  * “Files added/removed between version X and Y”

---

## 10) Concurrency & Locks

* Writer obtains `<db>.lock` (file lock) for the duration of the **process** or individual boundary operations (simpler: hold for process, release on exit).
* Readers open DuckDB read-only; no lock required.
* If a writer dies unexpectedly:

  * Lock file is released (advisory locks) or cleaned on next boot with timeout heuristic.
* Document that *multiple parallel writers are not supported*; serialize via orchestrator or lock.

---

## 11) Performance Guidance

* Use **batched inserts** with DuckDB appender semantics (from Python API) for `extracted_files` and `validations`.
* Keep transactions **short** (bounded by a boundary), but group writes within the boundary.
* Consider **sorting** large tables periodically for scan locality (optional; DuckDB columnar storage already helps).
* `CHECKPOINT` after heavy loads (optional) to compact.

---

## 12) CLI Hooks (optional quality-of-life)

Add read-only subcommands that query DuckDB:

* `ontofetch db latest`
* `ontofetch db versions [--service=…]`
* `ontofetch db files --version <v> [--format=ttl]`
* `ontofetch db stats --version <v>`
* `ontofetch db doctor --fix` (FS↔DB reconciliation)
* `ontofetch db prune --dry-run | --apply`

All implemented via the internal facades; **no** SQL in CLI code.

---

## 13) Testing Matrix

**Unit**

* Bootstrap new DB; assert `schema_version`, tables exist.
* Upsert version/artifact; assert idempotence (no duplicates).
* Bulk insert extracted files; assert UNIQUE constraints and overwrite policy.
* Validations inserts; query failure rates.

**Integration**

* Full flow: download → DB artifacts → extract → DB files → validate → DB validations → set latest → write marker → read latest
* Crash simulations:

  * After archive FS write, **before** DB commit → doctor reconciles on next run
  * After DB commit, **before** marker rename → TX rolled back by choreography; marker remains consistent

**Property-based**

* Two identical extractions are idempotent (row counts stable; no duplicate files).
* Tightening overwrite policy never increases number of *writes*.

**Prune**

* Create synthetic orphans; dry-run and apply; verify deletions and DB rows remain.

**Performance**

* ~10k file entries commit in < threshold time (set per CI runner).

---

## 14) Execution Plan (thin PRs)

**PR-A: Boot + Schema**

* Settings, lock, connection, PRAGMAs
* Migrations 0001 (versions, artifacts, latest_pointer)
* Facades: upsert_version, upsert_artifact, latest get/set

**PR-B: Extraction Writes**

* Table `extracted_files`
* Batched insert facade
* Wire extraction boundary to DB commit

**PR-C: Validations**

* Table `validations`, insert facade
* Wire validators boundary

**PR-D: Events (optional)**

* Table `events` or Parquet path; write minimal extract.* events

**PR-E: Prune**

* Facade to compute orphans from DB vs FS
* CLI `prune --dry-run|--apply`

**PR-F: Doctor**

* Reconciliation scans & fixes
* CLI `doctor --fix` (optional)

**PR-G: CLI Queries**

* `db latest|versions|files|stats`

---

## 15) Acceptance Checklist

* [ ] DB file created at `ONTOFETCH_DB_PATH`; writer locking enforced.
* [ ] Schema ready with `schema_version`, `versions`, `artifacts`, `extracted_files`, `validations`, `latest_pointer` (events optional).
* [ ] Pipeline boundaries commit **after** FS success; DB↔FS consistency maintained.
* [ ] Idempotent re-runs: no duplicate rows; overwrite policy honored.
* [ ] `latest_pointer` in DB and `LATEST.json` on disk are kept in sync.
* [ ] `prune` dry-run & apply remove only orphans; logs/audits reflect actions.
* [ ] Query facades cover versions, files, stats, latest; no SQL leaks to callers.
* [ ] Test suite covers bootstrap, boundaries, idempotence, prune, and performance budget.

---

### TL;DR

Use DuckDB as the **brains** (catalog & analytics) and your local filesystem as the **brawn** (atomic, streaming blobs). Wire writes at pipeline boundaries, keep DB↔FS in lock-step, and you’ll gain **fast queries, safe pruning, clean provenance**, and **idempotent** operations—all locally, with minimal operational burden.
