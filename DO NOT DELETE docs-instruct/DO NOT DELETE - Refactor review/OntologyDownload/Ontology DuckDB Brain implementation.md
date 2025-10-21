Absolutely — here’s a **best-in-class, code-free, agent-ready** implementation plan to make **DuckDB the “brain”** of `src/DocsToKG/OntologyDownload`, while keeping the **filesystem for the bytes**. It’s written so an AI programming agent can execute step-by-step without guesswork.

---

# 0) North-star & Definition of Done

**Goal:** OntologyDownload has a **transactional, queryable, audit-friendly** catalog backed by DuckDB. Files (archives, extracted files, audit JSON) stay on the **local filesystem** with atomic writes; **DuckDB becomes the source of truth** for everything you want to *answer quickly*:

* Versions, artifacts, extracted files, validations, events
* **LATEST** pointer (authoritative)
* Doctor/Prune (reconcile FS↔DB without guesswork)
* Deltas & stats (what changed; how big; pass/fail by validator)

**Boundaries are atomic:**

* Only **after** a filesystem operation succeeds do we **commit** the DuckDB rows for that boundary.

**Observability:** every boundary emits consistent events (`db.tx.*`, `extract.*`, `storage.*`) with `{run_id, config_hash}`.

---

# 1) Module layout (exact files & responsibilities)

```
src/DocsToKG/OntologyDownload/
  catalog/
    __init__.py
    connection.py           # writer/reader connectors, PRAGMAs, writer file lock
    migrations.py           # idempotent runner for 0001..000N
    repo.py                 # high-level API (upserts, bulk inserts, queries)
    boundaries.py           # transaction helpers for download/extract/validate/latest
    prune.py                # FS↔DB set-diff & GC based on staging view
    doctor.py               # reconciliation & fixers (DB or FS side)
    profile.py              # EXPLAIN ANALYZE helpers, timing wrappers
    migrations/
      0001_init.sql
      0002_files.sql
      0003_validations.sql
      0004_events.sql            # optional
      0005_staging_prune.sql
      0006_views.sql
      0007_delta_macros.sql      # version-to-version deltas (files/renames/formats/validation)
  storage/
    localfs_duckdb.py       # (if you’re adopting the “DuckDB-backed STORAGE” facade)
  settings.py               # DuckDBSettings + StorageSettings hooks
  observability/events.py   # emit & buffer (stdout JSON + DuckDB flush)
  cli/db_cmd.py             # db latest|versions|files|stats|doctor|prune|migrate|backup
```

> If you already created the DDLs, point `migrations.py` at that folder and keep the names stable; adding future DDL is then trivial.

---

# 2) Settings (strict Pydantic; all feed `config_hash`)

**DuckDBSettings**

* `path: Path = <root>/.catalog/ontofetch.duckdb`
* `threads: int = min(8, CPU)`
* `readonly: bool = False`
* `writer_lock: bool = True`               # serialize writers with a `.duck.lock` file
* `parquet_events: bool = False`           # store events as parquet instead of table (optional)

**StorageSettings** (local bytes + optional JSON `LATEST` mirror)

* `root: Path = <project>/ontologies`
* `latest_name: str = "LATEST.json"`
* `write_latest_mirror: bool = True`

**Validation/normalization**

* Create parent dirs, normalize to absolute POSIX, include in `config_hash`.

---

# 3) Schema (canonical; one source of truth)

*(Matches the migrations you’ve already sketched; included here for completeness.)*

* `schema_version(version TEXT PK, applied_at TIMESTAMP)`
* `versions(version_id TEXT PK, service TEXT, created_at TIMESTAMP, plan_hash TEXT)`
* `artifacts(artifact_id TEXT PK, version_id TEXT, service TEXT, source_url TEXT, etag TEXT, last_modified TIMESTAMP, content_type TEXT, size_bytes BIGINT, fs_relpath TEXT, status TEXT CHECK status IN ('fresh','cached','failed'), UNIQUE(version_id, fs_relpath))`
* `extracted_files(file_id TEXT PK, artifact_id TEXT, version_id TEXT, relpath_in_version TEXT, format TEXT, size_bytes BIGINT, mtime TIMESTAMP, cas_relpath TEXT, UNIQUE(version_id, relpath_in_version))`
* `validations(validation_id TEXT PK, file_id TEXT, validator TEXT, passed BOOLEAN, details_json JSON, run_at TIMESTAMP)`
* `latest_pointer(slot TEXT PK DEFAULT 'default', version_id TEXT, updated_at TIMESTAMP, by TEXT)`
* `events(run_id TEXT, ts TIMESTAMP, type TEXT, level TEXT, payload JSON)` *(optional if you prefer parquet)*
* `staging_fs_listing(scope TEXT, relpath TEXT, size_bytes BIGINT, mtime TIMESTAMP)` *(for prune/doctor)*

**Views**

* `v_version_stats` (files, bytes, validations passed/failed)
* `v_latest_files` (files for `latest_pointer`)
* `v_validation_failures`, `v_artifacts_status`, `v_latest_formats`
* Delta macros in `0007_delta_macros.sql`: `version_delta_files`, `…_rename_aware`, `…_summary`, `…_formats`, `version_validation_delta`

---

# 4) Connection & concurrency (connection.py)

* **One writer** per process acquired under a **file lock**: `<db>.lock`. Keep lock for the process lifetime (simplest & safest).
* `duckdb.connect(path, read_only=False)` for **writer**; `read_only=True` for **reader** connections when you need them.
* PRAGMAs on open:

  * `PRAGMA threads = settings.db.threads;`
* Context manager helper: `with writer_tx(boundary="extract"):` yields a cursor with `BEGIN`/`COMMIT` and `ROLLBACK` on exceptions; emits `db.tx.commit|rollback`.

---

# 5) Migration runner (migrations.py)

* At module init or first write, call `apply_pending_migrations(db_path, migrations_dir)`.
* Determine applied versions from `schema_version`; apply pending `000N_*.sql` **in a single TX**, recording each in `schema_version`.
* CLI: `ontofetch db migrate --dry-run` prints unapplied files; `--apply` runs them.

---

# 6) High-level repo API (repo.py)

### Upserts & inserts

* `upsert_version(version_id, service, plan_hash=None) -> None`
* `upsert_artifact(artifact: ArtifactMeta) -> None`
* `insert_extracted_files(rows: Iterable[ExtractedFileMeta]) -> None`
  *(Use DuckDB Appender or Arrow to batch insert 5–50k at a time.)*
* `insert_validations(rows: Iterable[ValidationMeta]) -> None`

### Queries

* `get_latest() -> str|None`
* `set_latest(version_id: str, by: str|None) -> None`  *(DB pointer)*
  Optional: mirror to JSON via StorageSettings if `write_latest_mirror=True`.
* `version_stats(version_id) -> VersionStats`
* `list_versions(service: str|None = None, limit: int = 50) -> list[VersionRow]`
* `list_files(version_id, format: str|None = None) -> list[FileRow]`
* `where_is(file_id) -> Path`  *(Resolve absolute FS path under root)*

> All repo methods are **writer-safe**: they require an active `writer_tx` or acquire one internally. Reads may use the reader connection.

---

# 7) Boundary choreography (boundaries.py)

**Invariant:** *Commit DB only after FS success. Never the reverse.*

### Download boundary (archive ingestion)

1. **FS**: stream → `archive.tmp` → `fsync(file)` → rename to `fs_relpath` under StorageSettings.root; `fsync(dir)`.
2. **DB TX**:

   * `upsert_version(version_id, service, plan_hash)`
   * `upsert_artifact({artifact_id=sha256(file), version_id, service, source_url, etag, last_modified, content_type, size_bytes, fs_relpath, status='fresh'|'cached'})`
3. Emit `db.tx.commit`.

### Extraction boundary

1. **FS**: use your new **secure extraction** pipeline (libarchive two-phase) to write `data/**` + optional audit JSON atomically under `<root>/<service>/<version>/…`.
2. **DB TX** (bulk): `insert_extracted_files(rows)` where each row references `artifact_id` & `version_id`.
3. Emit `db.tx.commit`.

### Validation boundary

1. **Compute**: run validators (SHACL/ROBOT/etc.), collect results.
2. **DB TX**: `insert_validations(rows)` (batch).
3. Emit `db.tx.commit`.

### Set latest boundary

1. **DB TX**: `set_latest(version_id, by=user/host)`
2. If `write_latest_mirror=True`: write `LATEST.json.tmp` → rename → `fsync(dir)` *(mirror only; DB is authoritative)*.
3. Emit `db.tx.commit`.

---

# 8) Observability (events & structured logs)

* Every boundary emits:

  * `db.tx.commit|rollback {boundary, rows, ms}`
  * `storage.put|mv|delete|latest.set` (if you adopt the STORAGE facade)
  * `extract.*` events from the extraction pipeline (already planned)
* Event sink strategy:

  * **Immediate** JSON to stdout (structlog)
  * **Buffered** to DuckDB `events` table (or parquet) flushed at boundary completion & at CLI exit

---

# 9) Doctor & Prune (doctor.py, prune.py)

### Prune

* FS→DB **set diff** using `staging_fs_listing`:

  1. Walk `<root>` and load `scope='version'` paths into `staging_fs_listing` (truncate before load).
  2. `v_fs_orphans` = `staging_fs_listing` minus (`artifacts.fs_relpath` ∪ `service||'/'||version||'/'||relpath_in_version`).
  3. **Dry-run**: show counts & bytes; **Apply**: delete FS orphans in small batches with progress logs.
  4. Emit `prune.*` events.

### Doctor

* Reconcile inconsistencies:

  * **DB row present, file missing**: offer to drop rows or mark as failed (audit friendly).
  * **File present, DB row missing**: offer quick-insert (artifact or file row) or delete file.
  * **LATEST mismatch**: DB pointer ≠ JSON mirror → set JSON to DB (or vice-versa via flag).
* CLI: `ontofetch db doctor --fix` with clear prompts or `--yes` for CI.

---

# 10) CLI (Typer) — db_cmd.py

* `ontofetch db migrate [--dry-run|--apply]`
* `ontofetch db latest [get|set <version>]`
* `ontofetch db versions [--service X] [--limit N] [--format table|json]`
* `ontofetch db files --version <v> [--format ttl|rdf|owl|obo]`
* `ontofetch db stats --version <v>`
* `ontofetch db delta summary A B` *(uses macros)*
  also: `delta files A B`, `delta renames A B`, `delta formats A B`, `delta validation A B`
* `ontofetch db doctor [--fix]`
* `ontofetch db prune [--dry-run|--apply]`
* (optional) `ontofetch db backup` → `CHECKPOINT` then copy the `.duckdb` (plus schemas) to a timestamped folder

All commands support: `--format`, `-v/-vv`, `--profile` (prints EXPLAIN plans), and read settings from the traced loader (so `config_hash` is reported in events).

---

# 11) Tests (fast, deterministic)

**Unit**

* Migration runner idempotent; `schema_version` updated in order.
* Repo upserts: re-running does not duplicate rows.
* Latest pointer: set/get; JSON mirror optional & atomic.

**Component**

* Download boundary: FS rename then DB commit; simulate FS failure → DB **not** committed.
* Extraction boundary: bulk insert 10–50k file rows via Arrow/appender (no Python row loops).
* Validation boundary: insert rows; `v_version_stats` matches expected totals.

**Doctor/Prune**

* Load a staged FS listing; `v_fs_orphans` lists the right paths; `--apply` deletes only orphans.
* Mismatch scenarios: doctor fix applies the intended direction.

**Delta macros**

* Small fixture for A/B versions; `delta summary/files/renames/formats/validation` give stable, correct results.

**Concurrency**

* Writer lock prevents concurrent writers; readers can read concurrently.
* Crash between FS write & DB commit: doctor recovers in one step.

**Performance sanity**

* Bulk insert 50k `extracted_files` rows **< 1.5s** (CI class)
* `v_version_stats` on 200k rows **< 200ms** (measure & adjust per runner)

---

# 12) Performance & profiling (profile.py)

* `with profile_query(conn, sql, params, enable=flag):` wrapper prints `EXPLAIN ANALYZE` when `--profile` enabled.
* For heavy reports (deltas), ensure correct join order and indexes (DuckDB will choose well; inspect plan to confirm hash joins on keys).
* Consider materializing very heavy views on demand with a TTL if you later need instant dashboards.

---

# 13) Rollout plan (small PRs, minimal risk)

**PR-DB1 — Bootstrapping**

* Add `connection.py`, `migrations.py`, migrations 0001..0006; `apply_pending_migrations` on startup.
* Add `repo.py` with `upsert_version`, `upsert_artifact`, `set/get_latest`, and minimal queries.
* Add `db_cmd.py` with `migrate`, `latest`.

**PR-DB2 — Boundaries**

* Implement `boundaries.py` download/extract/validate/set_latest with correct **FS→DB** choreography.
* Emit `db.tx.*` events.

**PR-DB3 — Bulk ingest & deltas**

* `insert_extracted_files` via Arrow/appender; migrate `0007_delta_macros.sql`; add `db delta` CLI.

**PR-DB4 — Doctor/Prune**

* `staging_fs_listing`, `v_fs_orphans`, prune & doctor CLIs; tests.

**PR-DB5 — Docs & polish**

* README sections (schema, commands, guarantees); examples; add `--profile` guidance; perf budgets in docs.

---

# 14) Acceptance checklist (paste into PR)

* [ ] DuckDB migrations applied idempotently; schema at head.
* [ ] Writer lock enforced; single writer per process; readers safe.
* [ ] Boundaries commit DB **after** FS success; roll back on failure.
* [ ] `versions`, `artifacts`, `extracted_files`, `validations`, `latest_pointer` populated as expected.
* [ ] `set_latest/get_latest` work; JSON mirror optional and atomic.
* [ ] `db delta` commands run & return correct summaries/renames/format diffs/validation deltas.
* [ ] `doctor` & `prune` reconcile safely; `--dry-run` and `--apply` covered by tests.
* [ ] Bulk inserts use Appender/Arrow (no Python loops); perf budgets met on CI runners.
* [ ] Events emitted (`db.tx.*`, `storage.*` if used) with `{run_id, config_hash}` and flushed at boundary end.
* [ ] Docs updated; CLI help & examples accurate; `--profile` prints EXPLAIN plans.

---

## Why this works

* Filesystem remains the **brawn** (atomic bytes), DuckDB the **brain** (transactions + queries).
* Boundaries are **provably atomic**, observability is **first-class**, and the catalog answers every Monday-morning question in seconds: *what changed, what’s latest, what can we delete, what failed, how big is it*.

If you’d like, I can turn this into scaffold stubs (empty modules with method signatures & TODOs) so you can open PR-DB1 in minutes.
