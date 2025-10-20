Awesome — here’s a **complete, implementation-ready spec** for your **data layout (filesystem)** and **DuckDB schemas**. It’s designed so an AI agent can create the DB, write migrations, and wire the pipeline with zero guesswork.

---

# 1) Filesystem data layout (bytes-on-disk)

> Keep blobs on disk for streaming + atomic renames. DuckDB catalogs the metadata.

## 1.1 Root & naming

* **Root**: `<ONTODOC_ROOT>/ontologies/`
* **Encapsulation** (recommended): every extraction lands under a deterministic subdir:

  * `<ONTODOC_ROOT>/ontologies/<service>/<version>/`  ← human-friendly path
  * Optionally mirror into **CAS** for dedupe: `<ONTODOC_ROOT>/ontologies/.cas/sha256/<aa>/<bb>/<digest>`
* **Audit** (per extraction):
  `<ONTODOC_ROOT>/ontologies/<service>/<version>/.extract.audit.json`
* **Latest marker** (human-friendly pointer):
  `<ONTODOC_ROOT>/ontologies/LATEST.json`

### 1.2 Example tree

```
ontologies/
  LATEST.json
  .cas/
    sha256/aa/bb/aa…bb…digest       # optional content-addressed copies (or reflinks)
  OLS/
    2025-10-20T01-23-45Z/           # version
      src/archives/obo-basic.zip    # original archive(s)
      src/archives/…                # more
      data/…                        # extracted files (ttl/rdf/owl/obo/…)
      .extract.audit.json           # provenance & metrics
  BioPortal/
    2025-10-15/…
```

**Atomic write rule**: always stream to a temp file in the final directory and **rename** into place.

---

# 2) DuckDB catalog (brains / metadata)

> Use DuckDB as the **authoritative catalog** for versions, artifacts, extracted files, validations, “latest”, and (optionally) events.
> **Conventions**:
>
> * Types: `TEXT`, `BIGINT`, `BOOLEAN`, `TIMESTAMP`, `JSON`. (All `TIMESTAMP` values are **UTC**.)
> * **Primary keys and UNIQUE** are enforced. Treat `FOREIGN KEY` as informational (enforce in code).
> * All table names are `snake_case`.
> * All sizes are **bytes**.

## 2.1 Schema DDL (migrations 0001…0004)

### 0001_init.sql — core entities

```sql
-- schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
  version       TEXT PRIMARY KEY,
  applied_at    TIMESTAMP NOT NULL
);

-- a logical release of a fetched corpus
CREATE TABLE IF NOT EXISTS versions (
  version_id    TEXT PRIMARY KEY,        -- canonical version label (e.g., 2025-10-20T01:23:45Z)
  service       TEXT NOT NULL,           -- OLS, BioPortal, <publisher_key>, etc.
  created_at    TIMESTAMP NOT NULL,      -- when this version was first materialized
  plan_hash     TEXT                     -- optional: hash of the plan manifest used
);

-- one row per downloaded archive
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id   TEXT PRIMARY KEY,        -- sha256 of the archive bytes
  version_id    TEXT NOT NULL,
  service       TEXT NOT NULL,
  source_url    TEXT NOT NULL,           -- normalized URL (punycoded, path-clean)
  etag          TEXT,                    -- HTTP ETag if present
  last_modified TIMESTAMP,               -- HTTP Last-Modified if present
  content_type  TEXT,                    -- server content-type
  size_bytes    BIGINT NOT NULL,         -- archive size on disk
  fs_relpath    TEXT NOT NULL,           -- path under ontologies/ root (POSIX)
  status        TEXT NOT NULL CHECK (status IN ('fresh','cached','failed')),

  UNIQUE (version_id, fs_relpath)
  -- FOREIGN KEY (version_id) REFERENCES versions(version_id) -- (informational)
);

-- pointer to the latest version
CREATE TABLE IF NOT EXISTS latest_pointer (
  slot         TEXT PRIMARY KEY DEFAULT 'default',
  version_id   TEXT NOT NULL,
  updated_at   TIMESTAMP NOT NULL,
  by           TEXT                      -- hostname/user/process
  -- FOREIGN KEY (version_id) REFERENCES versions(version_id)
);

INSERT OR IGNORE INTO schema_version VALUES ('0001_init', now());
```

**Indexes (DuckDB supports CREATE INDEX):**

```sql
CREATE INDEX IF NOT EXISTS idx_versions_service_created
  ON versions(service, created_at);

CREATE INDEX IF NOT EXISTS idx_artifacts_version
  ON artifacts(version_id);

CREATE INDEX IF NOT EXISTS idx_artifacts_service
  ON artifacts(service);

CREATE INDEX IF NOT EXISTS idx_artifacts_source_url
  ON artifacts(source_url);
```

---

### 0002_files.sql — extracted file catalog

```sql
-- one row per extracted regular file
CREATE TABLE IF NOT EXISTS extracted_files (
  file_id            TEXT PRIMARY KEY,   -- sha256 of file bytes
  artifact_id        TEXT NOT NULL,
  version_id         TEXT NOT NULL,
  relpath_in_version TEXT NOT NULL,      -- canonical relative path under <service>/<version>/
  format             TEXT NOT NULL,      -- rdf|ttl|owl|obo|other
  size_bytes         BIGINT NOT NULL,
  mtime              TIMESTAMP,          -- final on-disk mtime (policy-dependent)
  cas_relpath        TEXT,               -- optional: where CAS copy lives

  UNIQUE (version_id, relpath_in_version)
  -- FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id),
  -- FOREIGN KEY (version_id)  REFERENCES versions(version_id)
);

CREATE INDEX IF NOT EXISTS idx_files_version
  ON extracted_files(version_id);

CREATE INDEX IF NOT EXISTS idx_files_format
  ON extracted_files(format);

CREATE INDEX IF NOT EXISTS idx_files_artifact
  ON extracted_files(artifact_id);

INSERT OR IGNORE INTO schema_version VALUES ('0002_files', now());
```

---

### 0003_validations.sql — validator outcomes

```sql
-- validator outcomes (SHACL, ROBOT, Arelle, custom probes, etc.)
CREATE TABLE IF NOT EXISTS validations (
  validation_id  TEXT PRIMARY KEY,       -- e.g., ULID or sha256(file_id|validator|run_at)
  file_id        TEXT NOT NULL,
  validator      TEXT NOT NULL,          -- 'pySHACL', 'ROBOT', 'Arelle', 'Custom:<name>', ...
  passed         BOOLEAN NOT NULL,
  details_json   JSON,                   -- concise JSON summary (NOT raw logs)
  run_at         TIMESTAMP NOT NULL,

  -- FOREIGN KEY (file_id) REFERENCES extracted_files(file_id)
);

CREATE INDEX IF NOT EXISTS idx_validations_file
  ON validations(file_id);

CREATE INDEX IF NOT EXISTS idx_validations_validator_time
  ON validations(validator, run_at);

INSERT OR IGNORE INTO schema_version VALUES ('0003_validations', now());
```

---

### 0004_events.sql — (optional) structured events

```sql
-- append-only structured observability events (optional; can be Parquet instead)
CREATE TABLE IF NOT EXISTS events (
  run_id    TEXT NOT NULL,               -- UUID per extraction run
  ts        TIMESTAMP NOT NULL,
  type      TEXT NOT NULL,               -- extract.start|pre_scan.done|extract.done|extract.error|audit.emitted|storage.*
  level     TEXT NOT NULL,               -- INFO|WARN|ERROR
  payload   JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_time
  ON events(run_id, ts);

INSERT OR IGNORE INTO schema_version VALUES ('0004_events', now());
```

---

## 2.2 Views & rollups (read-only helpers)

> Create as **views** so downstream analytics remain stable even if table details evolve.

```sql
-- quick per-version stats (files, bytes, validations)
CREATE OR REPLACE VIEW v_version_stats AS
SELECT
  v.version_id,
  v.service,
  v.created_at,
  COUNT(f.file_id)                    AS files,
  SUM(f.size_bytes)                  AS bytes,
  SUM(CASE WHEN val.passed THEN 1 ELSE 0 END) AS validations_passed,
  SUM(CASE WHEN val.passed THEN 0 ELSE 1 END) AS validations_failed
FROM versions v
LEFT JOIN extracted_files f ON f.version_id = v.version_id
LEFT JOIN (
  SELECT file_id, BOOL_OR(passed) AS passed  -- or last result semantics
  FROM validations
  GROUP BY 1
) val ON val.file_id = f.file_id
GROUP BY 1,2,3;

-- content listing for the current latest version
CREATE OR REPLACE VIEW v_latest_files AS
SELECT f.*
FROM latest_pointer lp
JOIN extracted_files f ON f.version_id = lp.version_id;
```

**Delta helper** (X vs Y) — use as a template for a parameterized query:

```sql
-- files added/removed between two versions (template)
-- SELECT * FROM v_version_delta('2025-10-01', '2025-10-20');
CREATE OR REPLACE VIEW v_version_delta AS
SELECT
  'ADDED'  AS change, b.relpath_in_version, b.size_bytes
FROM extracted_files b
LEFT JOIN extracted_files a
  ON a.version_id = (SELECT version_id FROM latest_pointer)  -- replace in app with target A
 AND a.relpath_in_version = b.relpath_in_version
WHERE b.version_id = (SELECT version_id FROM latest_pointer) -- replace in app with target B
  AND a.file_id IS NULL
UNION ALL
SELECT
  'REMOVED' AS change, a.relpath_in_version, a.size_bytes
FROM extracted_files a
LEFT JOIN extracted_files b
  ON b.version_id = (SELECT version_id FROM latest_pointer)  -- replace in app with target B
 AND b.relpath_in_version = a.relpath_in_version
WHERE a.version_id = (SELECT version_id FROM latest_pointer) -- replace in app with target A
  AND b.file_id IS NULL;
```

> In code, substitute the two `version_id` parameters rather than relying on `latest_pointer`.

---

## 2.3 Staging tables for pruning / reconciliation

> You’ll compute the **filesystem listing** in Python, then load it into a staging table for set-difference against the catalog.

```sql
-- ephemeral: loaded before prune/doctor; truncate afterwards
CREATE TABLE IF NOT EXISTS staging_fs_listing (
  scope          TEXT NOT NULL,          -- 'cas' | 'version'
  relpath        TEXT NOT NULL,          -- path relative to ontologies/ root
  size_bytes     BIGINT NOT NULL,
  mtime          TIMESTAMP
);

-- view: orphans on disk (present in FS but not referenced by catalog)
CREATE OR REPLACE VIEW v_fs_orphans AS
SELECT s.*
FROM staging_fs_listing s
LEFT JOIN (
  SELECT fs_relpath AS relpath FROM artifacts
  UNION ALL
  SELECT CONCAT(service, '/', version_id, '/', relpath_in_version) FROM extracted_files
) cat ON cat.relpath = s.relpath
WHERE cat.relpath IS NULL;
```

---

# 3) JSON schema for `.extract.audit.json` (on disk)

> The audit record is your **tamper-evident** provenance file. Keep it compact and deterministic.

**Location**
`<ONTODOC_ROOT>/ontologies/<service>/<version>/.extract.audit.json`

**Draft-07 JSON Schema**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "OntoFetch Extract Audit",
  "type": "object",
  "required": ["schema_version", "run_id", "archive_path", "archive_sha256", "format", "metrics", "entries"],
  "properties": {
    "schema_version": { "type": "string", "enum": ["1.0"] },
    "run_id":         { "type": "string" },
    "libarchive_version": { "type": "string" },
    "archive_path":   { "type": "string" },
    "archive_sha256": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
    "format":         { "type": "string" },
    "filters":        { "type": "array", "items": { "type": "string" } },
    "policy":         { "type": "object" },   // materialized knobs (limits, normalization, overwrite, etc.)
    "metrics": {
      "type": "object",
      "required": ["entries_total", "entries_included", "entries_skipped", "bytes_declared", "bytes_written", "duration_ms"],
      "properties": {
        "entries_total":     { "type": "integer", "minimum": 0 },
        "entries_included":  { "type": "integer", "minimum": 0 },
        "entries_skipped":   { "type": "integer", "minimum": 0 },
        "bytes_declared":    { "type": "integer", "minimum": 0 },
        "bytes_written":     { "type": "integer", "minimum": 0 },
        "ratio_total":       { "type": "number",  "minimum": 0 },
        "duration_ms":       { "type": "integer", "minimum": 0 },
        "space": {
          "type": "object",
          "properties": {
            "available_bytes": { "type": "integer", "minimum": 0 },
            "needed_bytes":    { "type": "integer", "minimum": 0 },
            "margin":          { "type": "number",  "minimum": 1.0 }
          }
        }
      }
    },
    "entries": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["path_rel", "size", "sha256"],
        "properties": {
          "path_rel":  { "type": "string" },
          "scan_index":{ "type": "integer", "minimum": 0 },
          "size":      { "type": "integer", "minimum": 0 },
          "sha256":    { "type": "string",  "pattern": "^[a-f0-9]{64}$" },
          "mtime":     { "type": "string" }  // ISO-8601 UTC
        }
      }
    }
  }
}
```

**Determinism**

* Sort `entries` by either **header order** (scan_index) or **path_asc** depending on your configured mode, and **emit stable key order** in JSON.

---

# 4) Data invariants & integrity rules

* `artifacts.version_id` **must exist** in `versions` (enforce in code + doctor checks).
* `extracted_files.version_id` and `artifact_id` **must exist** (enforce in code).
* `(version_id, relpath_in_version)` **uniquely identifies** a file in the version tree.
* `file_id` (sha256) **uniquely identifies bytes** across the entire repo (dedupe/CAS possible).
* `latest_pointer.version_id` **must exist** in `versions`.
* **No write** to DB unless the corresponding **FS write** has succeeded (two-phase choreography).
* **Idempotence**: re-running with identical blobs should produce **no new rows** (DELETE+INSERT or MERGE semantics in a single TX).

---

# 5) Suggested helper views (optional but handy)

```sql
-- artifacts status overview
CREATE OR REPLACE VIEW v_artifacts_status AS
SELECT
  service, version_id,
  COUNT(*)                AS archives,
  SUM(size_bytes)         AS archive_bytes,
  SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
FROM artifacts
GROUP BY 1,2;

-- validation failure details for a version
CREATE OR REPLACE VIEW v_validation_failures AS
SELECT
  f.version_id, f.relpath_in_version, v.validator, v.run_at, v.details_json
FROM extracted_files f
JOIN validations v ON v.file_id = f.file_id
WHERE v.passed = FALSE;

-- quick head of latest files by format
CREATE OR REPLACE VIEW v_latest_formats AS
SELECT format, COUNT(*) AS files, SUM(size_bytes) AS bytes
FROM v_latest_files
GROUP BY 1 ORDER BY bytes DESC;
```

---

# 6) Practical DDL/typing notes (DuckDB)

* Use `TEXT` for URLs, paths, IDs, and hashes. (Store hashes in lowercase hex.)
* `JSON` type is stored compactly and queryable; avoid huge payloads (log links to files instead).
* All timestamps are **UTC** `TIMESTAMP`. Do not store local offsets.
* Indexes: DuckDB supports `CREATE INDEX`; they’re beneficial for joins on `version_id`, and filters on `format` or `service`.
* Foreign keys: keep `REFERENCES` in comments only; enforce referential integrity in Python + “doctor” command.

---

# 7) Migration plan (outline)

1. **0001_init.sql** → `versions`, `artifacts`, `latest_pointer`
2. **0002_files.sql** → `extracted_files`
3. **0003_validations.sql** → `validations`
4. **0004_events.sql** → `events` (optional)

Each migration inserts a row into `schema_version`. At boot, apply all **forward** migrations within a TX.

---

# 8) Acceptance checklist (for this data layer)

* [ ] Filesystem root + version trees conform to the layout above; audit JSON emitted per extraction.
* [ ] DuckDB file created; migrations 0001–0004 applied; `schema_version` shows latest.
* [ ] Unique/PK constraints active; `CREATE INDEX` statements executed.
* [ ] Views created and return sensible results on sample data.
* [ ] Two-phase boundary choreography keeps DB and FS consistent.
* [ ] Doctor/prune use `staging_fs_listing` + `v_fs_orphans` for safe cleanup.
* [ ] Deterministic audit JSON validates against the provided schema.

---

If you want, I can turn this into **executable DDL migration files** and a **markdown data dictionary** per table (columns, meaning, allowed values, producers/consumers) tailored to your repo paths.
