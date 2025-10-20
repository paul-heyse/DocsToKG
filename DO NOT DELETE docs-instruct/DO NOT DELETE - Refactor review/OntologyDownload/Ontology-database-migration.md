Perfect—here are the **migration files inline** (ready to paste into your repo), followed by a **concise data dictionary**. They’re ordered; apply 0001 → 0006. I kept these identical in spirit to the downloadable versions, just trimmed comments to keep this message compact.

---

## `migrations/0001_init.sql`

```sql
-- 0001_init.sql — core entities (UTC timestamps; POSIX paths)
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS schema_version (
  version     TEXT PRIMARY KEY,
  applied_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
  version_id  TEXT PRIMARY KEY,           -- e.g., 2025-10-20T01:23:45Z
  service     TEXT NOT NULL,              -- OLS, BioPortal, etc.
  created_at  TIMESTAMP NOT NULL,
  plan_hash   TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id   TEXT PRIMARY KEY,         -- sha256 (hex, lowercase)
  version_id    TEXT NOT NULL,
  service       TEXT NOT NULL,
  source_url    TEXT NOT NULL,            -- normalized URL
  etag          TEXT,
  last_modified TIMESTAMP,
  content_type  TEXT,
  size_bytes    BIGINT NOT NULL,
  fs_relpath    TEXT NOT NULL,            -- path under ontologies/ root
  status        TEXT NOT NULL CHECK (status IN ('fresh','cached','failed')),
  UNIQUE (version_id, fs_relpath)
  -- FOREIGN KEY (version_id) REFERENCES versions(version_id)
);

CREATE TABLE IF NOT EXISTS latest_pointer (
  slot        TEXT PRIMARY KEY DEFAULT 'default',
  version_id  TEXT NOT NULL,
  updated_at  TIMESTAMP NOT NULL,
  by          TEXT
  -- FOREIGN KEY (version_id) REFERENCES versions(version_id)
);

CREATE INDEX IF NOT EXISTS idx_versions_service_created
  ON versions(service, created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_version  ON artifacts(version_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_service  ON artifacts(service);
CREATE INDEX IF NOT EXISTS idx_artifacts_source_url ON artifacts(source_url);

INSERT OR IGNORE INTO schema_version VALUES ('0001_init', now());
COMMIT;
```

---

## `migrations/0002_files.sql`

```sql
-- 0002_files.sql — extracted file catalog
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS extracted_files (
  file_id            TEXT PRIMARY KEY,    -- sha256 of file bytes
  artifact_id        TEXT NOT NULL,
  version_id         TEXT NOT NULL,
  relpath_in_version TEXT NOT NULL,       -- <service>/<version>/...
  format             TEXT NOT NULL,       -- rdf|ttl|owl|obo|jsonld|xml|nt|nq|trig|trix|other
  size_bytes         BIGINT NOT NULL,
  mtime              TIMESTAMP,
  cas_relpath        TEXT,
  UNIQUE (version_id, relpath_in_version)
  -- FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
  -- FOREIGN KEY (version_id)  REFERENCES versions(version_id)
);

CREATE INDEX IF NOT EXISTS idx_files_version  ON extracted_files(version_id);
CREATE INDEX IF NOT EXISTS idx_files_format   ON extracted_files(format);
CREATE INDEX IF NOT EXISTS idx_files_artifact ON extracted_files(artifact_id);

INSERT OR IGNORE INTO schema_version VALUES ('0002_files', now());
COMMIT;
```

---

## `migrations/0003_validations.sql`

```sql
-- 0003_validations.sql — validator outcomes
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS validations (
  validation_id  TEXT PRIMARY KEY,        -- ULID or sha256(file_id|validator|run_at)
  file_id        TEXT NOT NULL,
  validator      TEXT NOT NULL,           -- pySHACL|ROBOT|Arelle|Custom:...
  passed         BOOLEAN NOT NULL,
  details_json   JSON,
  run_at         TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_validations_file
  ON validations(file_id);
CREATE INDEX IF NOT EXISTS idx_validations_validator_time
  ON validations(validator, run_at);

INSERT OR IGNORE INTO schema_version VALUES ('0003_validations', now());
COMMIT;
```

---

## `migrations/0004_events.sql` (optional)

```sql
-- 0004_events.sql — structured observability events (optional)
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS events (
  run_id   TEXT NOT NULL,                 -- UUID per extraction run
  ts       TIMESTAMP NOT NULL,            -- UTC
  type     TEXT NOT NULL,                 -- extract.start|pre_scan.done|extract.done|extract.error|audit.emitted|storage.*
  level    TEXT NOT NULL,                 -- INFO|WARN|ERROR
  payload  JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_time ON events(run_id, ts);

INSERT OR IGNORE INTO schema_version VALUES ('0004_events', now());
COMMIT;
```

---

## `migrations/0005_staging_prune.sql`

```sql
-- 0005_staging_prune.sql — ephemeral FS listing + orphans view
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS staging_fs_listing (
  scope       TEXT NOT NULL,              -- 'cas' | 'version'
  relpath     TEXT NOT NULL,              -- relative to ontologies/ root
  size_bytes  BIGINT NOT NULL,
  mtime       TIMESTAMP
);

CREATE OR REPLACE VIEW v_fs_orphans AS
SELECT s.*
FROM staging_fs_listing s
LEFT JOIN (
  SELECT fs_relpath AS relpath FROM artifacts
  UNION ALL
  SELECT service || '/' || version_id || '/' || relpath_in_version FROM extracted_files
) cat ON cat.relpath = s.relpath
WHERE cat.relpath IS NULL;

INSERT OR IGNORE INTO schema_version VALUES ('0005_staging_prune', now());
COMMIT;
```

---

## `migrations/0006_views.sql`

```sql
-- 0006_views.sql — convenience analytic views
BEGIN TRANSACTION;

WITH _noop AS (SELECT 1) SELECT 1;

-- Per-version rollup
CREATE OR REPLACE VIEW v_version_stats AS
WITH val AS (
  SELECT file_id, bool_or(passed) AS passed_any
  FROM validations
  GROUP BY file_id
)
SELECT
  v.version_id,
  v.service,
  v.created_at,
  COUNT(f.file_id)                                       AS files,
  COALESCE(SUM(f.size_bytes), 0)                         AS bytes,
  COALESCE(SUM(CASE WHEN val.passed_any THEN 1 ELSE 0 END), 0) AS validations_passed,
  COALESCE(SUM(CASE WHEN val.passed_any THEN 0 ELSE 1 END), 0) AS validations_failed
FROM versions v
LEFT JOIN extracted_files f ON f.version_id = v.version_id
LEFT JOIN val ON val.file_id = f.file_id
GROUP BY 1,2,3;

-- Files for current latest
CREATE OR REPLACE VIEW v_latest_files AS
SELECT f.*
FROM latest_pointer lp
JOIN extracted_files f ON f.version_id = lp.version_id;

-- Artifact status overview
CREATE OR REPLACE VIEW v_artifacts_status AS
SELECT
  service, version_id,
  COUNT(*) AS archives,
  COALESCE(SUM(size_bytes), 0) AS archive_bytes,
  SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
FROM artifacts
GROUP BY 1,2;

-- Failing validations with paths
CREATE OR REPLACE VIEW v_validation_failures AS
SELECT
  f.version_id, f.relpath_in_version, v.validator, v.run_at, v.details_json
FROM extracted_files f
JOIN validations v ON v.file_id = f.file_id
WHERE v.passed = FALSE;

-- Latest formats distribution
CREATE OR REPLACE VIEW v_latest_formats AS
SELECT format, COUNT(*) AS files, COALESCE(SUM(size_bytes), 0) AS bytes
FROM v_latest_files
GROUP BY 1
ORDER BY bytes DESC;

INSERT OR IGNORE INTO schema_version VALUES ('0006_views', now());
COMMIT;
```

---

# Concise Data Dictionary (what each table stores)

**`versions`**

* One row per logical release/version.
* Columns: `version_id (PK)`, `service`, `created_at`, `plan_hash`.

**`artifacts`**

* One row per downloaded archive (zip/tar.*).
* Columns: `artifact_id (PK, sha256)`, `version_id`, `service`, `source_url`, `etag`, `last_modified`, `content_type`, `size_bytes`, `fs_relpath`, `status`.
* Unique: `(version_id, fs_relpath)`.

**`extracted_files`**

* One row per **extracted regular file**.
* Columns: `file_id (PK, sha256)`, `artifact_id`, `version_id`, `relpath_in_version`, `format`, `size_bytes`, `mtime`, `cas_relpath`.
* Unique: `(version_id, relpath_in_version)`.

**`validations`**

* Outcomes of SHACL/ROBOT/Arelle/etc.
* Columns: `validation_id (PK)`, `file_id`, `validator`, `passed`, `details_json`, `run_at`.

**`latest_pointer`**

* Pointer to “current” version (by slot; default single row).
* Columns: `slot (PK)`, `version_id`, `updated_at`, `by`.

**`events`** (optional)

* Structured observability events (`extract.*`, `storage.*`).
* Columns: `run_id`, `ts`, `type`, `level`, `payload (JSON)`.

**`staging_fs_listing`** (ephemeral)

* Filesystem scan results loaded before prune/doctor.
* Columns: `scope`, `relpath`, `size_bytes`, `mtime`.
* View `v_fs_orphans` shows on-disk files not referenced by the catalog.

**Views**

* `v_version_stats`: files/bytes/validation rollups per version.
* `v_latest_files`: all files for current latest.
* `v_artifacts_status`: #archives, bytes, failed count per (service, version).
* `v_validation_failures`: failing validations with paths.
* `v_latest_formats`: format distribution in latest.

---

## Apply order (DuckDB CLI)

```bash
duckdb /path/to/ontofetch.duckdb
.read migrations/0001_init.sql
.read migrations/0002_files.sql
.read migrations/0003_validations.sql
.read migrations/0004_events.sql
.read migrations/0005_staging_prune.sql
.read migrations/0006_views.sql
```

If you want me to tailor any column names, add more indexes, or produce **parameterized delta views** (A vs B version comparison), say the word and I’ll drop those in as well.
