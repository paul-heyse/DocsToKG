amazing—here’s a single, drop-in **migration file** that defines a **comprehensive, parameterized delta suite** for DuckDB. It gives you file-level, rename-aware, summary, per-format, and validation deltas. Everything is **parameterized** by two `version_id`s (and is usable like a table function): `SELECT * FROM version_delta_files('A','B');`

You can paste this as `migrations/0007_delta_macros.sql` and apply it after 0006.

---

# `migrations/0007_delta_macros.sql`

```sql
-- 0007_delta_macros.sql — Parameterized delta suite for version-to-version comparisons
-- DuckDB requirement: table macros (DuckDB ≥ 0.8)

BEGIN TRANSACTION;

--------------------------------------------------------------------
-- Helper notes (semantics)
--  - “path” means extracted_files.relpath_in_version (normalized)
--  - Identity of bytes: extracted_files.file_id (sha256)
--  - Formats: extracted_files.format (rdf|ttl|owl|obo|…)
--  - Size: extracted_files.size_bytes (bytes)
--  - A = older version; B = newer version
-- Change classes we produce:
--   ADDED      : path present only in B
--   REMOVED    : path present only in A
--   MODIFIED   : path present in both, but bytes (file_id) OR format OR size changed
--   UNCHANGED  : path present in both with identical (file_id, format, size)
--   RENAMED    : content moved (same file_id exists in A and B) with a different path
--------------------------------------------------------------------

--------------------------------------------------------------------
-- Macro: version_delta_files(a, b)
-- Returns path-based delta (includes UNCHANGED; caller can filter)
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_delta_files(a, b) AS TABLE (
WITH
A AS (
  SELECT relpath_in_version AS path,
         file_id, size_bytes, format
  FROM extracted_files WHERE version_id = a
),
B AS (
  SELECT relpath_in_version AS path,
         file_id, size_bytes, format
  FROM extracted_files WHERE version_id = b
)
SELECT
  COALESCE(A.path, B.path)                    AS path,
  CASE
    WHEN A.path IS NULL THEN 'ADDED'
    WHEN B.path IS NULL THEN 'REMOVED'
    WHEN (A.file_id IS DISTINCT FROM B.file_id)
      OR (A.format  IS DISTINCT FROM B.format)
      OR (A.size_bytes IS DISTINCT FROM B.size_bytes)
      THEN 'MODIFIED'
    ELSE 'UNCHANGED'
  END                                         AS change,
  A.file_id                                   AS file_id_a,
  B.file_id                                   AS file_id_b,
  A.size_bytes                                AS size_a,
  B.size_bytes                                AS size_b,
  (COALESCE(B.size_bytes,0)-COALESCE(A.size_bytes,0)) AS bytes_delta,
  A.format                                    AS format_a,
  B.format                                    AS format_b
FROM A
FULL OUTER JOIN B USING (path)
);

--------------------------------------------------------------------
-- Macro: version_delta_renames(a, b)
-- Content-preserving renames (same file_id, different path)
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_delta_renames(a, b) AS TABLE (
WITH
A AS (
  SELECT relpath_in_version AS path, file_id, size_bytes, format
  FROM extracted_files WHERE version_id = a
),
B AS (
  SELECT relpath_in_version AS path, file_id, size_bytes, format
  FROM extracted_files WHERE version_id = b
)
SELECT
  A.path AS old_path,
  B.path AS new_path,
  A.file_id,
  A.size_bytes AS size_bytes,
  A.format     AS format
FROM A
JOIN B USING (file_id)
WHERE A.path <> B.path
);

--------------------------------------------------------------------
-- Macro: version_delta_files_rename_aware(a, b)
-- Path delta with 'RENAMED' category; added/removed exclude renames
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_delta_files_rename_aware(a, b) AS TABLE (
WITH
A AS (SELECT relpath_in_version AS path, file_id, size_bytes, format FROM extracted_files WHERE version_id = a),
B AS (SELECT relpath_in_version AS path, file_id, size_bytes, format FROM extracted_files WHERE version_id = b),

-- content-preserving renames
R AS (
  SELECT A.path AS old_path, B.path AS new_path, A.file_id, B.file_id AS file_id_b
  FROM A JOIN B USING (file_id)
  WHERE A.path <> B.path
),

-- core path match
P AS (
  SELECT
    COALESCE(A.path, B.path) AS path,
    A.file_id  AS file_id_a, B.file_id  AS file_id_b,
    A.size_bytes AS size_a,  B.size_bytes AS size_b,
    A.format    AS format_a, B.format    AS format_b,
    CASE
      WHEN A.path IS NULL THEN 'ADDED'
      WHEN B.path IS NULL THEN 'REMOVED'
      WHEN (A.file_id IS DISTINCT FROM B.file_id)
        OR (A.format  IS DISTINCT FROM B.format)
        OR (A.size_bytes IS DISTINCT FROM B.size_bytes) THEN 'MODIFIED'
      ELSE 'UNCHANGED'
    END AS change
  FROM A
  FULL OUTER JOIN B USING (path)
),

-- mark rows that correspond to renames to exclude them from ADDED/REMOVED
RENAMED_TARGETS AS (
  SELECT new_path AS path FROM R
  UNION
  SELECT old_path AS path FROM R
)

SELECT
  path,
  CASE
    WHEN change IN ('ADDED','REMOVED') AND path IN (SELECT path FROM RENAMED_TARGETS)
      THEN 'RENAMED'
    ELSE change
  END AS change,
  file_id_a, file_id_b, size_a, size_b,
  (COALESCE(size_b,0)-COALESCE(size_a,0)) AS bytes_delta,
  format_a, format_b
FROM P
);

--------------------------------------------------------------------
-- Macro: version_delta_summary(a, b)
-- High-level metrics (rename-aware). Unchanged omitted from counts by default.
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_delta_summary(a, b) AS TABLE (
WITH
A_BYTES AS (SELECT COALESCE(SUM(size_bytes),0) AS bytes_a FROM extracted_files WHERE version_id = a),
B_BYTES AS (SELECT COALESCE(SUM(size_bytes),0) AS bytes_b FROM extracted_files WHERE version_id = b),
D AS (SELECT * FROM version_delta_files_rename_aware(a, b))
SELECT
  (SELECT bytes_a FROM A_BYTES)                                  AS bytes_total_a,
  (SELECT bytes_b FROM B_BYTES)                                  AS bytes_total_b,
  (SELECT bytes_b FROM B_BYTES) - (SELECT bytes_a FROM A_BYTES)  AS bytes_total_delta,

  SUM(CASE WHEN change='ADDED'    THEN 1 ELSE 0 END)             AS files_added,
  SUM(CASE WHEN change='ADDED'    THEN COALESCE(size_b,0) ELSE 0 END) AS bytes_added,

  SUM(CASE WHEN change='REMOVED'  THEN 1 ELSE 0 END)             AS files_removed,
  SUM(CASE WHEN change='REMOVED'  THEN COALESCE(size_a,0) ELSE 0 END) AS bytes_removed,

  SUM(CASE WHEN change='MODIFIED' THEN 1 ELSE 0 END)             AS files_modified,
  SUM(CASE WHEN change='MODIFIED' THEN ABS(COALESCE(size_b,0)-COALESCE(size_a,0)) ELSE 0 END) AS bytes_delta_modified,

  SUM(CASE WHEN change='RENAMED'  THEN 1 ELSE 0 END)             AS files_renamed,
  SUM(CASE WHEN change='RENAMED'  THEN COALESCE(size_b,COALESCE(size_a,0)) ELSE 0 END)        AS bytes_renamed,

  SUM(CASE WHEN change='UNCHANGED' THEN 1 ELSE 0 END)            AS files_unchanged
FROM D
);

--------------------------------------------------------------------
-- Macro: version_delta_formats(a, b)
-- Format-level aggregation (rename-aware). Breaks out added/removed/modified/renamed per format.
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_delta_formats(a, b) AS TABLE (
WITH
D AS (SELECT * FROM version_delta_files_rename_aware(a, b)),
-- choose the most relevant format per change
FF AS (
  SELECT
    CASE
      WHEN change IN ('ADDED','RENAMED') THEN format_b
      WHEN change = 'REMOVED'            THEN format_a
      WHEN change = 'MODIFIED'           THEN COALESCE(format_b, format_a)
      ELSE COALESCE(format_b, format_a)
    END AS fmt,
    change,
    size_a, size_b
  FROM D
)
SELECT
  fmt AS format,
  SUM(CASE WHEN change='ADDED'    THEN 1 ELSE 0 END) AS files_added,
  SUM(CASE WHEN change='ADDED'    THEN COALESCE(size_b,0) ELSE 0 END) AS bytes_added,
  SUM(CASE WHEN change='REMOVED'  THEN 1 ELSE 0 END) AS files_removed,
  SUM(CASE WHEN change='REMOVED'  THEN COALESCE(size_a,0) ELSE 0 END) AS bytes_removed,
  SUM(CASE WHEN change='MODIFIED' THEN 1 ELSE 0 END) AS files_modified,
  SUM(CASE WHEN change='MODIFIED' THEN ABS(COALESCE(size_b,0)-COALESCE(size_a,0)) ELSE 0 END) AS bytes_delta_modified,
  SUM(CASE WHEN change='RENAMED'  THEN 1 ELSE 0 END) AS files_renamed
FROM FF
GROUP BY 1
ORDER BY bytes_added + bytes_removed + bytes_delta_modified DESC NULLS LAST
);

--------------------------------------------------------------------
-- Macro: version_validation_delta(a, b)
-- Path-based validation transitions by validator (FIXED / REGRESSED / NEW / REMOVED)
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_validation_delta(a, b) AS TABLE (
WITH
A AS (SELECT relpath_in_version AS path, file_id FROM extracted_files WHERE version_id = a),
B AS (SELECT relpath_in_version AS path, file_id FROM extracted_files WHERE version_id = b),

VAL_A AS (
  SELECT v.file_id, v.validator, bool_or(v.passed) AS passed_any
  FROM validations v
  GROUP BY 1,2
),
VAL_B AS (
  SELECT v.file_id, v.validator, bool_or(v.passed) AS passed_any
  FROM validations v
  GROUP BY 1,2
),

JOINED AS (
  SELECT
    COALESCE(A.path, B.path)                     AS path,
    COALESCE(VAL_A.validator, VAL_B.validator)   AS validator,
    VAL_A.passed_any                              AS passed_a,
    VAL_B.passed_any                              AS passed_b
  FROM A
  FULL OUTER JOIN B USING (path)
  LEFT JOIN VAL_A ON VAL_A.file_id = A.file_id
  LEFT JOIN VAL_B ON VAL_B.file_id = B.file_id
)
SELECT
  path,
  validator,
  CASE
    WHEN passed_a IS NULL AND passed_b IS NOT NULL THEN 'NEW'
    WHEN passed_a IS NOT NULL AND passed_b IS NULL THEN 'REMOVED'
    WHEN passed_a = TRUE  AND passed_b = FALSE     THEN 'REGRESSED'
    WHEN passed_a = FALSE AND passed_b = TRUE      THEN 'FIXED'
    WHEN passed_a = passed_b AND passed_a IS NOT NULL
         THEN 'UNCHANGED'
    ELSE 'UNKNOWN'
  END AS status_change,
  passed_a, passed_b
FROM JOINED
WHERE status_change <> 'UNCHANGED'
);

--------------------------------------------------------------------
-- Macro: version_delta_renames_summary(a, b)
-- Just the rename count and bytes (useful in dashboards)
--------------------------------------------------------------------
CREATE OR REPLACE MACRO version_delta_renames_summary(a, b) AS TABLE (
SELECT
  COUNT(*)                                AS files_renamed,
  COALESCE(SUM(size_bytes),0)             AS bytes_renamed
FROM version_delta_renames(a, b)
);

-- Record migration
INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES ('0007_delta_macros', now());

COMMIT;
```

---

## How to use (examples)

**1) File-level delta (path-based)**

```sql
SELECT * FROM version_delta_files('2025-10-15', '2025-10-20')
WHERE change <> 'UNCHANGED'
ORDER BY change, path
LIMIT 200;
```

**2) Rename-aware delta**

```sql
SELECT * FROM version_delta_files_rename_aware('2025-10-15', '2025-10-20')
WHERE change IN ('ADDED','REMOVED','MODIFIED','RENAMED')
ORDER BY change, path;
```

**3) High-level summary**

```sql
SELECT * FROM version_delta_summary('2025-10-15', '2025-10-20');
```

**4) Format breakdown**

```sql
SELECT * FROM version_delta_formats('2025-10-15', '2025-10-20');
```

**5) Pure rename report**

```sql
SELECT * FROM version_delta_renames('2025-10-15', '2025-10-20')
ORDER BY size_bytes DESC
LIMIT 50;
```

**6) Validation transitions**

```sql
-- All FIXED/REGRESSED/NEW/REMOVED by validator
SELECT * FROM version_validation_delta('2025-10-15', '2025-10-20')
ORDER BY status_change, validator, path;
```

---

## Design notes & best practices

* **Rename awareness:** we compute a content-based mapping (by `file_id`) and exclude those paths from ADDED/REMOVED to avoid double counting churn.
* **MODIFIED definition:** any of {bytes changed, format changed, file_id changed}. If you want “content-only” changes, drop the format/size clauses.
* **Bytes deltas:** summaries report net bytes (`bytes_total_b - bytes_total_a`) and *modified delta* (`ABS(size_b - size_a)`) to separately capture growth and churn.
* **Validation deltas:** path-based so you can track regressions by *where* the ontology lives. If you also want *content-based* transitions (follow `file_id` across moves), I can add a `version_validation_delta_by_fileid(a,b)` macro.
* **Performance:** All macros use compact scans and joins on indexed columns. DuckDB will handle them well up to millions of rows; consider adding additional indexes on `extracted_files(version_id, relpath_in_version)` if needed (DuckDB supports compound indexes).

If you want this split into **multiple smaller migration files** (e.g., `0007_files_delta.sql`, `0008_validation_delta.sql`) or prefer different column names, I can tailor it instantly.
