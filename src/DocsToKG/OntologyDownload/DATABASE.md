# DuckDB Catalog Integration for OntologyDownload

## Overview

The `database` module provides a **transactional, single-node catalog** for tracking ontology versions, artifacts, extracted files, validations, and provenance metadata. Binary payloads remain on the filesystem for streaming and atomicity; DuckDB stores all metadata and enables fast queries, safe pruning, clean lineage tracking, and deterministic replays.

**Key principles:**

- **Filesystem is the source of truth for blobs**: Downloads, archives, and extracted files live on disk under `<PYSTOW_HOME>/ontology-fetcher/ontologies/<service>/<version>/`.
- **DuckDB catalogs metadata**: Version pointers, artifact hashes, file listings, validation outcomes, events, and lineage.
- **One writer at a time**: File-based locks (`<db>.lock`) serialize writers; readers open read-only without locks.
- **Two-phase choreography**: Writes to filesystem commit first, then DB transactions record the metadata. If DB commit fails, the entire transaction rolls back.
- **Idempotence via content hashing**: All records keyed by `sha256(content)` so re-runs with identical blobs produce no duplicates.

---

## Architecture

### Data Layout (Filesystem)

```
~/.data/ontology-fetcher/ontologies/
├── LATEST.json                           # latest version marker (human-readable)
├── .cas/sha256/aa/bb/<digest>            # optional CAS mirrors
├── OLS/
│   └── 2025-10-20T01:23:45Z/
│       ├── src/archives/obo-basic.zip    # downloaded archives
│       ├── data/                         # extracted files
│       │   ├── ontology.ttl
│       │   └── ontology.rdf
│       └── .extract.audit.json           # per-extraction provenance
├── BioPortal/
│   └── 2025-10-15/...
```

### Database Layout (DuckDB)

```
~/.data/ontology-fetcher/.catalog/ontofetch.duckdb
```

**Tables:**

- `schema_version`: Migration tracking.
- `versions`: Logical releases of fetched corpora.
- `artifacts`: Downloaded archives and their metadata.
- `extracted_files`: Regular files extracted from archives.
- `validations`: Validator outcomes (RDF/SHACL/ROBOT/Arelle).
- `latest_pointer`: Pointer to the current "latest" version.
- `events`: Append-only observability events (optional; can be Parquet).
- `staging_fs_listing`: Ephemeral staging table for orphan detection.

---

## Usage Patterns

### 1. Basic Bootstrap

```python
from DocsToKG.OntologyDownload.database import Database, DatabaseConfiguration

config = DatabaseConfiguration(
    db_path=Path("/path/to/ontofetch.duckdb"),
    readonly=False,
    enable_locks=True,
    threads=8,
)

db = Database(config)
db.bootstrap()  # Creates schema, applies migrations
try:
    # ... queries ...
finally:
    db.close()
```

### 2. Context Manager

```python
with Database(config) as db:
    # Auto-bootstrap and close
    db.upsert_version("v1.0", "OLS", plan_hash="hash123")
    latest = db.get_latest_version()
```

### 3. Global Singleton

```python
from DocsToKG.OntologyDownload.database import get_database, close_database

db = get_database()  # Thread-safe; bootstraps on first call
versions = db.list_versions(service="OLS", limit=10)
close_database()  # Clean shutdown
```

### 4. Transactional Writes (Two-Phase Boundary)

```python
# 1. Write to filesystem (e.g., extract archive, write files)
archive_path = extract_archive(...)

# 2. Begin transactional DB write
with db.transaction():
    # Upsert version record
    db.upsert_version("v1.0", "OLS", plan_hash="hash123")

    # Record the downloaded archive
    db.upsert_artifact(
        artifact_id="sha256_abc123...",
        version_id="v1.0",
        service="OLS",
        source_url="https://example.com/obo-basic.zip",
        size_bytes=5242880,
        fs_relpath="OLS/v1.0/src/archives/obo-basic.zip",
        status="fresh",
        etag="etag123",
    )

    # Batch insert extracted files
    from DocsToKG.OntologyDownload.database import FileRow
    files = [
        FileRow(
            file_id="sha256_file1...",
            artifact_id="sha256_abc123...",
            version_id="v1.0",
            relpath_in_version="data/ontology.ttl",
            format="ttl",
            size_bytes=1048576,
        ),
        FileRow(
            file_id="sha256_file2...",
            artifact_id="sha256_abc123...",
            version_id="v1.0",
            relpath_in_version="data/ontology.rdf",
            format="rdf",
            size_bytes=1048576,
        ),
    ]
    db.insert_extracted_files(files)

    # Set latest version
    db.set_latest_version("v1.0", by=os.getenv("USER"))

# Transaction commits here; if any step fails, entire TX rolls back
```

---

## Query Facades (Public API)

All queries are encapsulated behind facades; **no SQL leaks to callers**.

### Versions

```python
# Upsert (create or update)
db.upsert_version(version_id="v1.0", service="OLS", plan_hash="hash123")

# Get latest
latest: Optional[VersionRow] = db.get_latest_version()

# Set latest pointer
db.set_latest_version("v1.0", by="user@example.com")

# List versions
versions: List[VersionRow] = db.list_versions(service="OLS", limit=50)
for v in versions:
    print(f"{v.version_id}: {v.service} ({v.created_at})")
```

### Artifacts

```python
# Upsert archive metadata
db.upsert_artifact(
    artifact_id="sha256_abc...",
    version_id="v1.0",
    service="OLS",
    source_url="https://example.com/obo-basic.zip",
    size_bytes=5242880,
    fs_relpath="OLS/v1.0/src/archives/obo-basic.zip",
    status="fresh",
    etag="etag123",
    last_modified=datetime.now(timezone.utc),
    content_type="application/zip",
)

# List artifacts for a version
artifacts = db.list_artifacts("v1.0")

# Filter by status
fresh = db.list_artifacts("v1.0", status="fresh")
cached = db.list_artifacts("v1.0", status="cached")
failed = db.list_artifacts("v1.0", status="failed")
```

### Extracted Files

```python
from DocsToKG.OntologyDownload.database import FileRow

# Batch insert
files = [
    FileRow("file_sha256_1", "artifact_sha256", "v1.0", "data/file1.ttl", "ttl", 1024),
    FileRow("file_sha256_2", "artifact_sha256", "v1.0", "data/file2.rdf", "rdf", 2048),
]
db.insert_extracted_files(files)

# List all extracted files for a version
all_files = db.list_extracted_files("v1.0")

# Filter by format
ttl_files = db.list_extracted_files("v1.0", format_filter="ttl")
rdf_files = db.list_extracted_files("v1.0", format_filter="rdf")
```

### Validations

```python
from DocsToKG.OntologyDownload.database import ValidationRow

# Batch insert validation results
validations = [
    ValidationRow(
        validation_id="vid1",
        file_id="file_sha256_1",
        validator="rdflib",
        passed=True,
        run_at=datetime.now(timezone.utc),
        details_json={"message": "Valid RDF"},
    ),
    ValidationRow(
        validation_id="vid2",
        file_id="file_sha256_2",
        validator="rdflib",
        passed=False,
        run_at=datetime.now(timezone.utc),
        details_json={"error": "Invalid namespace"},
    ),
]
db.insert_validations(validations)

# Get all failures for a version
failures = db.get_validation_failures("v1.0")
for v in failures:
    print(f"Failed: {v.file_id} ({v.validator})")
    if v.details_json:
        print(f"  Details: {v.details_json}")
```

### Statistics

```python
from DocsToKG.OntologyDownload.database import VersionStats

stats: Optional[VersionStats] = db.get_version_stats("v1.0")
if stats:
    print(f"Version: {stats.version_id}")
    print(f"  Files: {stats.files}")
    print(f"  Total size: {stats.bytes} bytes")
    print(f"  Validations passed: {stats.validations_passed}")
    print(f"  Validations failed: {stats.validations_failed}")
```

---

## Schema Reference

### `versions`

| Column | Type | Description |
| --- | --- | --- |
| `version_id` | TEXT PRIMARY KEY | Canonical version label (e.g., `2025-10-20T01:23:45Z`) |
| `service` | TEXT NOT NULL | OLS, BioPortal, etc. |
| `created_at` | TIMESTAMP NOT NULL | When this version was first materialized |
| `plan_hash` | TEXT | Hash of the plan manifest used |

### `artifacts`

| Column | Type | Description |
| --- | --- | --- |
| `artifact_id` | TEXT PRIMARY KEY | sha256 of archive bytes |
| `version_id` | TEXT NOT NULL | Reference to `versions.version_id` |
| `service` | TEXT NOT NULL | OLS, BioPortal, etc. |
| `source_url` | TEXT NOT NULL | Normalized URL |
| `etag` | TEXT | HTTP ETag if present |
| `last_modified` | TIMESTAMP | HTTP Last-Modified if present |
| `content_type` | TEXT | Server Content-Type |
| `size_bytes` | BIGINT NOT NULL | Archive size on disk |
| `fs_relpath` | TEXT NOT NULL | Path under ontologies/ root (POSIX) |
| `status` | TEXT NOT NULL CHECK(...) | 'fresh', 'cached', or 'failed' |
| UNIQUE | (version_id, fs_relpath) | Ensures only one archive per location per version |

### `extracted_files`

| Column | Type | Description |
| --- | --- | --- |
| `file_id` | TEXT PRIMARY KEY | sha256 of file bytes |
| `artifact_id` | TEXT NOT NULL | Reference to `artifacts.artifact_id` |
| `version_id` | TEXT NOT NULL | Reference to `versions.version_id` |
| `relpath_in_version` | TEXT NOT NULL | Canonical path under `<service>/<version>/` |
| `format` | TEXT NOT NULL | rdf, ttl, owl, obo, other |
| `size_bytes` | BIGINT NOT NULL | File size |
| `mtime` | TIMESTAMP | File modification time (policy-dependent) |
| `cas_relpath` | TEXT | Optional: where CAS copy lives |
| UNIQUE | (version_id, relpath_in_version) | Unique file per version tree |

### `validations`

| Column | Type | Description |
| --- | --- | --- |
| `validation_id` | TEXT PRIMARY KEY | ULID or sha256(file_id + validator + run_at) |
| `file_id` | TEXT NOT NULL | Reference to `extracted_files.file_id` |
| `validator` | TEXT NOT NULL | pySHACL, ROBOT, Arelle, Custom:<name>, etc. |
| `passed` | BOOLEAN NOT NULL | True if validation passed |
| `details_json` | JSON | Compact message or summary |
| `run_at` | TIMESTAMP NOT NULL | When validation was run |

### `latest_pointer`

| Column | Type | Description |
| --- | --- | --- |
| `slot` | TEXT PRIMARY KEY DEFAULT 'default' | Slot name (typically 'default') |
| `version_id` | TEXT NOT NULL | Reference to `versions.version_id` |
| `updated_at` | TIMESTAMP NOT NULL | When pointer was updated |
| `by` | TEXT | Hostname/user/process that updated it |

---

## Configuration

### DatabaseConfiguration (Pydantic Model)

```python
from DocsToKG.OntologyDownload.settings import DatabaseConfiguration

config = DatabaseConfiguration(
    db_path=None,                  # Defaults to ~/.data/ontology-fetcher/.catalog/ontofetch.duckdb
    readonly=False,                # Open in read-only mode
    enable_locks=True,             # File-based writer lock
    threads=None,                  # CPU count by default
    memory_limit=None,             # Auto by default
    enable_object_cache=True,      # Cache Parquet metadata
    parquet_events=False,          # Store events as Parquet (optional)
)
```

### Environment Variables

```bash
export ONTOFETCH_DB_READONLY=false
export ONTOFETCH_DB_THREADS=8
export ONTOFETCH_DB_MEMORY_LIMIT="16GB"
export ONTOFETCH_DB_ENABLE_OBJECT_CACHE=true
```

---

## Concurrency Model

- **Writers**: One at a time, serialized by file-based lock (`<db>.lock`).
- **Readers**: Many concurrent readers; no lock required.
- **Lock mode**: Exclusive (`fcntl.LOCK_EX`) acquired by writers before any modification.
- **Lock scope**: Typically held for the duration of extraction/validation pipeline stages.

**Best practices:**

- Keep transactions short (bounded by a pipeline stage).
- Use `readonly=True` when opening connections for queries only.
- Readers should never be blocked by locks.

---

## Data Invariants & Integrity Rules

1. `artifacts.version_id` **must exist** in `versions` (enforce in code + doctor checks).
2. `extracted_files.version_id` and `artifact_id` **must exist** (enforce in code).
3. `(version_id, relpath_in_version)` **uniquely identifies** a file in the version tree.
4. `file_id` (sha256) **uniquely identifies bytes** across the entire repo (dedup/CAS possible).
5. `latest_pointer.version_id` **must exist** in `versions`.
6. **No write to DB** unless the corresponding **FS write** has succeeded (two-phase choreography).
7. **Idempotence**: Re-running with identical blobs produces **no new rows**.

---

## Migration Strategy

The database uses a **forward-only migration system**. Each migration is a numbered SQL file:

```
0001_init.sql        → schema_version, versions, artifacts, latest_pointer
0002_files.sql       → extracted_files
0003_validations.sql → validations
0004_events.sql      → events (optional)
```

On bootstrap:

1. Check current `schema_version`.
2. Apply all **forward** migrations in order within a single transaction.
3. Insert new `schema_version` row for each applied migration.

---

## Performance Considerations

- **Indexes**: All important join columns and filters are indexed (`version_id`, `format`, `service`, `source_url`).
- **Batching**: Use `insert_extracted_files()` and `insert_validations()` to batch inserts.
- **Memory**: Set `memory_limit` to match available RAM; use `threads` to control parallelism.
- **Object cache**: Enable `enable_object_cache` if repeatedly scanning large remote files.
- **Row groups**: (Future) Tune Parquet row-group sizing when writing event logs.

---

## Testing

See `tests/ontology_download/test_database.py` for comprehensive test coverage:

```bash
./.venv/bin/pytest tests/ontology_download/test_database.py -v
```

**Test categories:**

- Bootstrap and schema creation
- Version CRUD operations
- Artifact management
- Extracted file insertion
- Validation recording
- Statistics computation
- Transaction semantics
- Idempotence properties
- Context manager behavior

---

## Future Enhancements

1. **Views**: Pre-built SQL views for common queries (`v_version_stats`, `v_validation_failures`).
2. **Events table**: Full event logging with optional Parquet sink.
3. **Orphan detection**: Helper to find and prune orphaned blobs (FS entries not in catalog).
4. **Doctor command**: Reconciliation and repair of DB↔FS consistency.
5. **Remote DuckDB**: Read-only access to remote `.duckdb` files over HTTPS/S3 (DuckDB v1.1+).
6. **Arrow interop**: Direct export of query results as PyArrow Tables for downstream analytics.

---

## See Also

- **`src/DocsToKG/OntologyDownload/database.py`**: Implementation.
- **`DO NOT DELETE docs-instruct/.../Ontology-database-layout.md`**: Detailed filesystem/schema design.
- **`DO NOT DELETE docs-instruct/.../Ontology-database-scope.md`**: DuckDB integration plan and operating model.
- **PyArrow & DuckDB guides**: Reference documentation for best-in-class capabilities.
