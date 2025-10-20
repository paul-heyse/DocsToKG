# DuckDB Database Integration Guide for OntologyDownload

## Overview

This guide shows how to integrate the new DuckDB catalog module into the OntologyDownload pipeline, enabling transactional metadata tracking, deterministic versioning, and fast queries over ontology artifacts and validation results.

---

## Status

- ✅ **Phase 1: CLI Integration** — COMPLETE
  - Added `db` subcommand with `latest`, `versions`, `stats`, `files`, `validations` sub-subcommands
  - Integrated database queries into CLI with JSON/table output formatting

- ✅ **Phase 2: Doctor Command Integration** — COMPLETE
  - Added `_database_health_check()` function
  - Integrated database health checks into `doctor` command
  - Shows catalog statistics (versions, artifacts, files, validations)
  - Reports database status and schema migration count

- ⏳ **Phase 3: Prune Command Integration** — PENDING
  - Add orphan detection using database
  - Implement dry-run and apply modes

- ⏳ **Phase 4: Plan & Plan-Diff Integration** — PENDING
  - Cache planning decisions in database
  - Enable replay and comparison of plan runs

- ⏳ **Phase 5: Export & Reporting** — PENDING
  - Export database state for dashboards
  - Enable downstream analytics

---

## Phase 1: CLI Integration

### 1.1 Add Database Configuration to Settings

In `planning.py` and `api.py`, import the database module:

```python
from DocsToKG.OntologyDownload.database import (
    Database,
    DatabaseConfiguration,
    get_database,
    close_database,
    FileRow,
    ValidationRow,
    VersionRow,
)
```

### 1.2 Update fetch_all / plan_all to Bootstrap Database

Modify the main entry point `fetch_all` in `planning.py`:

```python
def fetch_all(specs: List[FetchSpec], config: ResolvedConfig) -> List[Manifest]:
    """Download and validate ontologies, recording metadata in DuckDB catalog."""

    # Initialize database
    db_config = DatabaseConfiguration(
        db_path=config.database.db_path,
        readonly=False,
        enable_locks=True,
        threads=config.http.concurrent_downloads,
    )
    db = Database(db_config)
    db.bootstrap()

    try:
        # ... existing fetch logic ...

        # At extraction boundary (after extract_archive_safe succeeds)
        with db.transaction():
            # Record version
            db.upsert_version(version_id, service, plan_hash=plan_hash)

            # Record artifacts
            for artifact in artifacts:
                db.upsert_artifact(
                    artifact_id=artifact.id,
                    version_id=version_id,
                    service=service,
                    source_url=artifact.url,
                    size_bytes=artifact.size,
                    fs_relpath=artifact.fs_path,
                    status="fresh",
                    etag=artifact.etag,
                    last_modified=artifact.last_modified,
                    content_type=artifact.content_type,
                )

            # Record extracted files
            file_rows = [
                FileRow(
                    file_id=sha256(file_bytes),
                    artifact_id=artifact.id,
                    version_id=version_id,
                    relpath_in_version=file_relpath,
                    format=detect_format(file_relpath),
                    size_bytes=len(file_bytes),
                )
                for artifact, (file_relpath, file_bytes) in extracted_items
            ]
            db.insert_extracted_files(file_rows)

            # Set latest version
            db.set_latest_version(version_id, by=os.getenv("USER"))

        # At validation boundary (after run_validators completes)
        with db.transaction():
            validation_rows = [
                ValidationRow(
                    validation_id=f"{result.file_id}:{result.validator}:{int(time.time())}",
                    file_id=result.file_id,
                    validator=result.validator,
                    passed=result.passed,
                    run_at=datetime.now(timezone.utc),
                    details_json=result.details,
                )
                for result in validation_results
            ]
            db.insert_validations(validation_rows)

    finally:
        db.close()
```

### 1.3 Add CLI Subcommands for Database Queries

In `cli.py`, add new subcommands for database introspection:

```python
import argparse
from DocsToKG.OntologyDownload.database import get_database, close_database

def add_db_subparsers(subparsers):
    """Add database query subcommands."""

    db_parser = subparsers.add_parser(
        "db",
        help="Query the ontology metadata catalog",
    )
    db_subs = db_parser.add_subparsers(dest="db_cmd")

    # db latest
    latest_parser = db_subs.add_parser("latest", help="Show latest version")
    latest_parser.set_defaults(func=cmd_db_latest)

    # db versions
    versions_parser = db_subs.add_parser("versions", help="List versions")
    versions_parser.add_argument("--service", help="Filter by service")
    versions_parser.add_argument("--limit", type=int, default=50)
    versions_parser.set_defaults(func=cmd_db_versions)

    # db stats
    stats_parser = db_subs.add_parser("stats", help="Get version statistics")
    stats_parser.add_argument("version_id")
    stats_parser.set_defaults(func=cmd_db_stats)

    # db files
    files_parser = db_subs.add_parser("files", help="List extracted files")
    files_parser.add_argument("version_id")
    files_parser.add_argument("--format", help="Filter by format")
    files_parser.set_defaults(func=cmd_db_files)

    # db validations
    val_parser = db_subs.add_parser("validations", help="Show validation failures")
    val_parser.add_argument("version_id")
    val_parser.set_defaults(func=cmd_db_validations)


def cmd_db_latest(args):
    """Show latest version."""
    db = get_database()
    try:
        latest = db.get_latest_version()
        if latest:
            print(f"Latest: {latest.version_id} ({latest.service})")
            print(f"Created: {latest.created_at}")
        else:
            print("No versions recorded.")
    finally:
        close_database()


def cmd_db_versions(args):
    """List versions."""
    db = get_database()
    try:
        versions = db.list_versions(service=args.service, limit=args.limit)
        for v in versions:
            print(f"{v.version_id:30} {v.service:15} {v.created_at}")
    finally:
        close_database()


def cmd_db_stats(args):
    """Show version statistics."""
    db = get_database()
    try:
        stats = db.get_version_stats(args.version_id)
        if stats:
            print(f"Version: {stats.version_id}")
            print(f"Service: {stats.service}")
            print(f"Files: {stats.files}")
            print(f"Total size: {stats.bytes / (1024**3):.2f} GB")
            print(f"Validations passed: {stats.validations_passed}")
            print(f"Validations failed: {stats.validations_failed}")
        else:
            print(f"Version not found: {args.version_id}")
    finally:
        close_database()
```

---

## Phase 2: Doctor Command Integration

Update the `doctor` command to include database health checks:

```python
def cmd_doctor(args):
    """Audit environment, disk, credentials, and database consistency."""

    # ... existing checks ...

    # Database checks
    try:
        print("\n[Database]")
        db_config = DatabaseConfiguration()
        db_path = db_config.db_path or Path.home() / ".data" / "ontology-fetcher" / ".catalog" / "ontofetch.duckdb"

        if db_path.exists():
            db = Database(db_config)
            db.bootstrap()

            # Check schema
            result = db._connection.execute(
                "SELECT COUNT(*) FROM schema_version"
            ).fetchone()
            print(f"✓ Database initialized with {result[0]} migrations")

            # Check versions
            versions = db.list_versions(limit=1)
            print(f"✓ Recorded versions: {len(versions)}")

            db.close()
        else:
            print(f"ℹ Database not yet created at {db_path}")

    except Exception as e:
        print(f"✗ Database error: {e}")
        if args.fix:
            print("  Attempting recovery...")
            # Optionally recreate/repair database
```

---

## Phase 3: Prune Command Integration

Update prune to use database for orphan detection:

```python
def cmd_prune(args):
    """Remove orphaned ontology artifacts and files."""

    db = get_database()
    try:
        # Scan filesystem
        fs_entries = []
        ontology_root = LOCAL_ONTOLOGY_DIR
        for version_dir in ontology_root.glob("*/*/"):
            for f in version_dir.rglob("*"):
                if f.is_file():
                    relpath = f.relative_to(ontology_root)
                    fs_entries.append((str(relpath), f.stat().st_size, f.stat().st_mtime))

        # Stage filesystem listing
        db.stage_filesystem_listing("version", fs_entries)

        # Get orphans
        orphans = db.get_orphaned_files("version")

        if args.dry_run:
            total_bytes = sum(size for _, size in orphans)
            print(f"Would remove {len(orphans)} orphaned files ({total_bytes / (1024**3):.2f} GB)")
            for relpath, size in orphans[:10]:
                print(f"  {relpath} ({size} bytes)")
            if len(orphans) > 10:
                print(f"  ... and {len(orphans) - 10} more")
        else:
            for relpath, size in orphans:
                (ontology_root / relpath).unlink()
            print(f"Removed {len(orphans)} orphaned files")

    finally:
        close_database()
```

---

## Phase 4: Plan & Plan-Diff Integration

Update manifests to include database lookups:

```python
def plan_all(specs: List[FetchSpec], config: ResolvedConfig) -> List[PlannedFetch]:
    """Create a plan from specs and existing database state."""

    db = get_database()
    try:
        plans = []
        for spec in specs:
            # Check if version already in DB
            existing_versions = db.list_versions(service=spec.service, limit=1)

            if existing_versions and not config.force:
                # Use existing version; skip planning
                logger.info(f"Using cached version: {spec.service}/{existing_versions[0].version_id}")
                plans.append(PlannedFetch(
                    spec=spec,
                    url=None,
                    version_id=existing_versions[0].version_id,
                    from_cache=True,
                ))
            else:
                # Plan new fetch
                plans.append(plan_fetch(spec, config))

        return plans

    finally:
        close_database()
```

---

## Phase 5: Export & Reporting

Add methods to export database state for dashboards:

```python
def export_manifest_from_db(version_id: str) -> Dict[str, Any]:
    """Export version metadata from database as manifest JSON."""

    db = get_database()
    try:
        stats = db.get_version_stats(version_id)
        artifacts = db.list_artifacts(version_id)
        files = db.list_extracted_files(version_id)
        failures = db.get_validation_failures(version_id)

        return {
            "schema_version": "1.0",
            "version_id": version_id,
            "service": stats.service if stats else None,
            "created_at": stats.created_at.isoformat() if stats else None,
            "statistics": {
                "files": stats.files if stats else 0,
                "bytes": stats.bytes if stats else 0,
                "validations_passed": stats.validations_passed if stats else 0,
                "validations_failed": stats.validations_failed if stats else 0,
            },
            "artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "source_url": a.source_url,
                    "size_bytes": a.size_bytes,
                    "status": a.status,
                }
                for a in artifacts
            ],
            "files": [
                {
                    "relpath": f.relpath_in_version,
                    "format": f.format,
                    "size_bytes": f.size_bytes,
                }
                for f in files
            ],
            "validation_failures": [
                {
                    "file": v.file_id,
                    "validator": v.validator,
                    "details": v.details_json,
                }
                for v in failures
            ],
        }

    finally:
        close_database()
```

---

## Testing the Integration

### Unit Tests

```bash
./.venv/bin/pytest tests/ontology_download/test_database.py -v
```

### Integration Tests

```python
# tests/ontology_download/test_integration_database.py
def test_full_pipeline_with_database(tmp_path):
    """Test complete fetch → extract → validate → db workflow."""

    # Create test setup
    config = DatabaseConfiguration(db_path=tmp_path / "test.duckdb")

    with Database(config) as db:
        # Simulate fetch
        db.upsert_version("v1.0", "OLS")
        db.upsert_artifact("art1", "v1.0", "OLS", "http://example.com", 1024, "OLS/v1.0/file", "fresh")

        # Simulate extraction
        files = [FileRow("f1", "art1", "v1.0", "data/f1.ttl", "ttl", 512)]
        db.insert_extracted_files(files)

        # Simulate validation
        validations = [ValidationRow("v1", "f1", "rdflib", True, datetime.now(timezone.utc))]
        db.insert_validations(validations)

        # Set latest
        db.set_latest_version("v1.0")

        # Query back
        stats = db.get_version_stats("v1.0")
        assert stats.files == 1
        assert stats.bytes == 512
        assert stats.validations_passed == 1
```

---

## Best Practices

1. **Always use transactions**: Wrap related DB writes in `with db.transaction():` blocks.
2. **Bootstrap once per process**: Use `get_database()` singleton rather than creating new connections.
3. **Close on exit**: Always call `close_database()` or use context managers.
4. **Idempotent operations**: All upserts are safe to retry without causing duplicates.
5. **Two-phase choreography**: Write to filesystem first, then record in DB within a transaction.
6. **Read-only for queries**: Open with `readonly=True` when only reading data.

---

## Troubleshooting

### Database File Locked

If you see "Database file locked", it means a writer is in progress. Solutions:

```bash
# Check if process is hung
lsof ~/.data/ontology-fetcher/.catalog/ontofetch.duckdb

# Remove stale lock file (if safe)
rm ~/.data/ontology-fetcher/.catalog/ontofetch.duckdb.lock
```

### Migration Errors

If migrations fail to apply:

```bash
# Check current schema version
./.venv/bin/python -c "
from DocsToKG.OntologyDownload.database import Database
db = Database()
db.bootstrap()
result = db._connection.execute('SELECT * FROM schema_version').fetchall()
print(result)
db.close()
"

# Manually inspect table
./.venv/bin/python -c "
import duckdb
con = duckdb.connect(str(Path.home() / '.data/ontology-fetcher/.catalog/ontofetch.duckdb'))
print(con.execute('DESCRIBE versions').df())
con.close()
"
```

### Out-of-Memory Errors

If queries run out of memory:

```python
# Reduce memory limit
config = DatabaseConfiguration(memory_limit="4GB")
db = Database(config)
```

---

## See Also

- `src/DocsToKG/OntologyDownload/database.py`: Main implementation
- `tests/ontology_download/test_database.py`: Comprehensive tests
- `src/DocsToKG/OntologyDownload/DATABASE.md`: Full API documentation
- **Reference docs**: Ontology-database-layout.md, Ontology-database-scope.md
