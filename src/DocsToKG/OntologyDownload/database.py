# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.database",
#   "purpose": "DuckDB catalog for ontology versions, artifacts, extracted files, validations, and provenance",
#   "sections": [
#     {"id": "init", "name": "Initialization & Bootstrap", "anchor": "INI", "kind": "api"},
#     {"id": "migrations", "name": "Schema Migrations", "anchor": "MIG", "kind": "infra"},
#     {"id": "transactions", "name": "Transaction Boundaries", "anchor": "TXN", "kind": "api"},
#     {"id": "queries", "name": "Query Facades", "anchor": "QRY", "kind": "api"},
#     {"id": "types", "name": "Data Transfer Objects", "anchor": "DTO", "kind": "models"}
#   ]
# }
# === /NAVMAP ===

"""DuckDB catalog integration for OntologyDownload.

This module provides a transactional, single-node catalog for tracking
ontology versions, artifacts, extracted files, validations, and events.
Binary payloads live on the filesystem; DuckDB stores metadata and lineage.

Key design principles:
- Filesystem layout: ontologies/<service>/<version>/ with optional CAS mirrors
- DB location: <PYSTOW_HOME>/.catalog/ontofetch.duckdb
- One writer at a time (file lock); many readers
- Two-phase choreography: FS write then DB commit for atomicity
- Idempotence via content hashing (sha256)
- Query facades encapsulate SQL; no leakage to callers
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for the OntologyDownload catalog. "
        "Ensure the project .venv is properly initialized."
    ) from exc

from .errors import OntologyDownloadError
from .settings import DatabaseConfiguration

logger = logging.getLogger(__name__)


# ============================================================================
# Data Transfer Objects (DTO) — API layer
# ============================================================================


@dataclass
class VersionRow:
    """A logical release of a fetched ontology corpus."""

    version_id: str  # e.g., "2025-10-20T01:23:45Z"
    service: str  # OLS, BioPortal, etc.
    created_at: datetime
    plan_hash: Optional[str] = None


@dataclass
class ArtifactRow:
    """Downloaded archive metadata."""

    artifact_id: str  # sha256(archive bytes)
    version_id: str
    service: str
    source_url: str
    size_bytes: int
    fs_relpath: str  # Path under ontologies/ root (POSIX)
    status: str  # 'fresh', 'cached', 'failed'
    etag: Optional[str] = None
    last_modified: Optional[datetime] = None
    content_type: Optional[str] = None


@dataclass
class FileRow:
    """Extracted file metadata."""

    file_id: str  # sha256(file bytes)
    artifact_id: str
    version_id: str
    relpath_in_version: str
    format: str  # rdf, ttl, owl, obo, other
    size_bytes: int
    mtime: Optional[datetime] = None
    cas_relpath: Optional[str] = None


@dataclass
class ValidationRow:
    """Validator outcome (SHACL, ROBOT, Arelle, custom, etc.)."""

    validation_id: str  # e.g., ULID or sha256(file_id|validator|run_at)
    file_id: str
    validator: str  # 'rdflib', 'ROBOT', 'Arelle', 'Custom:<name>', ...
    passed: bool
    run_at: datetime
    details_json: Optional[Dict[str, Any]] = None


@dataclass
class PlanRow:
    """Cached plan for an ontology specification."""

    plan_id: str  # sha256(ontology_id + resolver + timestamp)
    ontology_id: str
    resolver: str
    version: Optional[str]
    url: str
    service: Optional[str]
    license: Optional[str]
    media_type: Optional[str]
    content_length: Optional[int]
    cached_at: datetime
    plan_json: Dict[str, Any]  # Full serialized PlannedFetch
    is_current: bool = False  # Mark as current for plan-diff


@dataclass
class PlanDiffRow:
    """Historical comparison between two plan runs."""

    diff_id: str  # ULID or sha256(older_plan_id + newer_plan_id)
    older_plan_id: str
    newer_plan_id: str
    ontology_id: str
    comparison_at: datetime
    added_count: int
    removed_count: int
    modified_count: int
    diff_json: Dict[str, Any]  # Full diff payload


@dataclass
class VersionStats:
    """Summary statistics for a version."""

    version_id: str
    service: str
    created_at: datetime
    files: int
    bytes: int
    validations_passed: int
    validations_failed: int


# ============================================================================
# Schema & Migrations
# ============================================================================


_MIGRATIONS: List[Tuple[str, str]] = [
    (
        "0001_init",
        """
        -- Schema versioning
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT now()
        );

        -- Logical release of a fetched corpus
        CREATE TABLE IF NOT EXISTS versions (
            version_id TEXT PRIMARY KEY,
            service TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT now(),
            plan_hash TEXT
        );

        -- Downloaded archive metadata
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL,
            service TEXT NOT NULL,
            source_url TEXT NOT NULL,
            etag TEXT,
            last_modified TIMESTAMP,
            content_type TEXT,
            size_bytes BIGINT NOT NULL,
            fs_relpath TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('fresh', 'cached', 'failed')),
            UNIQUE (version_id, fs_relpath)
        );

        -- Pointer to the latest version
        CREATE TABLE IF NOT EXISTS latest_pointer (
            slot TEXT PRIMARY KEY DEFAULT 'default',
            version_id TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT now(),
            updated_by TEXT
        );

        -- Indexes for fast lookups
        CREATE INDEX IF NOT EXISTS idx_versions_service_created
            ON versions(service, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_artifacts_version
            ON artifacts(version_id);

        CREATE INDEX IF NOT EXISTS idx_artifacts_service
            ON artifacts(service);

        CREATE INDEX IF NOT EXISTS idx_artifacts_source_url
            ON artifacts(source_url);

        INSERT OR IGNORE INTO schema_version VALUES ('0001_init', now());
        """,
    ),
    (
        "0002_files",
        """
        -- Extracted file catalog
        CREATE TABLE IF NOT EXISTS extracted_files (
            file_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            relpath_in_version TEXT NOT NULL,
            format TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            mtime TIMESTAMP,
            cas_relpath TEXT,
            UNIQUE (version_id, relpath_in_version)
        );

        CREATE INDEX IF NOT EXISTS idx_files_version
            ON extracted_files(version_id);

        CREATE INDEX IF NOT EXISTS idx_files_format
            ON extracted_files(format);

        CREATE INDEX IF NOT EXISTS idx_files_artifact
            ON extracted_files(artifact_id);

        INSERT OR IGNORE INTO schema_version VALUES ('0002_files', now());
        """,
    ),
    (
        "0003_validations",
        """
        -- Validator outcomes (SHACL, ROBOT, Arelle, custom, etc.)
        CREATE TABLE IF NOT EXISTS validations (
            validation_id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            validator TEXT NOT NULL,
            passed BOOLEAN NOT NULL,
            details_json JSON,
            run_at TIMESTAMP NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_validations_file
            ON validations(file_id);

        CREATE INDEX IF NOT EXISTS idx_validations_validator_time
            ON validations(validator, run_at DESC);

        INSERT OR IGNORE INTO schema_version VALUES ('0003_validations', now());
        """,
    ),
    (
        "0004_events",
        """
        -- Append-only structured observability events (optional; can be Parquet instead)
        CREATE TABLE IF NOT EXISTS events (
            run_id TEXT NOT NULL,
            ts TIMESTAMP NOT NULL DEFAULT now(),
            type TEXT NOT NULL,
            level TEXT NOT NULL,
            payload JSON NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_run_time
            ON events(run_id, ts DESC);

        INSERT OR IGNORE INTO schema_version VALUES ('0004_events', now());
        """,
    ),
    (
        "0005_plans",
        """
        -- Cached plans for ontology specifications
        CREATE TABLE IF NOT EXISTS plans (
            plan_id TEXT PRIMARY KEY,
            ontology_id TEXT NOT NULL,
            resolver TEXT NOT NULL,
            version TEXT,
            url TEXT NOT NULL,
            service TEXT,
            license TEXT,
            media_type TEXT,
            content_length BIGINT,
            cached_at TIMESTAMP NOT NULL DEFAULT now(),
            plan_json JSON NOT NULL,
            is_current BOOLEAN NOT NULL DEFAULT FALSE
        );

        -- Historical plan diffs
        CREATE TABLE IF NOT EXISTS plan_diffs (
            diff_id TEXT PRIMARY KEY,
            older_plan_id TEXT NOT NULL,
            newer_plan_id TEXT NOT NULL,
            ontology_id TEXT NOT NULL,
            comparison_at TIMESTAMP NOT NULL DEFAULT now(),
            added_count INTEGER NOT NULL,
            removed_count INTEGER NOT NULL,
            modified_count INTEGER NOT NULL,
            diff_json JSON NOT NULL
        );

        -- Indexes for plans table
        CREATE INDEX IF NOT EXISTS idx_plans_ontology_id
            ON plans(ontology_id);

        CREATE INDEX IF NOT EXISTS idx_plans_is_current
            ON plans(ontology_id, is_current);

        CREATE INDEX IF NOT EXISTS idx_plans_cached_at
            ON plans(ontology_id, cached_at DESC);

        -- Indexes for plan_diffs table
        CREATE INDEX IF NOT EXISTS idx_plan_diffs_ontology
            ON plan_diffs(ontology_id, comparison_at DESC);

        CREATE INDEX IF NOT EXISTS idx_plan_diffs_older_plan
            ON plan_diffs(older_plan_id);

        INSERT OR IGNORE INTO schema_version VALUES ('0005_plans', now());
        """,
    ),
]


# ============================================================================
# DuckDB Connection & Bootstrap
# ============================================================================


class Database:
    """Transactional catalog for ontology metadata.

    Usage::

        db = Database(config)
        db.bootstrap()
        try:
            # ... queries ...
        finally:
            db.close()
    """

    def __init__(self, config: Optional[DatabaseConfiguration] = None):
        """Initialize database configuration."""

        self.config = config or DatabaseConfiguration()
        self._db_path = self._resolve_db_path()
        self._lock_path = Path(str(self._db_path) + ".lock")
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._lock_file: Optional[Any] = None
        self._local = threading.local()

    def _resolve_db_path(self) -> Path:
        """Resolve the database file path with defaults."""

        if self.config.db_path:
            return self.config.db_path

        # Default: ~/.data/ontology-fetcher/.catalog/ontofetch.duckdb
        pystow_home = Path.home() / ".data" / "ontology-fetcher" / ".catalog" / "ontofetch.duckdb"
        return pystow_home

    @contextlib.contextmanager
    def _write_lock(self) -> Generator[None, None, None]:
        """Acquire an exclusive file lock for writes."""

        if not self.config.enable_locks or self.config.readonly:
            yield
            return

        lock_file = self._lock_path
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Open lock file for exclusive lock
            self._lock_file = open(lock_file, "w")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            logger.debug(f"Acquired write lock at {lock_file}")
            yield
        finally:
            if self._lock_file:
                try:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                    self._lock_file.close()
                except Exception:
                    pass
                self._lock_file = None

    def bootstrap(self) -> None:
        """Initialize database, apply migrations, and prepare for operations."""

        db_path = self._db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Opening DuckDB at {db_path} (read_only={self.config.readonly})")

        config_dict = {}
        if self.config.threads is not None:
            config_dict["threads"] = self.config.threads
        elif os.cpu_count():
            config_dict["threads"] = os.cpu_count()

        if self.config.memory_limit is not None:
            config_dict["memory_limit"] = self.config.memory_limit

        self._connection = duckdb.connect(
            str(db_path),
            read_only=self.config.readonly,
            config=config_dict if config_dict else None,
        )

        if self.config.enable_object_cache:
            self._connection.execute("PRAGMA enable_object_cache;")

        if not self.config.readonly:
            with self._write_lock():
                self._apply_migrations()

    def _apply_migrations(self) -> None:
        """Apply pending schema migrations in a transaction."""

        assert self._connection is not None
        try:
            # Get current schema version
            result = self._connection.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            ).fetchall()
            current_version = result[0][0] if result else None
        except duckdb.CatalogException:
            current_version = None

        # Apply forward migrations
        for migration_name, migration_sql in _MIGRATIONS:
            if current_version is None or migration_name > current_version:
                logger.info(f"Applying migration: {migration_name}")
                self._connection.execute(migration_sql)

    def close(self) -> None:
        """Close the database connection."""

        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Database:
        """Context manager entry."""

        self.bootstrap()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""

        self.close()

    # ========================================================================
    # Transactions
    # ========================================================================

    @contextlib.contextmanager
    def transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Transactional context for batch writes."""

        assert self._connection is not None
        if self.config.readonly:
            raise OntologyDownloadError("Cannot write in read-only mode")

        with self._write_lock():
            self._connection.execute("BEGIN TRANSACTION")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception as e:
                self._connection.execute("ROLLBACK")
                logger.error(f"Transaction rolled back: {e}")
                raise

    # ========================================================================
    # Query Facades — Versions
    # ========================================================================

    def upsert_version(
        self, version_id: str, service: str, plan_hash: Optional[str] = None
    ) -> None:
        """Insert or update a version record."""

        assert self._connection is not None
        self._connection.execute(
            """
            INSERT OR REPLACE INTO versions (version_id, service, created_at, plan_hash)
            VALUES (?, ?, now(), ?)
            """,
            [version_id, service, plan_hash],
        )

    def get_latest_version(self, service: Optional[str] = None) -> Optional[VersionRow]:
        """Get the latest version, optionally filtered by service."""

        assert self._connection is not None
        if service:
            query = "SELECT * FROM latest_pointer WHERE slot = 'default'"
            result = self._connection.execute(query).fetchone()
            if not result:
                return None
            version_id = result[1]
        else:
            query = "SELECT * FROM latest_pointer WHERE slot = 'default'"
            result = self._connection.execute(query).fetchone()
            if not result:
                return None
            version_id = result[1]

        query = (
            "SELECT version_id, service, created_at, plan_hash FROM versions WHERE version_id = ?"
        )
        result = self._connection.execute(query, [version_id]).fetchone()
        if not result:
            return None
        return VersionRow(
            version_id=result[0], service=result[1], created_at=result[2], plan_hash=result[3]
        )

    def set_latest_version(self, version_id: str, by: Optional[str] = None) -> None:
        """Set or update the latest version pointer."""

        assert self._connection is not None
        self._connection.execute(
            """
            INSERT OR REPLACE INTO latest_pointer (slot, version_id, updated_at, updated_by)
            VALUES ('default', ?, now(), ?)
            """,
            [version_id, by],
        )

    def list_versions(self, service: Optional[str] = None, limit: int = 50) -> List[VersionRow]:
        """List versions, optionally filtered by service."""

        assert self._connection is not None
        if service:
            query = """
                SELECT version_id, service, created_at, plan_hash
                FROM versions WHERE service = ?
                ORDER BY created_at DESC LIMIT ?
            """
            results = self._connection.execute(query, [service, limit]).fetchall()
        else:
            query = """
                SELECT version_id, service, created_at, plan_hash
                FROM versions ORDER BY created_at DESC LIMIT ?
            """
            results = self._connection.execute(query, [limit]).fetchall()

        return [
            VersionRow(version_id=row[0], service=row[1], created_at=row[2], plan_hash=row[3])
            for row in results
        ]

    # ========================================================================
    # Query Facades — Artifacts
    # ========================================================================

    def upsert_artifact(
        self,
        artifact_id: str,
        version_id: str,
        service: str,
        source_url: str,
        size_bytes: int,
        fs_relpath: str,
        status: str,
        etag: Optional[str] = None,
        last_modified: Optional[datetime] = None,
        content_type: Optional[str] = None,
    ) -> None:
        """Insert or update an artifact record."""

        assert self._connection is not None
        # DuckDB requires explicit conflict target for tables with multiple unique constraints
        # Use DELETE+INSERT for reliability
        self._connection.execute(
            "DELETE FROM artifacts WHERE artifact_id = ?",
            [artifact_id],
        )
        self._connection.execute(
            """
            INSERT INTO artifacts
            (artifact_id, version_id, service, source_url, size_bytes, fs_relpath, status, etag, last_modified, content_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                artifact_id,
                version_id,
                service,
                source_url,
                size_bytes,
                fs_relpath,
                status,
                etag,
                last_modified,
                content_type,
            ],
        )

    def list_artifacts(self, version_id: str, status: Optional[str] = None) -> List[ArtifactRow]:
        """List artifacts for a version, optionally filtered by status."""

        assert self._connection is not None
        if status:
            query = """
                SELECT artifact_id, version_id, service, source_url, size_bytes, fs_relpath, status,
                       etag, last_modified, content_type
                FROM artifacts WHERE version_id = ? AND status = ?
            """
            results = self._connection.execute(query, [version_id, status]).fetchall()
        else:
            query = """
                SELECT artifact_id, version_id, service, source_url, size_bytes, fs_relpath, status,
                       etag, last_modified, content_type
                FROM artifacts WHERE version_id = ?
            """
            results = self._connection.execute(query, [version_id]).fetchall()

        return [
            ArtifactRow(
                artifact_id=row[0],
                version_id=row[1],
                service=row[2],
                source_url=row[3],
                size_bytes=row[4],
                fs_relpath=row[5],
                status=row[6],
                etag=row[7],
                last_modified=row[8],
                content_type=row[9],
            )
            for row in results
        ]

    # ========================================================================
    # Query Facades — Extracted Files
    # ========================================================================

    def insert_extracted_files(self, files: List[FileRow]) -> None:
        """Batch insert extracted file records."""

        assert self._connection is not None
        for f in files:
            # Delete first to handle potential duplicate unique constraint
            self._connection.execute(
                "DELETE FROM extracted_files WHERE file_id = ?",
                [f.file_id],
            )
            self._connection.execute(
                """
                INSERT INTO extracted_files
                (file_id, artifact_id, version_id, relpath_in_version, format, size_bytes, mtime, cas_relpath)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    f.file_id,
                    f.artifact_id,
                    f.version_id,
                    f.relpath_in_version,
                    f.format,
                    f.size_bytes,
                    f.mtime,
                    f.cas_relpath,
                ],
            )

    def list_extracted_files(
        self, version_id: str, format_filter: Optional[str] = None
    ) -> List[FileRow]:
        """List extracted files for a version, optionally filtered by format."""

        assert self._connection is not None
        if format_filter:
            query = """
                SELECT file_id, artifact_id, version_id, relpath_in_version, format, size_bytes, mtime, cas_relpath
                FROM extracted_files WHERE version_id = ? AND format = ?
            """
            results = self._connection.execute(query, [version_id, format_filter]).fetchall()
        else:
            query = """
                SELECT file_id, artifact_id, version_id, relpath_in_version, format, size_bytes, mtime, cas_relpath
                FROM extracted_files WHERE version_id = ?
            """
            results = self._connection.execute(query, [version_id]).fetchall()

        return [
            FileRow(
                file_id=row[0],
                artifact_id=row[1],
                version_id=row[2],
                relpath_in_version=row[3],
                format=row[4],
                size_bytes=row[5],
                mtime=row[6],
                cas_relpath=row[7],
            )
            for row in results
        ]

    # ========================================================================
    # Query Facades — Validations
    # ========================================================================

    def insert_validations(self, validations: List[ValidationRow]) -> None:
        """Batch insert validation records."""

        assert self._connection is not None
        for v in validations:
            details_json = json.dumps(v.details_json) if v.details_json else None
            self._connection.execute(
                """
                INSERT OR IGNORE INTO validations
                (validation_id, file_id, validator, passed, details_json, run_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [v.validation_id, v.file_id, v.validator, v.passed, details_json, v.run_at],
            )

    def get_validation_failures(self, version_id: str) -> List[ValidationRow]:
        """Get all validation failures for a version."""

        assert self._connection is not None
        query = """
            SELECT v.validation_id, v.file_id, v.validator, v.passed, v.details_json, v.run_at
            FROM validations v
            JOIN extracted_files f ON v.file_id = f.file_id
            WHERE f.version_id = ? AND v.passed = FALSE
            ORDER BY v.run_at DESC
        """
        results = self._connection.execute(query, [version_id]).fetchall()

        return [
            ValidationRow(
                validation_id=row[0],
                file_id=row[1],
                validator=row[2],
                passed=row[3],
                details_json=json.loads(row[4]) if row[4] else None,
                run_at=row[5],
            )
            for row in results
        ]

    # ========================================================================
    # Query Facades — Statistics & Reporting
    # ========================================================================

    def get_version_stats(self, version_id: str) -> Optional[VersionStats]:
        """Get summary statistics for a version."""

        assert self._connection is not None
        query = """
            SELECT
                v.version_id,
                v.service,
                v.created_at,
                COUNT(DISTINCT f.file_id) AS files,
                COALESCE(SUM(f.size_bytes), 0) AS bytes,
                SUM(CASE WHEN val.passed THEN 1 ELSE 0 END) AS validations_passed,
                SUM(CASE WHEN val.passed THEN 0 ELSE 1 END) AS validations_failed
            FROM versions v
            LEFT JOIN extracted_files f ON f.version_id = v.version_id
            LEFT JOIN (
                SELECT file_id, BOOL_OR(passed) AS passed
                FROM validations
                GROUP BY 1
            ) val ON val.file_id = f.file_id
            WHERE v.version_id = ?
            GROUP BY 1, 2, 3
        """

        result = self._connection.execute(query, [version_id]).fetchone()
        if not result:
            return None

        return VersionStats(
            version_id=result[0],
            service=result[1],
            created_at=result[2],
            files=result[3],
            bytes=result[4],
            validations_passed=result[5],
            validations_failed=result[6],
        )

    # ========================================================================
    # Query Facades — Prune & Orphan Detection
    # ========================================================================

    def stage_filesystem_listing(
        self, scope: str, entries: List[Tuple[str, int, Optional[datetime]]]
    ) -> None:
        """Stage filesystem entries for orphan detection.

        Args:
            scope: 'cas' or 'version'
            entries: List of (relpath, size_bytes, mtime) tuples
        """

        assert self._connection is not None
        self._connection.execute("DELETE FROM staging_fs_listing WHERE scope = ?", [scope])

        for relpath, size_bytes, mtime in entries:
            self._connection.execute(
                """
                INSERT INTO staging_fs_listing (scope, relpath, size_bytes, mtime)
                VALUES (?, ?, ?, ?)
                """,
                [scope, relpath, size_bytes, mtime],
            )

    def get_orphaned_files(self, scope: str) -> List[Tuple[str, int]]:
        """Get orphaned files (present in FS but not in catalog).

        Returns:
            List of (relpath, size_bytes) tuples
        """

        assert self._connection is not None
        # Create staging table if not exists
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_fs_listing (
                scope TEXT NOT NULL,
                relpath TEXT NOT NULL,
                size_bytes BIGINT NOT NULL,
                mtime TIMESTAMP
            )
            """
        )

        query = """
            SELECT s.relpath, s.size_bytes
            FROM staging_fs_listing s
            LEFT JOIN (
                SELECT fs_relpath AS relpath FROM artifacts
                UNION ALL
                SELECT CONCAT(service, '/', version_id, '/', relpath_in_version) FROM extracted_files
            ) cat ON cat.relpath = s.relpath
            WHERE s.scope = ? AND cat.relpath IS NULL
        """

        results = self._connection.execute(query, [scope]).fetchall()
        return results or []

    # ========================================================================
    # PHASE 4: Plan Caching & Comparison Queries
    # ========================================================================

    def upsert_plan(
        self,
        plan_id: str,
        ontology_id: str,
        resolver: str,
        plan_json: Dict[str, Any],
        is_current: bool = False,
    ) -> None:
        """Store or update a cached plan.

        Args:
            plan_id: Unique identifier (sha256 hash or ULID)
            ontology_id: e.g., 'hp', 'chebi'
            resolver: e.g., 'obo', 'ols', 'bioportal'
            plan_json: Full serialized PlannedFetch as dictionary
            is_current: Whether this is the current/latest plan for this ontology
        """

        assert self._connection is not None
        version = plan_json.get("version")
        url = plan_json.get("url", "")
        service = plan_json.get("service")
        license_str = plan_json.get("license")
        media_type = plan_json.get("media_type")
        content_length = plan_json.get("content_length")

        # Mark previous plans for this ontology as non-current if this one is current
        if is_current:
            self._connection.execute(
                "UPDATE plans SET is_current = FALSE WHERE ontology_id = ?",
                [ontology_id],
            )

        # Delete and re-insert to ensure idempotence
        self._connection.execute("DELETE FROM plans WHERE plan_id = ?", [plan_id])
        self._connection.execute(
            """
            INSERT INTO plans (
                plan_id, ontology_id, resolver, version, url, service, license,
                media_type, content_length, plan_json, is_current
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                plan_id,
                ontology_id,
                resolver,
                version,
                url,
                service,
                license_str,
                media_type,
                content_length,
                json.dumps(plan_json),
                is_current,
            ],
        )

    def get_current_plan(self, ontology_id: str) -> Optional[PlanRow]:
        """Retrieve the current/latest plan for an ontology.

        Args:
            ontology_id: e.g., 'hp'

        Returns:
            PlanRow if found, else None
        """

        assert self._connection is not None
        result = self._connection.execute(
            """
            SELECT plan_id, ontology_id, resolver, version, url, service, license,
                   media_type, content_length, cached_at, plan_json, is_current
            FROM plans
            WHERE ontology_id = ? AND is_current = TRUE
            ORDER BY cached_at DESC
            LIMIT 1
            """,
            [ontology_id],
        ).fetchone()

        if not result:
            return None

        return PlanRow(
            plan_id=result[0],
            ontology_id=result[1],
            resolver=result[2],
            version=result[3],
            url=result[4],
            service=result[5],
            license=result[6],
            media_type=result[7],
            content_length=result[8],
            cached_at=result[9],
            plan_json=json.loads(result[10]) if isinstance(result[10], str) else result[10],
            is_current=result[11],
        )

    def list_plans(self, ontology_id: Optional[str] = None, limit: int = 100) -> List[PlanRow]:
        """List cached plans, optionally filtered by ontology.

        Args:
            ontology_id: Optional filter; if None, return all plans
            limit: Maximum number of results

        Returns:
            List of PlanRow objects ordered by most recent first
        """

        assert self._connection is not None
        if ontology_id:
            query = """
                SELECT plan_id, ontology_id, resolver, version, url, service, license,
                       media_type, content_length, cached_at, plan_json, is_current
                FROM plans
                WHERE ontology_id = ?
                ORDER BY cached_at DESC
                LIMIT ?
            """
            params = [ontology_id, limit]
        else:
            query = """
                SELECT plan_id, ontology_id, resolver, version, url, service, license,
                       media_type, content_length, cached_at, plan_json, is_current
                FROM plans
                ORDER BY cached_at DESC
                LIMIT ?
            """
            params = [limit]

        results = self._connection.execute(query, params).fetchall()
        return [
            PlanRow(
                plan_id=r[0],
                ontology_id=r[1],
                resolver=r[2],
                version=r[3],
                url=r[4],
                service=r[5],
                license=r[6],
                media_type=r[7],
                content_length=r[8],
                cached_at=r[9],
                plan_json=json.loads(r[10]) if isinstance(r[10], str) else r[10],
                is_current=r[11],
            )
            for r in results
        ]

    def insert_plan_diff(
        self,
        diff_id: str,
        older_plan_id: str,
        newer_plan_id: str,
        ontology_id: str,
        diff_result: Dict[str, Any],
    ) -> None:
        """Store the result of a plan comparison.

        Args:
            diff_id: Unique identifier for this diff comparison
            older_plan_id: Plan ID of the baseline
            newer_plan_id: Plan ID of the current
            ontology_id: Ontology being compared
            diff_result: Dict with keys: added (list), removed (list), modified (list)
        """

        assert self._connection is not None
        added_count = len(diff_result.get("added", []))
        removed_count = len(diff_result.get("removed", []))
        modified_count = len(diff_result.get("modified", []))

        self._connection.execute(
            """
            INSERT INTO plan_diffs (
                diff_id, older_plan_id, newer_plan_id, ontology_id,
                added_count, removed_count, modified_count, diff_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                diff_id,
                older_plan_id,
                newer_plan_id,
                ontology_id,
                added_count,
                removed_count,
                modified_count,
                json.dumps(diff_result),
            ],
        )

    def get_plan_diff_history(self, ontology_id: str, limit: int = 10) -> List[PlanDiffRow]:
        """Retrieve historical plan diffs for an ontology.

        Args:
            ontology_id: e.g., 'hp'
            limit: Maximum number of diffs to return

        Returns:
            List of PlanDiffRow objects ordered by most recent first
        """

        assert self._connection is not None
        results = self._connection.execute(
            """
            SELECT diff_id, older_plan_id, newer_plan_id, ontology_id, comparison_at,
                   added_count, removed_count, modified_count, diff_json
            FROM plan_diffs
            WHERE ontology_id = ?
            ORDER BY comparison_at DESC
            LIMIT ?
            """,
            [ontology_id, limit],
        ).fetchall()

        return [
            PlanDiffRow(
                diff_id=r[0],
                older_plan_id=r[1],
                newer_plan_id=r[2],
                ontology_id=r[3],
                comparison_at=r[4],
                added_count=r[5],
                removed_count=r[6],
                modified_count=r[7],
                diff_json=json.loads(r[8]) if isinstance(r[8], str) else r[8],
            )
            for r in results
        ]


# ============================================================================
# Global Singleton for Convenience
# ============================================================================


_db_singleton: Optional[Database] = None
_db_lock = threading.Lock()


def get_database(config: Optional[DatabaseConfiguration] = None) -> Database:
    """Get or create a global database instance.

    Thread-safe singleton pattern.
    """

    global _db_singleton
    if _db_singleton is None:
        with _db_lock:
            if _db_singleton is None:
                _db_singleton = Database(config)
                _db_singleton.bootstrap()
    return _db_singleton


def close_database() -> None:
    """Close the global database instance."""

    global _db_singleton
    if _db_singleton:
        _db_singleton.close()
        _db_singleton = None
