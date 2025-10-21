# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.queries",
#   "purpose": "Type-safe query façades for DuckDB catalog operations",
#   "sections": [
#     {"id": "types", "name": "Data Transfer Objects", "anchor": "DTO", "kind": "models"},
#     {"id": "versions", "name": "Version Queries", "anchor": "VER", "kind": "api"},
#     {"id": "artifacts", "name": "Artifact Queries", "anchor": "ART", "kind": "api"},
#     {"id": "files", "name": "File Queries", "anchor": "FIL", "kind": "api"},
#     {"id": "validations", "name": "Validation Queries", "anchor": "VAL", "kind": "api"},
#     {"id": "stats", "name": "Statistics Queries", "anchor": "STA", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Type-safe query façades for DuckDB catalog.

All SQL is encapsulated here; callers work only with typed DTOs.
Façades enforce consistent patterns: lazy evaluation, result typing,
error handling, and resource cleanup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

try:  # pragma: no cover
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for catalog queries. Ensure .venv is initialized."
    ) from exc

logger = logging.getLogger(__name__)


# ============================================================================
# DATA TRANSFER OBJECTS (DTO)
# ============================================================================


@dataclass(frozen=True)
class VersionRow:
    """Version metadata row."""

    version_id: str
    service: str
    latest_pointer: bool
    ts: datetime

    @classmethod
    def from_tuple(cls, row: tuple) -> VersionRow:
        """Create from database tuple."""
        return cls(
            version_id=row[0],
            service=row[1],
            latest_pointer=bool(row[2]),
            ts=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3])),
        )


@dataclass(frozen=True)
class ArtifactRow:
    """Artifact metadata row."""

    artifact_id: str
    version_id: str
    fs_relpath: str
    size: int
    etag: Optional[str]
    status: str
    downloaded_at: datetime

    @classmethod
    def from_tuple(cls, row: tuple) -> ArtifactRow:
        """Create from database tuple."""
        return cls(
            artifact_id=row[0],
            version_id=row[1],
            fs_relpath=row[2],
            size=int(row[3]),
            etag=row[4],
            status=row[5],
            downloaded_at=(
                row[6] if isinstance(row[6], datetime) else datetime.fromisoformat(str(row[6]))
            ),
        )


@dataclass(frozen=True)
class FileRow:
    """Extracted file metadata row."""

    file_id: str
    artifact_id: str
    relpath: str
    size: int
    format: Optional[str]
    sha256: Optional[str]
    mtime: Optional[datetime]
    extracted_at: datetime

    @classmethod
    def from_tuple(cls, row: tuple) -> FileRow:
        """Create from database tuple."""
        mtime = row[6]
        if mtime is not None and not isinstance(mtime, datetime):
            mtime = datetime.fromisoformat(str(mtime))

        extracted_at = row[7]
        if not isinstance(extracted_at, datetime):
            extracted_at = datetime.fromisoformat(str(extracted_at))

        return cls(
            file_id=row[0],
            artifact_id=row[1],
            relpath=row[2],
            size=int(row[3]),
            format=row[4],
            sha256=row[5],
            mtime=mtime,
            extracted_at=extracted_at,
        )


@dataclass(frozen=True)
class ValidationRow:
    """Validation result row."""

    validation_id: str
    file_id: str
    validator: str
    status: str
    details: Optional[str]
    validated_at: datetime

    @classmethod
    def from_tuple(cls, row: tuple) -> ValidationRow:
        """Create from database tuple."""
        return cls(
            validation_id=row[0],
            file_id=row[1],
            validator=row[2],
            status=row[3],
            details=row[4],
            validated_at=(
                row[5] if isinstance(row[5], datetime) else datetime.fromisoformat(str(row[5]))
            ),
        )


# ============================================================================
# VERSION QUERIES (VER)
# ============================================================================


def list_versions(conn: duckdb.DuckDBPyConnection) -> List[VersionRow]:
    """List all versions, sorted by timestamp descending.

    Args:
        conn: DuckDB connection (read-only safe)

    Returns:
        List of VersionRow objects
    """
    result = conn.execute(
        """
        SELECT version_id, service, latest_pointer, ts
        FROM versions
        ORDER BY ts DESC
    """
    ).fetchall()

    return [VersionRow.from_tuple(row) for row in result]


def get_latest(
    conn: duckdb.DuckDBPyConnection, service: Optional[str] = None
) -> Optional[VersionRow]:
    """Get latest version, optionally filtered by service.

    Args:
        conn: DuckDB connection
        service: Optional service filter

    Returns:
        Latest VersionRow or None if not found
    """
    if service:
        result = conn.execute(
            """
            SELECT version_id, service, latest_pointer, ts
            FROM versions
            WHERE service = ? AND latest_pointer = TRUE
            ORDER BY ts DESC
            LIMIT 1
        """,
            [service],
        ).fetchone()
    else:
        result = conn.execute(
            """
            SELECT version_id, service, latest_pointer, ts
            FROM versions
            WHERE latest_pointer = TRUE
            ORDER BY ts DESC
            LIMIT 1
        """
        ).fetchone()

    return VersionRow.from_tuple(result) if result else None


def get_version(conn: duckdb.DuckDBPyConnection, version_id: str) -> Optional[VersionRow]:
    """Get specific version by ID.

    Args:
        conn: DuckDB connection
        version_id: Version ID to fetch

    Returns:
        VersionRow or None if not found
    """
    result = conn.execute(
        """
        SELECT version_id, service, latest_pointer, ts
        FROM versions
        WHERE version_id = ?
    """,
        [version_id],
    ).fetchone()

    return VersionRow.from_tuple(result) if result else None


# ============================================================================
# ARTIFACT QUERIES (ART)
# ============================================================================


def list_artifacts(conn: duckdb.DuckDBPyConnection, version_id: str) -> List[ArtifactRow]:
    """List all artifacts for a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID

    Returns:
        List of ArtifactRow objects
    """
    result = conn.execute(
        """
        SELECT artifact_id, version_id, fs_relpath, size, etag, status, downloaded_at
        FROM artifacts
        WHERE version_id = ?
        ORDER BY downloaded_at DESC
    """,
        [version_id],
    ).fetchall()

    return [ArtifactRow.from_tuple(row) for row in result]


def get_artifact(conn: duckdb.DuckDBPyConnection, artifact_id: str) -> Optional[ArtifactRow]:
    """Get specific artifact by ID.

    Args:
        conn: DuckDB connection
        artifact_id: Artifact ID

    Returns:
        ArtifactRow or None if not found
    """
    result = conn.execute(
        """
        SELECT artifact_id, version_id, fs_relpath, size, etag, status, downloaded_at
        FROM artifacts
        WHERE artifact_id = ?
    """,
        [artifact_id],
    ).fetchone()

    return ArtifactRow.from_tuple(result) if result else None


# ============================================================================
# FILE QUERIES (FIL)
# ============================================================================


def list_files(conn: duckdb.DuckDBPyConnection, version_id: str) -> List[FileRow]:
    """List all extracted files for a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID

    Returns:
        List of FileRow objects
    """
    result = conn.execute(
        """
        SELECT f.file_id, f.artifact_id, f.relpath, f.size, f.format, f.sha256, f.mtime, f.extracted_at
        FROM extracted_files f
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        WHERE a.version_id = ?
        ORDER BY f.relpath ASC
    """,
        [version_id],
    ).fetchall()

    return [FileRow.from_tuple(row) for row in result]


def list_files_by_format(
    conn: duckdb.DuckDBPyConnection, version_id: str, format: str
) -> List[FileRow]:
    """List files of specific format for a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID
        format: File format to filter

    Returns:
        List of FileRow objects
    """
    result = conn.execute(
        """
        SELECT f.file_id, f.artifact_id, f.relpath, f.size, f.format, f.sha256, f.mtime, f.extracted_at
        FROM extracted_files f
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        WHERE a.version_id = ? AND f.format = ?
        ORDER BY f.size DESC
    """,
        [version_id, format],
    ).fetchall()

    return [FileRow.from_tuple(row) for row in result]


def get_file(conn: duckdb.DuckDBPyConnection, file_id: str) -> Optional[FileRow]:
    """Get specific file by ID.

    Args:
        conn: DuckDB connection
        file_id: File ID

    Returns:
        FileRow or None if not found
    """
    result = conn.execute(
        """
        SELECT file_id, artifact_id, relpath, size, format, sha256, mtime, extracted_at
        FROM extracted_files
        WHERE file_id = ?
    """,
        [file_id],
    ).fetchone()

    return FileRow.from_tuple(result) if result else None


# ============================================================================
# VALIDATION QUERIES (VAL)
# ============================================================================


def list_validations(conn: duckdb.DuckDBPyConnection, version_id: str) -> List[ValidationRow]:
    """List all validations for a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID

    Returns:
        List of ValidationRow objects
    """
    result = conn.execute(
        """
        SELECT v.validation_id, v.file_id, v.validator, v.status, v.details, v.validated_at
        FROM validations v
        JOIN extracted_files f ON v.file_id = f.file_id
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        WHERE a.version_id = ?
        ORDER BY v.validated_at DESC
    """,
        [version_id],
    ).fetchall()

    return [ValidationRow.from_tuple(row) for row in result]


def list_validations_by_status(
    conn: duckdb.DuckDBPyConnection, version_id: str, status: str
) -> List[ValidationRow]:
    """List validations with specific status for a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID
        status: Validation status to filter

    Returns:
        List of ValidationRow objects
    """
    result = conn.execute(
        """
        SELECT v.validation_id, v.file_id, v.validator, v.status, v.details, v.validated_at
        FROM validations v
        JOIN extracted_files f ON v.file_id = f.file_id
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        WHERE a.version_id = ? AND v.status = ?
        ORDER BY v.validated_at DESC
    """,
        [version_id, status],
    ).fetchall()

    return [ValidationRow.from_tuple(row) for row in result]


def get_validation(conn: duckdb.DuckDBPyConnection, validation_id: str) -> Optional[ValidationRow]:
    """Get specific validation by ID.

    Args:
        conn: DuckDB connection
        validation_id: Validation ID

    Returns:
        ValidationRow or None if not found
    """
    result = conn.execute(
        """
        SELECT validation_id, file_id, validator, status, details, validated_at
        FROM validations
        WHERE validation_id = ?
    """,
        [validation_id],
    ).fetchone()

    return ValidationRow.from_tuple(result) if result else None


# ============================================================================
# STATISTICS QUERIES (STA)
# ============================================================================


def get_artifact_stats(conn: duckdb.DuckDBPyConnection, version_id: str) -> Dict[str, int]:
    """Get statistics for artifacts in a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID

    Returns:
        Dictionary with stats: total_artifacts, total_size, avg_size
    """
    result = conn.execute(
        """
        SELECT COUNT(*), SUM(size), AVG(size)
        FROM artifacts
        WHERE version_id = ?
    """,
        [version_id],
    ).fetchone()

    if result and result[0]:
        return {
            "total_artifacts": int(result[0]),
            "total_size": int(result[1]) if result[1] else 0,
            "avg_size": int(result[2]) if result[2] else 0,
        }
    return {"total_artifacts": 0, "total_size": 0, "avg_size": 0}


def get_file_stats(conn: duckdb.DuckDBPyConnection, version_id: str) -> Dict[str, int]:
    """Get statistics for files in a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID

    Returns:
        Dictionary with stats: total_files, total_size, avg_size
    """
    result = conn.execute(
        """
        SELECT COUNT(*), SUM(f.size), AVG(f.size)
        FROM extracted_files f
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        WHERE a.version_id = ?
    """,
        [version_id],
    ).fetchone()

    if result and result[0]:
        return {
            "total_files": int(result[0]),
            "total_size": int(result[1]) if result[1] else 0,
            "avg_size": int(result[2]) if result[2] else 0,
        }
    return {"total_files": 0, "total_size": 0, "avg_size": 0}


def get_validation_stats(conn: duckdb.DuckDBPyConnection, version_id: str) -> Dict[str, int]:
    """Get validation statistics for a version.

    Args:
        conn: DuckDB connection
        version_id: Version ID

    Returns:
        Dictionary with stats: total_validations, passed, failed, timeout
    """
    result = conn.execute(
        """
        SELECT
            COUNT(*),
            SUM(CASE WHEN v.status = 'pass' THEN 1 ELSE 0 END),
            SUM(CASE WHEN v.status = 'fail' THEN 1 ELSE 0 END),
            SUM(CASE WHEN v.status = 'timeout' THEN 1 ELSE 0 END)
        FROM validations v
        JOIN extracted_files f ON v.file_id = f.file_id
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        WHERE a.version_id = ?
    """,
        [version_id],
    ).fetchone()

    if result:
        return {
            "total_validations": int(result[0]) if result[0] else 0,
            "passed": int(result[1]) if result[1] else 0,
            "failed": int(result[2]) if result[2] else 0,
            "timeout": int(result[3]) if result[3] else 0,
        }
    return {"total_validations": 0, "passed": 0, "failed": 0, "timeout": 0}


# ============================================================================
# ORPHAN DETECTION QUERIES (ORP)
# ============================================================================


def detect_orphans(conn: duckdb.DuckDBPyConnection, fs_entries: List[tuple]) -> List[tuple]:
    """Detect files on disk not referenced in the database.

    Scans filesystem entries (relpath, size, mtime) and compares against
    database artifacts/files. Returns orphaned entries not tracked by DB.

    Args:
        conn: DuckDB connection (read-only safe)
        fs_entries: List of (relpath: str, size: int, mtime: float) tuples
                   from filesystem scan

    Returns:
        List of (relpath: str, size: int) tuples for orphaned files
    """
    if not fs_entries:
        return []

    # Create temporary table from filesystem entries
    conn.execute("CREATE TEMP TABLE fs_scan (relpath VARCHAR, size INTEGER, mtime DOUBLE)")

    try:
        # Insert filesystem entries
        for relpath, size, mtime in fs_entries:
            conn.execute(
                "INSERT INTO fs_scan (relpath, size, mtime) VALUES (?, ?, ?)",
                [relpath, size, mtime],
            )

        # Check if artifacts and extracted_files tables exist
        artifacts_exist = bool(
            conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'artifacts'"
            ).fetchone()
        )
        files_exist = bool(
            conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'extracted_files'"
            ).fetchone()
        )

        # Build query based on what tables exist
        if artifacts_exist and files_exist:
            query = """
                SELECT DISTINCT fs.relpath, fs.size
                FROM fs_scan fs
                WHERE NOT EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.fs_relpath = fs.relpath
                )
                AND NOT EXISTS (
                    SELECT 1 FROM extracted_files f
                    WHERE f.relpath_in_version = fs.relpath
                )
                ORDER BY fs.relpath ASC
            """
        elif artifacts_exist:
            query = """
                SELECT DISTINCT fs.relpath, fs.size
                FROM fs_scan fs
                WHERE NOT EXISTS (
                    SELECT 1 FROM artifacts a
                    WHERE a.fs_relpath = fs.relpath
                )
                ORDER BY fs.relpath ASC
            """
        elif files_exist:
            query = """
                SELECT DISTINCT fs.relpath, fs.size
                FROM fs_scan fs
                WHERE NOT EXISTS (
                    SELECT 1 FROM extracted_files f
                    WHERE f.relpath_in_version = fs.relpath
                )
                ORDER BY fs.relpath ASC
            """
        else:
            # No database tables yet, all filesystem entries are orphans
            return [(relpath, size) for relpath, size, _ in fs_entries]

        result = conn.execute(query).fetchall()
        return list(result)

    finally:
        # Clean up temporary table
        conn.execute("DROP TABLE IF EXISTS fs_scan")


# ============================================================================
# PLAN CACHING QUERIES (PLAN)
# ============================================================================


@dataclass(frozen=True)
class CachedPlanRow:
    """Cached plan metadata row."""

    plan_id: str
    service: str
    resolver: str
    url: str
    version_id: str
    checksum: Optional[str]
    cached_at: datetime

    @classmethod
    def from_tuple(cls, row: tuple) -> "CachedPlanRow":
        """Create from database tuple."""
        return cls(
            plan_id=row[0],
            service=row[1],
            resolver=row[2],
            url=row[3],
            version_id=row[4],
            checksum=row[5],
            cached_at=row[6] if isinstance(row[6], datetime) else datetime.fromisoformat(str(row[6])),
        )


def cache_plan(
    conn: duckdb.DuckDBPyConnection,
    plan_id: str,
    service: str,
    resolver: str,
    url: str,
    version_id: str,
    checksum: Optional[str] = None,
) -> None:
    """Cache a planning decision for later replay.

    Args:
        conn: DuckDB connection (write mode)
        plan_id: Unique identifier for this plan
        service: Service/ontology ID (e.g., 'hp', 'chebi')
        resolver: Resolver used (e.g., 'obo', 'ols')
        url: Source URL for the plan
        version_id: Version identifier (timestamp or semver)
        checksum: Optional checksum (SHA256 or similar)
    """
    # Check if plans table exists; create if not
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plans (
            plan_id VARCHAR PRIMARY KEY,
            service VARCHAR NOT NULL,
            resolver VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            version_id VARCHAR NOT NULL,
            checksum VARCHAR,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Upsert plan (replace if exists)
    conn.execute(
        """
        INSERT OR REPLACE INTO plans
        (plan_id, service, resolver, url, version_id, checksum, cached_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """,
        [plan_id, service, resolver, url, version_id, checksum],
    )


def get_cached_plan(
    conn: duckdb.DuckDBPyConnection, plan_id: str
) -> Optional[CachedPlanRow]:
    """Retrieve a cached plan by ID.

    Args:
        conn: DuckDB connection
        plan_id: Plan identifier

    Returns:
        CachedPlanRow or None if not found
    """
    # Check if plans table exists
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'plans'"
    ).fetchone()

    if not tables:
        return None

    result = conn.execute(
        """
        SELECT plan_id, service, resolver, url, version_id, checksum, cached_at
        FROM plans
        WHERE plan_id = ?
    """,
        [plan_id],
    ).fetchone()

    return CachedPlanRow.from_tuple(result) if result else None


def get_cached_plan_by_service(
    conn: duckdb.DuckDBPyConnection, service: str
) -> Optional[CachedPlanRow]:
    """Retrieve most recent cached plan for a service.

    Args:
        conn: DuckDB connection
        service: Service/ontology ID

    Returns:
        Most recent CachedPlanRow or None if not found
    """
    # Check if plans table exists
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'plans'"
    ).fetchone()

    if not tables:
        return None

    result = conn.execute(
        """
        SELECT plan_id, service, resolver, url, version_id, checksum, cached_at
        FROM plans
        WHERE service = ?
        ORDER BY cached_at DESC
        LIMIT 1
    """,
        [service],
    ).fetchone()

    return CachedPlanRow.from_tuple(result) if result else None
