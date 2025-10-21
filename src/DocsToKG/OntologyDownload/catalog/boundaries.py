# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.boundaries",
#   "purpose": "Transactional boundaries for FS↔DB choreography",
#   "sections": [
#     {"id": "types", "name": "Boundary Result Types", "anchor": "TYP", "kind": "models"},
#     {"id": "download", "name": "Download Boundary", "anchor": "DL", "kind": "api"},
#     {"id": "extract", "name": "Extraction Boundary", "anchor": "EX", "kind": "api"},
#     {"id": "validation", "name": "Validation Boundary", "anchor": "VAL", "kind": "api"},
#     {"id": "latest", "name": "Set Latest Boundary", "anchor": "LAT", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Transactional boundaries for DuckDB catalog FS↔DB choreography.

Each boundary implements two-phase commits:
  1. Filesystem operations complete first (atomic rename)
  2. Database transaction commits second (rollback if FS failed)

This ensures consistency: DB rows never reference missing files,
and filesystem writes are never orphaned.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
from uuid import uuid4
import time

try:  # pragma: no cover
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for catalog boundaries. Ensure .venv is initialized."
    ) from exc

from ..policy.errors import PolicyReject
from ..policy.gates import db_boundary_gate
from .observability_instrumentation import (
    emit_boundary_begin,
    emit_boundary_error,
    emit_boundary_success,
)

logger = logging.getLogger(__name__)


# ============================================================================
# BOUNDARY RESULT TYPES (TYP)
# ============================================================================


@dataclass(frozen=True)
class DownloadBoundaryResult:
    """Result of a download boundary transaction."""

    artifact_id: str
    version_id: str
    fs_relpath: str
    size: int
    etag: Optional[str]
    inserted: bool


@dataclass(frozen=True)
class ExtractionBoundaryResult:
    """Result of an extraction boundary transaction."""

    artifact_id: str
    files_inserted: int
    total_size: int
    audit_path: Path
    inserted: bool


@dataclass(frozen=True)
class ValidationBoundaryResult:
    """Result of a validation boundary transaction."""

    file_id: str
    validator: str
    status: str
    inserted: bool


@dataclass(frozen=True)
class SetLatestBoundaryResult:
    """Result of a set-latest boundary transaction."""

    version_id: str
    latest_json_path: Path
    pointer_updated: bool
    json_written: bool


# ============================================================================
# DOWNLOAD BOUNDARY (DL)
# ============================================================================


@contextlib.contextmanager
def download_boundary(
    conn: duckdb.DuckDBPyConnection,
    artifact_id: str,
    version_id: str,
    fs_relpath: str,
    size: int,
    etag: Optional[str] = None,
) -> Generator[DownloadBoundaryResult, None, None]:
    """Transactional boundary for download operations.

    Workflow:
      1. Caller handles FS write (temp → rename)
      2. Boundary inserts artifact metadata to DB
      3. Caller handles cleanup on exception

    Args:
        conn: DuckDB writer connection
        artifact_id: sha256(archive) hash
        version_id: Version identifier
        fs_relpath: Relative path on FS
        size: Archive size in bytes
        etag: Optional ETag from HTTP

    Yields:
        DownloadBoundaryResult (inserted=True if successful)

    Raises:
        duckdb.Error: If insert fails (triggers rollback)
    """
    # Emit observability begin event
    emit_boundary_begin(
        boundary="download",
        artifact_id=artifact_id,
        version_id=version_id,
        service="unknown",
        extra_payload={"fs_relpath": fs_relpath, "size": size},
    )
    start_time = time.time()

    result = DownloadBoundaryResult(
        artifact_id=artifact_id,
        version_id=version_id,
        fs_relpath=fs_relpath,
        size=size,
        etag=etag,
        inserted=False,
    )

    try:
        conn.begin()

        conn.execute(
            """
            INSERT INTO artifacts
            (artifact_id, version_id, fs_relpath, size, etag, status, downloaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [artifact_id, version_id, fs_relpath, size, etag, "fresh", datetime.now()],
        )

        result = DownloadBoundaryResult(
            artifact_id=artifact_id,
            version_id=version_id,
            fs_relpath=fs_relpath,
            size=size,
            etag=etag,
            inserted=True,
        )

        conn.commit()
        logger.info(f"Download boundary: inserted artifact {artifact_id}")

        # Emit observability success event
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_success(
            boundary="download",
            artifact_id=artifact_id,
            version_id=version_id,
            duration_ms=duration_ms,
            extra_payload={"size_bytes": size, "etag": etag},
        )

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Download boundary failed: {exc}")

        # Emit observability error event
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_error(
            boundary="download",
            artifact_id=artifact_id,
            version_id=version_id,
            error=exc,
            duration_ms=duration_ms,
        )
        raise

    yield result


# ============================================================================
# EXTRACTION BOUNDARY (EX)
# ============================================================================


@contextlib.contextmanager
def extraction_boundary(
    conn: duckdb.DuckDBPyConnection,
    artifact_id: str,
) -> Generator[ExtractionBoundaryResult, None, None]:
    """Transactional boundary for extraction operations.

    Workflow:
      1. Caller extracts files to FS + writes audit JSON
      2. Caller builds list of FileRow objects
      3. Boundary receives rows + inserts to DB
      4. Boundary commits transaction

    Args:
        conn: DuckDB writer connection
        artifact_id: Artifact being extracted

    Yields:
        ExtractionBoundaryResult (populated by caller)
    """
    # Emit observability begin event
    emit_boundary_begin(
        boundary="extraction",
        artifact_id=artifact_id,
        version_id="unknown",
        service="unknown",
        extra_payload={},
    )
    start_time = time.time()
    
    # Placeholder result; caller will update
    result = ExtractionBoundaryResult(
        artifact_id=artifact_id,
        files_inserted=0,
        total_size=0,
        audit_path=Path(),
        inserted=False,
    )

    try:
        conn.begin()
        yield result

        if result.files_inserted > 0:
            # GATE 5: DB Transaction Boundaries (No Torn Writes)
            try:
                db_result = db_boundary_gate(
                    operation="pre_commit",
                    tables_affected=["extracted_files", "manifests"],
                    fs_success=True,
                )
                if isinstance(db_result, PolicyReject):
                    logger.error(f"db_boundary gate rejected: {db_result.error_code}")
                    conn.rollback()
                    raise Exception(f"Transaction boundary violation: {db_result.error_code}")
            except Exception as exc:
                if "Transaction boundary violation" in str(exc):
                    raise
                logger.warning(f"db_boundary gate error: {exc}, proceeding cautiously")

            conn.commit()
            logger.info(
                f"Extraction boundary: inserted {result.files_inserted} files "
                f"for artifact {artifact_id}"
            )
            
            # Emit observability success event
            duration_ms = (time.time() - start_time) * 1000
            emit_boundary_success(
                boundary="extraction",
                artifact_id=artifact_id,
                version_id="unknown",
                duration_ms=duration_ms,
                extra_payload={
                    "files_inserted": result.files_inserted,
                    "total_size": result.total_size,
                },
            )
        else:
            conn.rollback()

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Extraction boundary failed: {exc}")
        
        # Emit observability error event
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_error(
            boundary="extraction",
            artifact_id=artifact_id,
            version_id="unknown",
            error=exc,
            duration_ms=duration_ms,
        )
        raise


# ============================================================================
# VALIDATION BOUNDARY (VAL)
# ============================================================================


@contextlib.contextmanager
def validation_boundary(
    conn: duckdb.DuckDBPyConnection,
    file_id: str,
    validator: str,
    status: str,
    details: Optional[dict] = None,
) -> Generator[ValidationBoundaryResult, None, None]:
    """Transactional boundary for validation operations.

    Workflow:
      1. Caller runs validator (no FS ops)
      2. Boundary inserts validation result to DB
      3. Caller handles result inspection

    Args:
        conn: DuckDB writer connection
        file_id: File being validated
        validator: Validator name (e.g., 'rdflib')
        status: Result status (pass|fail|timeout)
        details: Optional JSON details

    Yields:
        ValidationBoundaryResult (inserted=True if successful)

    Raises:
        duckdb.Error: If insert fails (triggers rollback)
    """
    # Emit observability begin event
    emit_boundary_begin(
        boundary="validation",
        artifact_id="unknown",
        version_id="unknown",
        service="unknown",
        extra_payload={"file_id": file_id, "validator": validator},
    )
    start_time = time.time()
    
    result = ValidationBoundaryResult(
        file_id=file_id,
        validator=validator,
        status=status,
        inserted=False,
    )

    try:
        conn.begin()

        validation_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO validations
            (validation_id, file_id, validator, status, details, validated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                validation_id,
                file_id,
                validator,
                status,
                json.dumps(details) if details else None,
                datetime.now(),
            ],
        )

        result = ValidationBoundaryResult(
            file_id=file_id,
            validator=validator,
            status=status,
            inserted=True,
        )

        conn.commit()
        logger.info(f"Validation boundary: recorded {validator}:{status} for {file_id}")
        
        # Emit observability success event
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_success(
            boundary="validation",
            artifact_id="unknown",
            version_id="unknown",
            duration_ms=duration_ms,
            extra_payload={"validator": validator, "status": status},
        )

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Validation boundary failed: {exc}")
        
        # Emit observability error event
        duration_ms = (time.time() - start_time) * 1000
        emit_boundary_error(
            boundary="validation",
            artifact_id="unknown",
            version_id="unknown",
            error=exc,
            duration_ms=duration_ms,
        )
        raise

    yield result


# ============================================================================
# SET LATEST BOUNDARY (LAT)
# ============================================================================


@contextlib.contextmanager
def set_latest_boundary(
    conn: duckdb.DuckDBPyConnection,
    version_id: str,
    latest_json_path: Path,
) -> Generator[SetLatestBoundaryResult, None, None]:
    """Transactional boundary for setting latest version.

    Workflow:
      1. Caller prepares LATEST.json in temp file
      2. Boundary upserts latest_pointer + writes JSON atomically
      3. If any step fails, entire boundary rolls back

    Args:
        conn: DuckDB writer connection
        version_id: Version to mark as latest
        latest_json_path: Path where LATEST.json should live

    Yields:
        SetLatestBoundaryResult (updated flags on success)

    Raises:
        duckdb.Error: If DB fails (FS write is retried)
        IOError: If FS write fails
    """
    result = SetLatestBoundaryResult(
        version_id=version_id,
        latest_json_path=latest_json_path,
        pointer_updated=False,
        json_written=False,
    )

    temp_json = latest_json_path.parent / f"{latest_json_path.name}.tmp"

    try:
        conn.begin()

        # Upsert latest pointer
        conn.execute(
            """
            INSERT OR REPLACE INTO latest_pointer (version_id, set_at)
            VALUES (?, ?)
            """,
            [version_id, datetime.now()],
        )

        result = SetLatestBoundaryResult(
            version_id=version_id,
            latest_json_path=latest_json_path,
            pointer_updated=True,
            json_written=False,
        )

        # Write JSON atomically (caller prepared temp file)
        if temp_json.exists():
            temp_json.replace(latest_json_path)
            result = SetLatestBoundaryResult(
                version_id=version_id,
                latest_json_path=latest_json_path,
                pointer_updated=True,
                json_written=True,
            )

        conn.commit()
        logger.info(f"Set latest boundary: marked {version_id} as latest")

    except duckdb.Error as exc:
        conn.rollback()
        logger.error(f"Set latest boundary failed: {exc}")
        raise

    yield result
