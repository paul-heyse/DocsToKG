# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.doctor",
#   "purpose": "Database health checks and FS↔DB reconciliation",
#   "sections": [
#     {"id": "types", "name": "Doctor Result Types", "anchor": "TYP", "kind": "models"},
#     {"id": "scan", "name": "Filesystem Scanning", "anchor": "SCAN", "kind": "api"},
#     {"id": "check", "name": "Health Checks", "anchor": "CHK", "kind": "api"},
#     {"id": "detect", "name": "Drift Detection", "anchor": "DFT", "kind": "api"},
#     {"id": "repair", "name": "Auto-Repair Actions", "anchor": "RPR", "kind": "api"}
#   ]
# }
# === /NAVMAP ===

"""Doctor module for DuckDB catalog health and reconciliation.

Responsibilities:
- Scan filesystem for artifacts and files
- Detect DB↔FS mismatches (missing rows, orphaned files)
- Suggest or auto-repair drifts
- Report health metrics
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:  # pragma: no cover
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for catalog doctor. Ensure .venv is initialized."
    ) from exc

from .observability_instrumentation import emit_doctor_begin, emit_doctor_complete, emit_doctor_issue_found

logger = logging.getLogger(__name__)


# ============================================================================
# DOCTOR RESULT TYPES (TYP)
# ============================================================================


@dataclass(frozen=True)
class DoctorIssue:
    """A detected DB↔FS mismatch."""

    issue_type: str  # 'missing_db_row' | 'missing_fs_file' | 'latest_mismatch'
    artifact_id: Optional[str]
    file_id: Optional[str]
    fs_path: Optional[Path]
    description: str
    severity: str  # 'info' | 'warning' | 'error'
    size_bytes: Optional[int] = None


@dataclass(frozen=True)
class DoctorReport:
    """Doctor health check report."""

    timestamp: datetime
    total_artifacts: int
    total_files: int
    fs_artifacts_scanned: int
    fs_files_scanned: int
    issues_found: int
    critical_issues: int
    warnings: int
    issues: list[DoctorIssue]


@dataclass(frozen=True)
class HealthCheckResult:
    """Quick health check result."""

    is_healthy: bool
    message: str
    artifact_count: int
    file_count: int
    database_size_mb: float


# ============================================================================
# HEALTH CHECKS (CHK)
# ============================================================================


def quick_health_check(
    conn: duckdb.DuckDBPyConnection,
) -> HealthCheckResult:
    """Run a quick non-invasive health check.

    Checks:
    - Schema version exists and > 0
    - Can query key tables
    - Basic counts

    Args:
        conn: DuckDB reader connection

    Returns:
        HealthCheckResult with status

    Raises:
        duckdb.Error: If queries fail
    """
    try:
        # Get schema version (count of applied migrations)
        schema_count_result = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        schema_count = schema_count_result[0] if schema_count_result else 0
        schema_ok = schema_count > 0

        # Get counts
        artifact_result = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()
        file_result = conn.execute("SELECT COUNT(*) FROM extracted_files").fetchone()

        artifact_count = artifact_result[0] if artifact_result else 0
        file_count = file_result[0] if file_result else 0

        # Estimate DB size (rough)
        db_size_mb = 0.0

        is_healthy = schema_ok and (artifact_count >= 0) and (file_count >= 0)
        message = (
            f"OK: schema v{schema_count} (migrations applied), "
            f"{artifact_count} artifacts, {file_count} files"
            if is_healthy
            else "UNHEALTHY: schema missing or tables inaccessible"
        )

        return HealthCheckResult(
            is_healthy=is_healthy,
            message=message,
            artifact_count=artifact_count,
            file_count=file_count,
            database_size_mb=db_size_mb,
        )

    except duckdb.Error as exc:
        logger.error(f"Health check failed: {exc}")
        raise


# ============================================================================
# FILESYSTEM SCANNING (SCAN)
# ============================================================================


def scan_filesystem_artifacts(
    artifacts_root: Path,
) -> list[tuple[Path, int]]:
    """Scan filesystem for artifact files.

    Args:
        artifacts_root: Root directory containing artifacts

    Returns:
        List of (path, size) tuples for each file found
    """
    if not artifacts_root.exists():
        logger.warning(f"Artifacts root does not exist: {artifacts_root}")
        return []

    artifacts = []
    for artifact_path in artifacts_root.rglob("*.zip"):
        if artifact_path.is_file():
            artifacts.append((artifact_path, artifact_path.stat().st_size))

    logger.info(f"Scanned {len(artifacts)} artifacts from {artifacts_root}")
    return artifacts


def scan_filesystem_files(
    extracted_root: Path,
) -> list[tuple[Path, int]]:
    """Scan filesystem for extracted files.

    Args:
        extracted_root: Root directory containing extracted files

    Returns:
        List of (path, size) tuples for each file found
    """
    if not extracted_root.exists():
        logger.warning(f"Extracted root does not exist: {extracted_root}")
        return []

    files = []
    for file_path in extracted_root.rglob("*"):
        if file_path.is_file():
            files.append((file_path, file_path.stat().st_size))

    logger.info(f"Scanned {len(files)} extracted files from {extracted_root}")
    return files


# ============================================================================
# DRIFT DETECTION (DFT)
# ============================================================================


def detect_db_fs_drifts(
    conn: duckdb.DuckDBPyConnection,
    artifacts_root: Path,
    extracted_root: Path,
    *,
    fs_artifacts: Optional[Iterable[tuple[Path, int]]] = None,
    fs_files: Optional[Iterable[tuple[Path, int]]] = None,
) -> list[DoctorIssue]:
    """Detect mismatches between DB and filesystem.

    Checks:
    - DB artifacts without FS files
    - FS files without DB entries
    - Latest pointer consistency

    Args:
        conn: DuckDB reader connection
        artifacts_root: Root for artifact files
        extracted_root: Root for extracted files

    Returns:
        List of DoctorIssue objects
    """
    issues: list[DoctorIssue] = []

    artifacts_root = artifacts_root.resolve()
    extracted_root = extracted_root.resolve()

    fs_artifacts_list = list(fs_artifacts or scan_filesystem_artifacts(artifacts_root))
    fs_files_list = list(fs_files or scan_filesystem_files(extracted_root))

    fs_artifact_map = {
        path.resolve().relative_to(artifacts_root).as_posix(): size for path, size in fs_artifacts_list
    }
    fs_file_map = {
        path.resolve().relative_to(extracted_root).as_posix(): size for path, size in fs_files_list
    }

    # Artifacts in DB
    db_artifacts = conn.execute(
        "SELECT artifact_id, fs_relpath, size_bytes FROM artifacts"
    ).fetchall()
    db_artifact_map = {row[1]: (row[0], row[2]) for row in db_artifacts if row[1]}

    for artifact_id, fs_relpath, size_bytes in db_artifacts:
        if not fs_relpath:
            continue
        expected_path = artifacts_root / fs_relpath
        if not expected_path.exists():
            description = (
                f"Artifact {artifact_id} recorded in DB but missing on FS: {fs_relpath}"
            )
            issue = DoctorIssue(
                issue_type="missing_fs_artifact",
                artifact_id=artifact_id,
                file_id=None,
                fs_path=expected_path,
                description=description,
                severity="error",
                size_bytes=size_bytes,
            )
            issues.append(issue)
            emit_doctor_issue_found(
                issue_type="missing_fs_artifact",
                severity="error",
                description=description,
                extra={"artifact_id": artifact_id, "path": fs_relpath},
            )

    for relpath, size_bytes in fs_artifact_map.items():
        if relpath not in db_artifact_map:
            description = f"File present on FS but not in catalog: {relpath}"
            issue = DoctorIssue(
                issue_type="orphan_artifact_file",
                artifact_id=None,
                file_id=None,
                fs_path=artifacts_root / relpath,
                description=description,
                severity="warning",
                size_bytes=size_bytes,
            )
            issues.append(issue)
            emit_doctor_issue_found(
                issue_type="orphan_artifact_file",
                severity="warning",
                description=description,
                extra={"path": relpath},
            )

    # Extracted files in DB with service for path reconstruction
    db_files = conn.execute(
        """
        SELECT f.file_id, f.relpath_in_version, f.size_bytes, a.service, f.version_id
        FROM extracted_files f
        JOIN artifacts a ON f.artifact_id = a.artifact_id
        """
    ).fetchall()

    db_file_map: dict[str, tuple[str, int]] = {}
    for file_id, relpath_in_version, size_bytes, service, version_id in db_files:
        rel = Path(service) / version_id / relpath_in_version
        rel_posix = rel.as_posix()
        db_file_map[rel_posix] = (file_id, size_bytes)
        expected_path = extracted_root / rel
        if not expected_path.exists():
            description = (
                f"Extracted file {file_id} (service={service}, version={version_id}) missing on FS: {rel_posix}"
            )
            issue = DoctorIssue(
                issue_type="missing_fs_extracted_file",
                artifact_id=None,
                file_id=file_id,
                fs_path=expected_path,
                description=description,
                severity="error",
                size_bytes=size_bytes,
            )
            issues.append(issue)
            emit_doctor_issue_found(
                issue_type="missing_fs_extracted_file",
                severity="error",
                description=description,
                extra={"file_id": file_id, "path": rel_posix},
            )

    for relpath, size_bytes in fs_file_map.items():
        if relpath not in db_file_map:
            description = f"Extracted file present on FS but not in catalog: {relpath}"
            issue = DoctorIssue(
                issue_type="orphan_extracted_file",
                artifact_id=None,
                file_id=None,
                fs_path=extracted_root / relpath,
                description=description,
                severity="warning",
                size_bytes=size_bytes,
            )
            issues.append(issue)
            emit_doctor_issue_found(
                issue_type="orphan_extracted_file",
                severity="warning",
                description=description,
                extra={"path": relpath},
            )

    # Latest pointer consistency
    latest_result = conn.execute("SELECT version_id FROM latest_pointer LIMIT 1").fetchone()
    if latest_result:
        db_latest_version = latest_result[0]
        latest_json_path = artifacts_root.parent / "LATEST.json"
        if not latest_json_path.exists():
            description = (
                f"DB marks {db_latest_version} as latest, but LATEST.json missing on FS"
            )
            issue = DoctorIssue(
                issue_type="latest_mismatch",
                artifact_id=None,
                file_id=None,
                fs_path=latest_json_path,
                description=description,
                severity="warning",
            )
            issues.append(issue)
            emit_doctor_issue_found(
                issue_type="latest_mismatch",
                severity="warning",
                description=description,
                extra={"expected_latest": db_latest_version},
            )
        else:
            try:
                payload = json.loads(latest_json_path.read_text(encoding="utf-8"))
                fs_latest = payload.get("latest") or payload.get("version")
                if fs_latest and fs_latest != db_latest_version:
                    description = (
                        f"LATEST.json points to {fs_latest}, but catalog latest is {db_latest_version}"
                    )
                    issue = DoctorIssue(
                        issue_type="latest_mismatch",
                        artifact_id=None,
                        file_id=None,
                        fs_path=latest_json_path,
                        description=description,
                        severity="warning",
                    )
                    issues.append(issue)
                    emit_doctor_issue_found(
                        issue_type="latest_mismatch",
                        severity="warning",
                        description=description,
                        extra={"expected_latest": db_latest_version, "fs_latest": fs_latest},
                    )
            except json.JSONDecodeError:
                description = "LATEST.json could not be parsed"
                issue = DoctorIssue(
                    issue_type="latest_invalid",
                    artifact_id=None,
                    file_id=None,
                    fs_path=latest_json_path,
                    description=description,
                    severity="warning",
                )
                issues.append(issue)
                emit_doctor_issue_found(
                    issue_type="latest_invalid",
                    severity="warning",
                    description=description,
                )

    logger.info("Detected %s DB↔FS drift issues", len(issues))
    return issues


def generate_doctor_report(
    conn: duckdb.DuckDBPyConnection,
    artifacts_root: Path,
    extracted_root: Path,
) -> DoctorReport:
    """Generate comprehensive doctor report.

    Args:
        conn: DuckDB reader connection
        artifacts_root: Root for artifact files
        extracted_root: Root for extracted files

    Returns:
        DoctorReport with full reconciliation details
    """
    # Emit observability begin event
    emit_doctor_begin()
    start_time = time.time()

    # Get DB counts
    artifact_count_result = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()
    file_count_result = conn.execute("SELECT COUNT(*) FROM extracted_files").fetchone()

    artifact_count = artifact_count_result[0] if artifact_count_result else 0
    file_count = file_count_result[0] if file_count_result else 0

    # Scan FS
    fs_artifacts = scan_filesystem_artifacts(artifacts_root)
    fs_files = scan_filesystem_files(extracted_root)

    # Detect drifts
    issues = detect_db_fs_drifts(
        conn,
        artifacts_root,
        extracted_root,
        fs_artifacts=fs_artifacts,
        fs_files=fs_files,
    )

    # Categorize issues
    critical = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")

    report = DoctorReport(
        timestamp=datetime.now(),
        total_artifacts=artifact_count,
        total_files=file_count,
        fs_artifacts_scanned=len(fs_artifacts),
        fs_files_scanned=len(fs_files),
        issues_found=len(issues),
        critical_issues=critical,
        warnings=warnings,
        issues=issues,
    )

    logger.info(f"Doctor report: {len(issues)} issues ({critical} critical, {warnings} warnings)")

    # Emit observability complete event
    duration_ms = (time.time() - start_time) * 1000
    emit_doctor_complete(
        issues_found=len(issues),
        critical=critical,
        warnings=warnings,
        duration_ms=duration_ms,
    )

    return report
