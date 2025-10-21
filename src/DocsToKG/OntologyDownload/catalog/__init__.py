"""DuckDB catalog package for OntologyDownload.

Core modules:
- migrations: Idempotent schema runner
- queries: Type-safe query façades with DTOs
- boundaries: Transactional context managers (FS↔DB choreography)
- doctor: Health checks and drift detection
"""

from __future__ import annotations

# Lazy imports to avoid circular dependencies
__all__ = [
    "Database",
    "DatabaseConfiguration",
    "VersionRow",
    "ArtifactRow",
    "FileRow",
    "ValidationRow",
    "DownloadBoundaryResult",
    "ExtractionBoundaryResult",
    "ValidationBoundaryResult",
    "SetLatestBoundaryResult",
    "download_boundary",
    "extraction_boundary",
    "validation_boundary",
    "set_latest_boundary",
    "DoctorIssue",
    "DoctorReport",
    "HealthCheckResult",
    "quick_health_check",
    "detect_db_fs_drifts",
    "generate_doctor_report",
]
