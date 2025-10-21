"""Data Transfer Objects for catalog queries.

Type-safe result shapes for query responses. All DTOs are frozen dataclasses
to ensure immutability and hashability.

NAVMAP:
  - VersionStats: Comprehensive version metrics
  - VersionRow: Single version information
  - FileRow: File information row
  - ValidationResult: Single validation outcome
  - ValidationSummary: Aggregated validation metrics
  - ArtifactInfo: Artifact metadata
  - VersionDelta: Version-to-version comparison
  - StorageUsage: Disk usage breakdown
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class VersionStats:
    """Comprehensive metrics for a single version.

    Attributes:
        version_id: Unique version identifier
        service: Service name (e.g., "openalex")
        created_at: When version was created
        file_count: Number of extracted files
        total_size: Total bytes across all files
        validation_passed: Count of passing validations
        validation_failed: Count of failing validations
        artifacts_count: Number of artifacts in version
        last_accessed: When version was last accessed
    """

    version_id: str
    service: str
    created_at: datetime
    file_count: int
    total_size: int
    validation_passed: int
    validation_failed: int
    artifacts_count: int
    last_accessed: Optional[datetime] = None

    @property
    def validation_passed_pct(self) -> float:
        """Calculate percentage of validations passed."""
        total = self.validation_passed + self.validation_failed
        if total == 0:
            return 100.0
        return (self.validation_passed / total) * 100.0


@dataclass(frozen=True)
class VersionRow:
    """Single row from list_versions query.

    Attributes:
        version_id: Unique version identifier
        service: Service name
        created_at: Creation timestamp
        plan_hash: Hash of the plan used
    """

    version_id: str
    service: str
    created_at: datetime
    plan_hash: Optional[str] = None


@dataclass(frozen=True)
class FileRow:
    """Single row from list_files query.

    Attributes:
        file_id: Unique file identifier
        relpath_in_version: Path relative to version
        size_bytes: File size in bytes
        format: File format (ttl, rdf, owl, obo, etc.)
        mtime: Last modified time
    """

    file_id: str
    relpath_in_version: str
    size_bytes: int
    format: Optional[str] = None
    mtime: Optional[datetime] = None


@dataclass(frozen=True)
class ValidationResult:
    """Single validation result row.

    Attributes:
        validation_id: Unique validation identifier
        file_id: File being validated
        validator: Validator name (rdflib, pronto, etc.)
        passed: Whether validation passed
        details: Validation details as JSON string
        run_at: When validation was run
    """

    validation_id: str
    file_id: str
    validator: str
    passed: bool
    details: Optional[str] = None
    run_at: Optional[datetime] = None


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregated validation metrics.

    Attributes:
        total_validations: Total validation runs
        passed_count: Validations that passed
        failed_count: Validations that failed
        by_validator: Per-validator breakdown
        pass_rate_pct: Overall pass rate percentage
    """

    total_validations: int
    passed_count: int
    failed_count: int
    by_validator: dict[str, dict[str, int]]  # {validator: {passed: N, failed: N}}
    pass_rate_pct: float


@dataclass(frozen=True)
class ArtifactInfo:
    """Artifact metadata from find_by_artifact_id.

    Attributes:
        artifact_id: Unique artifact identifier
        version_id: Version containing artifact
        service: Service name
        source_url: Original source URL
        size_bytes: Artifact size
        etag: Entity tag for cache validation
        status: Artifact status (fresh, cached, failed)
    """

    artifact_id: str
    version_id: str
    service: str
    source_url: str
    size_bytes: int
    etag: Optional[str] = None
    status: str = "fresh"


@dataclass(frozen=True)
class VersionDelta:
    """Comparison between two versions.

    Attributes:
        version_a: First version ID
        version_b: Second version ID
        files_added: Files only in B
        files_removed: Files only in A
        files_common: Files in both
        format_changes: Files with format changes
        size_delta_bytes: Net size change
    """

    version_a: str
    version_b: str
    files_added: list[str]
    files_removed: list[str]
    files_common: list[str]
    format_changes: dict[str, tuple[Optional[str], Optional[str]]]
    size_delta_bytes: int

    @property
    def total_changes(self) -> int:
        """Calculate total number of changes."""
        return len(self.files_added) + len(self.files_removed) + len(self.format_changes)


@dataclass(frozen=True)
class StorageUsage:
    """Disk usage breakdown across formats.

    Attributes:
        total_bytes: Total bytes used
        by_format: Per-format breakdown
        by_version: Per-version breakdown
        file_count: Total files
        avg_file_size: Average file size
    """

    total_bytes: int
    by_format: dict[str, int]  # {format: bytes}
    by_version: dict[str, int]  # {version_id: bytes}
    file_count: int
    avg_file_size: float

    @property
    def total_mb(self) -> float:
        """Get total storage in MB."""
        return self.total_bytes / (1024 * 1024)

    @property
    def total_gb(self) -> float:
        """Get total storage in GB."""
        return self.total_bytes / (1024 * 1024 * 1024)
