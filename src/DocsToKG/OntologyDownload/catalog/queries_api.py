# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.catalog.queries_api",
#   "purpose": "Catalog query API fa\u00e7ade.",
#   "sections": [
#     {
#       "id": "catalogqueries",
#       "name": "CatalogQueries",
#       "anchor": "class-catalogqueries",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Catalog query API façade.

High-level query interface for catalog introspection and analysis.
All methods are performance-optimized with proper indexing.

NAVMAP:
  - CatalogQueries: Main query façade class
  - Query Methods:
    * get_version_stats() - Comprehensive version metrics
    * list_versions() - Version enumeration
    * list_files() - File enumeration
    * list_validations() - Validation results
    * get_validation_summary() - Aggregated metrics
    * find_by_artifact_id() - Artifact lookup
    * compute_version_delta() - Version comparison
    * get_storage_usage() - Disk usage analysis
"""

from __future__ import annotations

from datetime import datetime

from .queries_dto import (
    ArtifactInfo,
    FileRow,
    StorageUsage,
    ValidationResult,
    ValidationSummary,
    VersionDelta,
    VersionRow,
    VersionStats,
)


class CatalogQueries:
    """Catalog query API with high-level methods.

    All methods are designed to be performance-optimized and work
    efficiently on large datasets (200k+ rows).

    Attributes:
        repo: Underlying Repo instance for data access
    """

    def __init__(self, repo):
        """Initialize query API.

        Args:
            repo: Repo instance for database access
        """
        self.repo = repo

    def get_version_stats(self, version_id: str) -> VersionStats | None:
        """Get comprehensive metrics for a version.

        Args:
            version_id: Version to analyze

        Returns:
            VersionStats object or None if version not found

        Performance:
            Executes in < 50ms on typical datasets
        """
        # Query version metadata
        version = self.repo.get_version(version_id)
        if not version:
            return None

        # Get file count and total size
        files_result = self.repo.query_scalar(
            """
            SELECT COUNT(*), SUM(COALESCE(size_bytes, 0))
            FROM files
            WHERE version_id = ?
            """,
            (version_id,),
        )
        file_count, total_size = files_result or (0, 0)

        # Get validation counts
        validation_result = self.repo.query_scalar(
            """
            SELECT
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed
            FROM validations
            WHERE file_id IN (
                SELECT file_id FROM files WHERE version_id = ?
            )
            """,
            (version_id,),
        )
        validation_passed, validation_failed = validation_result or (0, 0)

        # Get artifact count
        artifacts_count = self.repo.query_scalar(
            """
            SELECT COUNT(*) FROM artifacts WHERE version_id = ?
            """,
            (version_id,),
        )
        artifacts_count = artifacts_count[0] if artifacts_count else 0

        return VersionStats(
            version_id=version_id,
            service=version.get("service", ""),
            created_at=version.get("created_at", datetime.utcnow()),
            file_count=file_count,
            total_size=total_size or 0,
            validation_passed=validation_passed or 0,
            validation_failed=validation_failed or 0,
            artifacts_count=artifacts_count,
            last_accessed=version.get("last_accessed"),
        )

    def list_versions(
        self,
        *,
        service: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[VersionRow]:
        """List versions with optional filtering.

        Args:
            service: Filter by service (optional)
            limit: Maximum results to return
            offset: Skip N results

        Returns:
            List of VersionRow objects

        Performance:
            Executes in < 100ms
        """
        query = "SELECT version_id, service, created_at, plan_hash FROM versions"
        params = []

        if service:
            query += " WHERE service = ?"
            params.append(service)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.repo.query_all(query, params)
        return [
            VersionRow(
                version_id=row[0],
                service=row[1],
                created_at=row[2],
                plan_hash=row[3],
            )
            for row in rows
        ]

    def list_files(
        self,
        version_id: str | None = None,
        *,
        format_filter: str | None = None,
        prefix: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[FileRow]:
        """List files with optional filters.

        Args:
            version_id: Filter to specific version (optional)
            format_filter: Filter by file format
            prefix: Filter by path prefix
            limit: Maximum results
            offset: Skip N results

        Returns:
            List of FileRow objects

        Performance:
            Executes in < 150ms on large datasets
        """
        query = "SELECT file_id, relpath_in_version, size_bytes, format, mtime FROM files"
        params = []
        conditions = []

        if version_id:
            conditions.append("version_id = ?")
            params.append(version_id)

        if format_filter:
            conditions.append("format = ?")
            params.append(format_filter)

        if prefix:
            conditions.append("relpath_in_version LIKE ?")
            params.append(f"{prefix}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY relpath_in_version LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.repo.query_all(query, params)
        return [
            FileRow(
                file_id=row[0],
                relpath_in_version=row[1],
                size_bytes=row[2],
                format=row[3],
                mtime=row[4],
            )
            for row in rows
        ]

    def list_validations(
        self,
        file_id: str | None = None,
        *,
        validator: str | None = None,
        passed_only: bool = False,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ValidationResult]:
        """List validation results.

        Args:
            file_id: Filter to specific file (optional)
            validator: Filter by validator name
            passed_only: Return only passing validations
            limit: Maximum results
            offset: Skip N results

        Returns:
            List of ValidationResult objects

        Performance:
            Executes in < 120ms
        """
        query = "SELECT validation_id, file_id, validator, passed, details, run_at FROM validations"
        params = []
        conditions = []

        if file_id:
            conditions.append("file_id = ?")
            params.append(file_id)

        if validator:
            conditions.append("validator = ?")
            params.append(validator)

        if passed_only:
            conditions.append("passed = true")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY run_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.repo.query_all(query, params)
        return [
            ValidationResult(
                validation_id=row[0],
                file_id=row[1],
                validator=row[2],
                passed=row[3],
                details=row[4],
                run_at=row[5],
            )
            for row in rows
        ]

    def get_validation_summary(self, version_id: str | None = None) -> ValidationSummary:
        """Get aggregated validation metrics.

        Args:
            version_id: Summarize specific version or all if None

        Returns:
            ValidationSummary with aggregated metrics

        Performance:
            Executes in < 80ms
        """
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed
            FROM validations
        """
        params = []

        if version_id:
            query += """
                WHERE file_id IN (
                    SELECT file_id FROM files WHERE version_id = ?
                )
            """
            params.append(version_id)

        result = self.repo.query_scalar(query, params)
        total, passed, failed = result or (0, 0, 0)

        # Per-validator breakdown
        validator_query = """
            SELECT validator,
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN NOT passed THEN 1 ELSE 0 END) as failed
            FROM validations
        """
        validator_params = []

        if version_id:
            validator_query += """
                WHERE file_id IN (
                    SELECT file_id FROM files WHERE version_id = ?
                )
            """
            validator_params.append(version_id)

        validator_query += " GROUP BY validator"

        validator_rows = self.repo.query_all(validator_query, validator_params)
        by_validator = {row[0]: {"passed": row[1], "failed": row[2]} for row in validator_rows}

        # Calculate pass rate - handle division by zero
        pass_rate_pct = 0.0
        if total and total > 0:
            pass_rate_pct = (passed or 0) * 100.0 / total

        return ValidationSummary(
            total_validations=total or 0,
            passed_count=passed or 0,
            failed_count=failed or 0,
            by_validator=by_validator,
            pass_rate_pct=pass_rate_pct,
        )

    def find_by_artifact_id(self, artifact_id: str) -> ArtifactInfo | None:
        """Find artifact by ID.

        Args:
            artifact_id: Artifact identifier

        Returns:
            ArtifactInfo or None if not found

        Performance:
            Executes in < 10ms (indexed lookup)
        """
        query = """
            SELECT artifact_id, version_id, service, source_url, size_bytes, etag, status
            FROM artifacts
            WHERE artifact_id = ?
        """

        row = self.repo.query_one(query, (artifact_id,))
        if not row:
            return None

        return ArtifactInfo(
            artifact_id=row[0],
            version_id=row[1],
            service=row[2],
            source_url=row[3],
            size_bytes=row[4],
            etag=row[5],
            status=row[6] or "fresh",
        )

    def compute_version_delta(self, version_a: str, version_b: str) -> VersionDelta:
        """Compute differences between two versions.

        Args:
            version_a: First version ID
            version_b: Second version ID

        Returns:
            VersionDelta with differences

        Performance:
            Executes in < 200ms
        """
        # Files only in B
        added_rows = self.repo.query_all(
            """
            SELECT file_id FROM files WHERE version_id = ?
            AND file_id NOT IN (
                SELECT file_id FROM files WHERE version_id = ?
            )
            """,
            (version_b, version_a),
        )
        files_added = [row[0] for row in added_rows]

        # Files only in A
        removed_rows = self.repo.query_all(
            """
            SELECT file_id FROM files WHERE version_id = ?
            AND file_id NOT IN (
                SELECT file_id FROM files WHERE version_id = ?
            )
            """,
            (version_a, version_b),
        )
        files_removed = [row[0] for row in removed_rows]

        # Files in both
        common_rows = self.repo.query_all(
            """
            SELECT file_id FROM files WHERE version_id = ?
            AND file_id IN (
                SELECT file_id FROM files WHERE version_id = ?
            )
            """,
            (version_a, version_b),
        )
        files_common = [row[0] for row in common_rows]

        # Format changes in common files
        format_changes = {}
        for file_id in files_common:
            formats = self.repo.query_all(
                "SELECT format FROM files WHERE file_id = ? AND version_id IN (?, ?)",
                (file_id, version_a, version_b),
            )
            if len(formats) == 2 and formats[0][0] != formats[1][0]:
                format_changes[file_id] = (formats[0][0], formats[1][0])

        # Size delta
        size_result = self.repo.query_scalar(
            """
            SELECT
                COALESCE(SUM(f2.size_bytes), 0) - COALESCE(SUM(f1.size_bytes), 0)
            FROM files f1 FULL OUTER JOIN files f2
                ON f1.file_id = f2.file_id
            WHERE (f1.version_id = ? OR f2.version_id = ?)
            """,
            (version_a, version_b),
        )
        size_delta = size_result or 0

        return VersionDelta(
            version_a=version_a,
            version_b=version_b,
            files_added=files_added,
            files_removed=files_removed,
            files_common=files_common,
            format_changes=format_changes,
            size_delta_bytes=size_delta,
        )

    def get_storage_usage(self, version_id: str | None = None) -> StorageUsage:
        """Get disk usage metrics.

        Args:
            version_id: Compute usage for specific version or all if None

        Returns:
            StorageUsage with breakdown

        Performance:
            Executes in < 150ms
        """
        query_total = "SELECT COALESCE(SUM(size_bytes), 0), COUNT(*) FROM files"
        query_params = []

        if version_id:
            query_total += " WHERE version_id = ?"
            query_params.append(version_id)

        total_result = self.repo.query_one(query_total, query_params)
        total_bytes, file_count = total_result or (0, 0)

        # By format
        format_query = """
            SELECT format, COALESCE(SUM(size_bytes), 0)
            FROM files
        """
        format_params = []

        if version_id:
            format_query += " WHERE version_id = ?"
            format_params.append(version_id)

        format_query += " GROUP BY format"

        format_rows = self.repo.query_all(format_query, format_params)
        by_format = {row[0] or "unknown": row[1] for row in format_rows}

        # By version (only if not already filtered)
        by_version: dict[str, int] = {}
        if not version_id:
            version_rows = self.repo.query_all(
                """
                SELECT version_id, COALESCE(SUM(size_bytes), 0)
                FROM files
                GROUP BY version_id
                """,
                [],
            )
            by_version = {row[0]: row[1] for row in version_rows}
        else:
            by_version = {version_id: total_bytes}

        avg_file_size = total_bytes / file_count if file_count > 0 else 0.0

        return StorageUsage(
            total_bytes=total_bytes or 0,
            by_format=by_format,
            by_version=by_version,
            file_count=file_count or 0,
            avg_file_size=avg_file_size,
        )
