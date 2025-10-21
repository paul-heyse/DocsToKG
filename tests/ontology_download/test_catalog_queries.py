"""Comprehensive tests for catalog query API.

Tests cover all query methods, error handling, and performance characteristics.
Uses mocks to avoid database dependencies.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from DocsToKG.OntologyDownload.catalog.queries_api import CatalogQueries
from DocsToKG.OntologyDownload.catalog.queries_dto import (
    StorageUsage,
    VersionDelta,
    VersionStats,
)


@pytest.fixture
def mock_repo():
    """Create mock Repo instance."""
    return MagicMock()


@pytest.fixture
def queries(mock_repo):
    """Create CatalogQueries instance with mock repo."""
    return CatalogQueries(mock_repo)


class TestVersionStats:
    """Test get_version_stats query method."""

    def test_get_stats_success(self, queries, mock_repo):
        """Test getting version stats successfully."""
        mock_repo.get_version.return_value = {
            "version_id": "v1.0",
            "service": "openalex",
            "created_at": datetime(2025, 1, 1),
        }
        mock_repo.query_scalar.side_effect = [
            (100, 5000000),  # file count and size
            (95, 5),  # validation passed/failed
            (1000,),  # artifact count
        ]

        result = queries.get_version_stats("v1.0")

        assert result.version_id == "v1.0"
        assert result.service == "openalex"
        assert result.file_count == 100
        assert result.total_size == 5000000
        assert result.validation_passed == 95
        assert result.validation_failed == 5
        assert result.artifacts_count == 1000

    def test_get_stats_not_found(self, queries, mock_repo):
        """Test getting stats for non-existent version."""
        mock_repo.get_version.return_value = None

        result = queries.get_version_stats("missing")

        assert result is None

    def test_validation_passed_pct(self, queries, mock_repo):
        """Test validation pass percentage calculation."""
        mock_repo.get_version.return_value = {
            "version_id": "v1.0",
            "service": "test",
            "created_at": datetime.utcnow(),
        }
        mock_repo.query_scalar.side_effect = [(10, 0), (80, 20), (100,)]

        result = queries.get_version_stats("v1.0")

        assert result.validation_passed_pct == 80.0


class TestListVersions:
    """Test list_versions query method."""

    def test_list_all_versions(self, queries, mock_repo):
        """Test listing all versions."""
        mock_repo.query_all.return_value = [
            ("v1.0", "openalex", datetime(2025, 1, 1), "hash1"),
            ("v1.1", "openalex", datetime(2025, 1, 2), "hash2"),
        ]

        result = queries.list_versions()

        assert len(result) == 2
        assert result[0].version_id == "v1.0"
        assert result[1].version_id == "v1.1"

    def test_list_versions_filter_by_service(self, queries, mock_repo):
        """Test listing versions filtered by service."""
        mock_repo.query_all.return_value = [
            ("v1.0", "openalex", datetime(2025, 1, 1), "hash1"),
        ]

        result = queries.list_versions(service="openalex")

        assert len(result) == 1
        assert result[0].service == "openalex"

    def test_list_versions_pagination(self, queries, mock_repo):
        """Test version listing with pagination."""
        mock_repo.query_all.return_value = []

        queries.list_versions(limit=50, offset=100)

        # Verify query includes pagination params
        call_args = mock_repo.query_all.call_args
        assert 50 in call_args[0][1]  # limit
        assert 100 in call_args[0][1]  # offset


class TestListFiles:
    """Test list_files query method."""

    def test_list_all_files(self, queries, mock_repo):
        """Test listing all files."""
        mock_repo.query_all.return_value = [
            ("f1", "file1.ttl", 1000, "ttl", datetime(2025, 1, 1)),
            ("f2", "file2.rdf", 2000, "rdf", datetime(2025, 1, 2)),
        ]

        result = queries.list_files()

        assert len(result) == 2
        assert result[0].file_id == "f1"
        assert result[1].file_id == "f2"

    def test_list_files_by_version(self, queries, mock_repo):
        """Test listing files for specific version."""
        mock_repo.query_all.return_value = [
            ("f1", "file1.ttl", 1000, "ttl", datetime(2025, 1, 1)),
        ]

        result = queries.list_files(version_id="v1.0")

        assert len(result) == 1
        assert result[0].file_id == "f1"

    def test_list_files_filter_by_format(self, queries, mock_repo):
        """Test filtering files by format."""
        mock_repo.query_all.return_value = [
            ("f1", "file1.ttl", 1000, "ttl", datetime(2025, 1, 1)),
        ]

        queries.list_files(format_filter="ttl")

        call_args = mock_repo.query_all.call_args
        assert "ttl" in call_args[0][1]

    def test_list_files_with_prefix(self, queries, mock_repo):
        """Test filtering files by path prefix."""
        mock_repo.query_all.return_value = []

        queries.list_files(prefix="data/")

        call_args = mock_repo.query_all.call_args
        assert "data/%" in call_args[0][1]


class TestListValidations:
    """Test list_validations query method."""

    def test_list_all_validations(self, queries, mock_repo):
        """Test listing all validations."""
        mock_repo.query_all.return_value = [
            ("v1", "f1", "rdflib", True, None, datetime(2025, 1, 1)),
            ("v2", "f2", "pronto", False, "error", datetime(2025, 1, 2)),
        ]

        result = queries.list_validations()

        assert len(result) == 2
        assert result[0].validation_id == "v1"
        assert result[0].passed is True
        assert result[1].passed is False

    def test_list_validations_for_file(self, queries, mock_repo):
        """Test listing validations for specific file."""
        mock_repo.query_all.return_value = [
            ("v1", "f1", "rdflib", True, None, datetime(2025, 1, 1)),
        ]

        result = queries.list_validations(file_id="f1")

        assert len(result) == 1
        assert result[0].file_id == "f1"

    def test_list_validations_passed_only(self, queries, mock_repo):
        """Test listing only passing validations."""
        mock_repo.query_all.return_value = [
            ("v1", "f1", "rdflib", True, None, datetime(2025, 1, 1)),
        ]

        queries.list_validations(passed_only=True)

        call_args = mock_repo.query_all.call_args
        assert "passed = true" in call_args[0][0]


class TestValidationSummary:
    """Test get_validation_summary query method."""

    def test_summary_all_validations(self, queries, mock_repo):
        """Test getting summary of all validations."""
        mock_repo.query_scalar.side_effect = [(100, 80, 20)]
        mock_repo.query_all.return_value = [
            ("rdflib", 70, 5),
            ("pronto", 10, 15),
        ]

        result = queries.get_validation_summary()

        assert result.total_validations == 100
        assert result.passed_count == 80
        assert result.failed_count == 20
        assert result.pass_rate_pct == 80.0

    def test_summary_by_validator(self, queries, mock_repo):
        """Test validator breakdown in summary."""
        mock_repo.query_scalar.return_value = (100, 80, 20)
        mock_repo.query_all.return_value = [
            ("rdflib", 70, 5),
            ("pronto", 10, 15),
        ]

        result = queries.get_validation_summary()

        assert "rdflib" in result.by_validator
        assert result.by_validator["rdflib"]["passed"] == 70
        assert result.by_validator["pronto"]["failed"] == 15


class TestFindArtifact:
    """Test find_by_artifact_id query method."""

    def test_find_artifact_exists(self, queries, mock_repo):
        """Test finding existing artifact."""
        mock_repo.query_one.return_value = (
            "a1",
            "v1.0",
            "openalex",
            "https://example.com/doc.pdf",
            5000,
            "etag123",
            "fresh",
        )

        result = queries.find_by_artifact_id("a1")

        assert result.artifact_id == "a1"
        assert result.version_id == "v1.0"
        assert result.status == "fresh"

    def test_find_artifact_not_found(self, queries, mock_repo):
        """Test finding non-existent artifact."""
        mock_repo.query_one.return_value = None

        result = queries.find_by_artifact_id("missing")

        assert result is None


class TestVersionDelta:
    """Test compute_version_delta query method."""

    def test_delta_with_changes(self, queries, mock_repo):
        """Test computing delta between versions."""
        # Mock the query_all calls: added, removed, common files, and format queries for each file
        mock_repo.query_all.side_effect = [
            [("f2",), ("f3",)],  # added files (2 files)
            [("f4",)],  # removed files
            [("f1",), ("f5",)],  # common files (2 files)
            # Format queries for each common file
            [("ttl",), ("ttl",)],  # f1 same format
            [("rdf",), ("ttl",)],  # f5 format changed
        ]
        mock_repo.query_scalar.return_value = 1000  # size delta

        result = queries.compute_version_delta("v1.0", "v1.1")

        assert result.version_a == "v1.0"
        assert result.version_b == "v1.1"
        assert len(result.files_added) == 2
        assert len(result.files_removed) == 1
        assert len(result.format_changes) == 1


class TestStorageUsage:
    """Test get_storage_usage query method."""

    def test_storage_usage_total(self, queries, mock_repo):
        """Test getting total storage usage."""
        mock_repo.query_one.return_value = (10000000, 100)
        mock_repo.query_all.side_effect = [
            [("ttl", 5000000), ("rdf", 5000000)],  # by_format
            [("v1.0", 5000000), ("v1.1", 5000000)],  # by_version
        ]

        result = queries.get_storage_usage()

        assert result.total_bytes == 10000000
        assert result.file_count == 100
        assert result.avg_file_size == 100000.0

    def test_storage_usage_by_format(self, queries, mock_repo):
        """Test storage breakdown by format."""
        mock_repo.query_one.return_value = (10000000, 100)
        mock_repo.query_all.side_effect = [
            [("ttl", 5000000), ("rdf", 5000000)],
            [("v1.0", 5000000), ("v1.1", 5000000)],
        ]

        result = queries.get_storage_usage()

        assert "ttl" in result.by_format
        assert result.by_format["ttl"] == 5000000

    def test_storage_usage_properties(self, queries, mock_repo):
        """Test storage usage property calculations."""
        mock_repo.query_one.return_value = (1073741824, 10)  # 1 GB
        mock_repo.query_all.side_effect = [[], []]

        result = queries.get_storage_usage()

        assert result.total_mb == 1024.0
        assert result.total_gb == 1.0


class TestDTOs:
    """Test DTO validation and properties."""

    def test_version_stats_dto(self):
        """Test VersionStats DTO."""
        stats = VersionStats(
            version_id="v1.0",
            service="openalex",
            created_at=datetime(2025, 1, 1),
            file_count=100,
            total_size=5000000,
            validation_passed=95,
            validation_failed=5,
            artifacts_count=1000,
        )

        assert stats.version_id == "v1.0"
        assert stats.validation_passed_pct == 95.0

    def test_version_delta_properties(self):
        """Test VersionDelta DTO properties."""
        delta = VersionDelta(
            version_a="v1.0",
            version_b="v1.1",
            files_added=["f2"],
            files_removed=["f3"],
            files_common=["f1"],
            format_changes={"f4": ("ttl", "rdf")},
            size_delta_bytes=1000,
        )

        assert delta.total_changes == 3

    def test_storage_usage_properties(self):
        """Test StorageUsage DTO properties."""
        usage = StorageUsage(
            total_bytes=1073741824,
            by_format={"ttl": 536870912},
            by_version={"v1.0": 1073741824},
            file_count=100,
            avg_file_size=10737418.24,
        )

        assert usage.total_mb == 1024.0
        assert usage.total_gb == 1.0


class TestErrorHandling:
    """Test error handling in queries."""

    def test_query_handles_none_results(self, queries, mock_repo):
        """Test handling of None query results."""
        mock_repo.query_one.return_value = None
        mock_repo.query_all.return_value = []

        # Should not raise
        result = queries.find_by_artifact_id("missing")
        assert result is None

        result = queries.list_versions()
        assert result == []

    def test_validation_summary_zero_validations(self, queries, mock_repo):
        """Test validation summary with zero validations."""
        mock_repo.query_scalar.return_value = (0, 0, 0)
        mock_repo.query_all.return_value = []

        result = queries.get_validation_summary()

        assert result.total_validations == 0
        assert result.pass_rate_pct == 0.0  # 0% when no validations
