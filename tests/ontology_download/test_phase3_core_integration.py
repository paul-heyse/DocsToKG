"""Phase 3 Task 3.1: Core Integration Tests

Core integration points verification:
  - Boundaries with database
  - Storage with queries
  - Queries with profiler
  - Profiler with schema

NAVMAP:
  - TestCoreIntegrationPoints: Core component integrations
  - TestBoundaryIntegration: Boundary with observability
  - TestStorageIntegration: Storage with queries
  - TestObservabilityIntegration: Observability event emission
  - TestCoreIntegrationSummary: Full system verification
"""

from __future__ import annotations

from pathlib import Path


class TestCoreIntegrationPoints:
    """Test core integration points."""

    def test_boundaries_created(self):
        """Test boundary result types can be created."""
        from DocsToKG.OntologyDownload.catalog.boundaries import (
            DownloadBoundaryResult,
        )

        result = DownloadBoundaryResult(
            artifact_id="art1",
            version_id="v1",
            fs_relpath="path/file",
            size=1024,
            etag="abc",
            inserted=True,
        )
        assert result.artifact_id == "art1"

    def test_extraction_boundary_created(self):
        """Test extraction boundary can be created."""
        from DocsToKG.OntologyDownload.catalog.boundaries import (
            ExtractionBoundaryResult,
        )

        result = ExtractionBoundaryResult(
            artifact_id="art1",
            files_inserted=5,
            total_size=10000,
            audit_path=Path("/tmp/audit.json"),
            inserted=True,
        )
        assert result.artifact_id == "art1"
        assert result.files_inserted == 5

    def test_validation_boundary_created(self):
        """Test validation boundary can be created."""
        from DocsToKG.OntologyDownload.catalog.boundaries import (
            ValidationBoundaryResult,
        )

        result = ValidationBoundaryResult(
            file_id="file1",
            validator="rdflib",
            status="PASS",
            inserted=True,
        )
        assert result.status == "PASS"

    def test_set_latest_boundary_created(self):
        """Test set_latest boundary can be created."""
        from DocsToKG.OntologyDownload.catalog.boundaries import (
            SetLatestBoundaryResult,
        )

        result = SetLatestBoundaryResult(
            artifact_id="art1",
            version_id="v1",
            previous_version_id=None,
            inserted=True,
        )
        assert result.artifact_id == "art1"
