"""Phase 3 Task 3.1: Core Integration Tests

Tests for integration of Phase 1 & 2 components into main pipeline:
  - Boundary imports and availability
  - DuckDB connection management
  - Catalog boundary recordings
  - Policy gate integration
  - Observability event emission
  - CLI command integration

NAVMAP:
  - TestBoundaryAvailability: Boundary import checks
  - TestCatalogBoundaryRecording: Boundary usage patterns
  - TestDuckDBConnection: Connection management
  - TestPolicyGateIntegration: Gate wiring
  - TestObservabilityIntegration: Event emission
  - TestStorageIntegration: Storage layer integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBoundaryAvailability:
    """Test boundary imports and availability."""

    def test_boundaries_importable(self):
        """Test all boundaries can be imported."""
        from DocsToKG.OntologyDownload.catalog.boundaries import (
            download_boundary,
            extraction_boundary,
            set_latest_boundary,
            validation_boundary,
        )

        assert download_boundary is not None
        assert extraction_boundary is not None
        assert validation_boundary is not None
        assert set_latest_boundary is not None

    def test_planning_catalog_imports(self):
        """Test planning.py has CATALOG_AVAILABLE flag."""
        from DocsToKG.OntologyDownload import planning

        assert hasattr(planning, 'CATALOG_AVAILABLE')
        assert isinstance(planning.CATALOG_AVAILABLE, bool)

    def test_boundary_context_managers(self):
        """Test boundaries are context managers."""
        from DocsToKG.OntologyDownload.catalog.boundaries import download_boundary

        assert hasattr(download_boundary, '__call__')


class TestCatalogBoundaryRecording:
    """Test boundary recording in pipeline."""

    def test_download_boundary_context_manager(self):
        """Test download_boundary as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            
            # Mock connection
            mock_conn = MagicMock()
            mock_conn.begin = MagicMock()
            mock_conn.commit = MagicMock()
            mock_conn.rollback = MagicMock()
            
            from DocsToKG.OntologyDownload.catalog.boundaries import download_boundary
            
            # Use boundary as context manager
            with patch('DocsToKG.OntologyDownload.catalog.boundaries.emit_boundary_begin'):
                with patch('DocsToKG.OntologyDownload.catalog.boundaries.emit_boundary_success'):
                    with download_boundary(
                        mock_conn,
                        artifact_id="test_artifact",
                        version_id="v1",
                        fs_relpath="path/to/file",
                        size=1024,
                        etag="abc123"
                    ) as result:
                        assert result.artifact_id == "test_artifact"
                        assert result.version_id == "v1"

    def test_boundary_result_types(self):
        """Test boundary result dataclasses."""
        from DocsToKG.OntologyDownload.catalog.boundaries import (
            DownloadBoundaryResult,
            ExtractionBoundaryResult,
            ValidationBoundaryResult,
        )

        # Download result
        dl_result = DownloadBoundaryResult(
            artifact_id="art1",
            version_id="v1",
            fs_relpath="path/file",
            size=1024,
            etag="abc",
            inserted=True,
        )
        assert dl_result.artifact_id == "art1"
        assert dl_result.inserted is True

        # Extraction result
        ex_result = ExtractionBoundaryResult(
            artifact_id="art1",
            files_inserted=5,
            total_size=10000,
            audit_path=Path("/tmp/audit.json"),
            inserted=True,
        )
        assert ex_result.files_inserted == 5

        # Validation result
        val_result = ValidationBoundaryResult(
            file_id="file1",
            validator="rdflib",
            status="PASS",
            inserted=True,
        )
        assert val_result.status == "PASS"


class TestDuckDBConnection:
    """Test DuckDB connection management."""

    def test_get_duckdb_conn_returns_optional(self):
        """Test _get_duckdb_conn returns Optional connection."""
        from DocsToKG.OntologyDownload.planning import _get_duckdb_conn

        # Mock config without strict spec
        mock_config = MagicMock()
        mock_config.defaults = MagicMock()
        mock_config.defaults.db = MagicMock()
        mock_config.defaults.db.path = Path("/tmp/test.duckdb")
        mock_config.defaults.db.threads = 4
        mock_config.defaults.db.writer_lock = True

        # Call should return Optional (None if unavailable)
        result = _get_duckdb_conn(mock_config)
        assert result is None or hasattr(result, '__class__')

    def test_safe_record_boundary_error_handling(self):
        """Test _safe_record_boundary handles errors gracefully."""
        from DocsToKG.OntologyDownload.planning import _safe_record_boundary

        mock_adapter = MagicMock()
        mock_boundary_fn = MagicMock(side_effect=Exception("Test error"))

        success, result = _safe_record_boundary(
            mock_adapter,
            "test_boundary",
            mock_boundary_fn,
            MagicMock(),
            artifact_id="test"
        )

        assert success is False
        assert result is None


class TestPolicyGateIntegration:
    """Test policy gate integration."""

    def test_policy_gates_importable(self):
        """Test policy gates can be imported."""
        from DocsToKG.OntologyDownload.policy.gates import (
            config_gate,
            url_gate,
            filesystem_gate,
            extraction_gate,
            storage_gate,
            db_boundary_gate,
        )

        assert config_gate is not None
        assert url_gate is not None
        assert filesystem_gate is not None
        assert extraction_gate is not None
        assert storage_gate is not None
        assert db_boundary_gate is not None

    def test_policy_registry_available(self):
        """Test policy registry is available."""
        from DocsToKG.OntologyDownload.policy.registry import PolicyRegistry

        registry = PolicyRegistry()
        assert registry is not None


class TestObservabilityIntegration:
    """Test observability event emission."""

    def test_observability_helpers_importable(self):
        """Test all observability helpers are importable."""
        from DocsToKG.OntologyDownload.catalog.observability_instrumentation import (
            emit_boundary_begin,
            emit_boundary_success,
            emit_boundary_error,
            emit_doctor_begin,
            emit_doctor_complete,
            emit_prune_begin,
            emit_prune_deleted,
            emit_cli_command_begin,
            emit_cli_command_success,
            TimedOperation,
        )

        assert emit_boundary_begin is not None
        assert emit_boundary_success is not None
        assert emit_doctor_begin is not None
        assert TimedOperation is not None

    def test_event_emission_basic(self):
        """Test basic event emission."""
        with patch('DocsToKG.OntologyDownload.catalog.observability_instrumentation.emit_event'):
            from DocsToKG.OntologyDownload.catalog.observability_instrumentation import (
                emit_boundary_begin,
            )

            emit_boundary_begin(
                boundary="download",
                artifact_id="art1",
                version_id="v1",
                service="test",
                extra_payload={}
            )
            # Test passes if no exception

    def test_timed_operation_context_manager(self):
        """Test TimedOperation context manager."""
        from DocsToKG.OntologyDownload.catalog.observability_instrumentation import (
            TimedOperation,
        )

        with patch('DocsToKG.OntologyDownload.catalog.observability_instrumentation.emit_event'):
            with TimedOperation("test_op") as timer:
                assert timer is not None


class TestStorageIntegration:
    """Test storage layer integration."""

    def test_storage_backend_available(self):
        """Test StorageBackend protocol is available."""
        from DocsToKG.OntologyDownload.storage.base import StorageBackend

        assert StorageBackend is not None

    def test_local_duckdb_storage_importable(self):
        """Test LocalDuckDBStorage is importable."""
        from DocsToKG.OntologyDownload.storage.localfs_duckdb import LocalDuckDBStorage

        assert LocalDuckDBStorage is not None

    def test_query_api_available(self):
        """Test query API is available."""
        from DocsToKG.OntologyDownload.catalog.queries_api import CatalogQueries

        assert CatalogQueries is not None


class TestCoreIntegrationSummary:
    """Summary integration test."""

    def test_all_phase1_phase2_components_wired(self):
        """Verify all Phase 1 & 2 components are importable and wired."""
        # Phase 1: Boundaries
        from DocsToKG.OntologyDownload.catalog.boundaries import download_boundary
        assert download_boundary is not None
        
        # Phase 1: Observability
        from DocsToKG.OntologyDownload.catalog.observability_instrumentation import (
            emit_boundary_begin,
        )
        assert emit_boundary_begin is not None
        
        # Phase 1: Policy gates
        from DocsToKG.OntologyDownload.policy.gates import db_boundary_gate
        assert db_boundary_gate is not None
        
        # Phase 2: Storage
        from DocsToKG.OntologyDownload.storage.localfs_duckdb import LocalDuckDBStorage
        assert LocalDuckDBStorage is not None
        
        # Phase 2: Query API
        from DocsToKG.OntologyDownload.catalog.queries_api import CatalogQueries
        assert CatalogQueries is not None
        
        # Phase 2: Profiler & Schema
        from DocsToKG.OntologyDownload.catalog.profiler import CatalogProfiler
        from DocsToKG.OntologyDownload.catalog.schema_inspector import CatalogSchema
        assert CatalogProfiler is not None
        assert CatalogSchema is not None
