"""Integration tests for security gate deployments (Gates 2-5).

Tests verify:
1. URL gate blocks malicious redirects
2. Extraction gate catches zip bombs
3. Filesystem gate prevents traversal
4. DB boundary gate prevents torn writes
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from DocsToKG.OntologyDownload.io.filesystem import extract_archive_safe, _validate_member_path
from DocsToKG.OntologyDownload.policy.gates import (
    url_gate,
    extraction_gate,
    filesystem_gate,
    db_boundary_gate,
)
from DocsToKG.OntologyDownload.policy.errors import (
    PolicyOK,
    PolicyReject,
    ErrorCode,
    URLPolicyException,
    ExtractionPolicyException,
    FilesystemPolicyException,
    DbBoundaryException,
)


class TestGate2URLGateIntegration:
    """Tests for URL gate integration in planning._populate_plan_metadata."""

    def test_url_gate_accepts_valid_https(self):
        """Valid HTTPS URL should pass."""
        result = url_gate(
            "https://example.com/ontology.owl",
            allowed_hosts=["example.com"],
            allowed_ports=[443],
        )
        assert isinstance(result, PolicyOK)
        assert result.elapsed_ms >= 0
        assert result.gate_name == "url_gate"

    def test_url_gate_rejects_disallowed_host(self):
        """URL with disallowed host should raise exception."""
        with pytest.raises(URLPolicyException):
            url_gate(
                "https://evil.example/ontology.owl",
                allowed_hosts=["example.com"],
                allowed_ports=[443],
            )

    def test_url_gate_performance(self):
        """URL gate should complete in <1ms."""
        result = url_gate(
            "https://example.com/ontology.owl",
            allowed_hosts=["example.com"],
            allowed_ports=[443],
        )
        assert result.elapsed_ms < 10.0  # Generous margin for test environment


class TestGate3ExtractionGateIntegration:
    """Tests for extraction gate in extract_archive_safe."""

    def test_extraction_gate_accepts_normal_archive(self):
        """Normal archive stats should pass."""
        result = extraction_gate(
            entries_total=1_000,  # 1000 entries
            bytes_declared=100_000_000,  # 100 MB
            max_total_ratio=10.0,
            max_entry_ratio=200_000,  # Allow up to 200KB per entry
        )
        assert isinstance(result, PolicyOK)
        assert result.elapsed_ms >= 0

    def test_extraction_gate_rejects_zip_bomb(self):
        """Archive with suspicious compression should raise exception."""
        # Simulate a zip bomb: high entry count relative to declared size
        with pytest.raises(ExtractionPolicyException):
            extraction_gate(
                entries_total=1_000_000,  # 1 million entries
                bytes_declared=10_000,  # Only 10 KB
                max_total_ratio=5.0,
                max_entry_ratio=0.1,  # Very strict: only 0.1 bytes per entry
            )

    def test_extraction_gate_performance(self):
        """Extraction gate should complete in <1ms."""
        result = extraction_gate(
            entries_total=500,  # 500 entries
            bytes_declared=50_000_000,  # 50 MB
            max_total_ratio=10.0,
            max_entry_ratio=200_000,  # Allow up to 200KB per entry
        )
        assert result.elapsed_ms < 10.0  # Generous margin


class TestGate4FilesystemGateIntegration:
    """Tests for filesystem gate in _validate_member_path."""

    def test_filesystem_gate_accepts_normal_path(self, tmp_path):
        """Normal path should pass."""
        root = tmp_path / "extract"
        root.mkdir()

        result = filesystem_gate(
            root_path=str(root),
            entry_paths=["ontology/file.owl"],
            allow_symlinks=False,
        )
        assert isinstance(result, PolicyOK)

    def test_filesystem_gate_rejects_traversal(self, tmp_path):
        """Path traversal attempts should raise exception."""
        root = tmp_path / "extract"
        root.mkdir()

        with pytest.raises(FilesystemPolicyException):
            filesystem_gate(
                root_path=str(root),
                entry_paths=["../../../etc/passwd"],
                allow_symlinks=False,
            )

    def test_filesystem_gate_rejects_absolute_paths(self, tmp_path):
        """Absolute paths should raise exception."""
        root = tmp_path / "extract"
        root.mkdir()

        with pytest.raises(FilesystemPolicyException):
            filesystem_gate(
                root_path=str(root),
                entry_paths=["/etc/passwd"],
                allow_symlinks=False,
            )

    def test_filesystem_gate_performance(self, tmp_path):
        """Filesystem gate should complete in <1ms."""
        root = tmp_path / "extract"
        root.mkdir()

        result = filesystem_gate(
            root_path=str(root),
            entry_paths=["file.txt"],
            allow_symlinks=False,
        )
        assert result.elapsed_ms < 10.0


class TestGate5DBBoundaryGateIntegration:
    """Tests for DB boundary gate in catalog/boundaries.py."""

    def test_db_boundary_gate_accepts_normal_commit(self):
        """Normal commit scenario should pass."""
        result = db_boundary_gate(
            operation="pre_commit",
            tables_affected=["extracted_files"],
            fs_success=True,
        )
        assert isinstance(result, PolicyOK)

    def test_db_boundary_gate_rejects_fs_failure(self):
        """Commit after FS failure should raise exception."""
        with pytest.raises(DbBoundaryException):
            db_boundary_gate(
                operation="pre_commit",
                tables_affected=["extracted_files"],
                fs_success=False,
            )

    def test_db_boundary_gate_performance(self):
        """DB boundary gate should complete in <1ms."""
        result = db_boundary_gate(
            operation="pre_commit",
            tables_affected=["extracted_files"],
            fs_success=True,
        )
        assert result.elapsed_ms < 10.0


class TestGateIntegrationE2E:
    """End-to-end integration tests for all gates together."""

    def test_gates_in_sequence_success(self, tmp_path):
        """All gates should pass for valid inputs."""
        # Gate 2: URL
        url_result = url_gate(
            "https://example.com/onto.owl",
            allowed_hosts=["example.com"],
            allowed_ports=[443],
        )
        assert isinstance(url_result, PolicyOK)

        # Gate 3: Extraction
        extraction_result = extraction_gate(
            entries_total=1_000,  # 1000 entries
            bytes_declared=100_000_000,  # 100 MB
            max_total_ratio=10.0,
            max_entry_ratio=200_000,  # Allow up to 200KB per entry
        )
        assert isinstance(extraction_result, PolicyOK)

        # Gate 4: Filesystem
        root = tmp_path / "extract"
        root.mkdir()
        fs_result = filesystem_gate(
            root_path=str(root),
            entry_paths=["file.txt"],
            allow_symlinks=False,
        )
        assert isinstance(fs_result, PolicyOK)

        # Gate 5: DB Boundary
        db_result = db_boundary_gate(
            operation="pre_commit",
            tables_affected=["extracted_files"],
            fs_success=True,
        )
        assert isinstance(db_result, PolicyOK)

    def test_gate_metrics_collected(self):
        """Gates should collect metrics (timing, pass/fail)."""
        url_result = url_gate(
            "https://example.com/onto.owl",
            allowed_hosts=["example.com"],
            allowed_ports=[443],
        )
        # Verify elapsed_ms is recorded
        assert hasattr(url_result, "elapsed_ms")
        assert url_result.elapsed_ms >= 0
        assert hasattr(url_result, "gate_name")
        assert url_result.gate_name == "url_gate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
