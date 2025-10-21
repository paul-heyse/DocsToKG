# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_observability_instrumentation",
#   "purpose": "Tests for observability instrumentation (Task 1.3)",
#   "sections": [
#     {"id": "setup", "name": "Test Setup", "anchor": "SETUP", "kind": "infra"},
#     {"id": "boundary", "name": "Boundary Event Tests", "anchor": "BOUND", "kind": "tests"},
#     {"id": "doctor", "name": "Doctor Event Tests", "anchor": "DOC", "kind": "tests"},
#     {"id": "prune", "name": "Prune Event Tests", "anchor": "PRUNE", "kind": "tests"},
#     {"id": "cli", "name": "CLI Event Tests", "anchor": "CLI", "kind": "tests"},
#     {"id": "perf", "name": "Performance Event Tests", "anchor": "PERF", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Tests for observability instrumentation (Task 1.3)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from DocsToKG.OntologyDownload.catalog.observability_instrumentation import (
    TimedOperation,
    emit_boundary_begin,
    emit_boundary_error,
    emit_boundary_success,
    emit_cli_command_begin,
    emit_cli_command_error,
    emit_cli_command_success,
    emit_doctor_begin,
    emit_doctor_complete,
    emit_doctor_fixed,
    emit_doctor_issue_found,
    emit_prune_begin,
    emit_prune_deleted,
    emit_prune_orphan_found,
    emit_slow_operation,
    emit_slow_query,
)


# ============================================================================
# SETUP (SETUP)
# ============================================================================


@pytest.fixture
def mock_emit():
    """Mock the emit_event function."""
    with patch(
        "DocsToKG.OntologyDownload.catalog.observability_instrumentation.emit_event"
    ) as mock:
        yield mock


# ============================================================================
# BOUNDARY EVENT TESTS (BOUND)
# ============================================================================


class TestBoundaryEvents:
    """Tests for boundary operation events."""

    def test_emit_boundary_begin(self, mock_emit):
        """Test boundary begin event emission."""
        emit_boundary_begin(
            boundary="download",
            artifact_id="abc123",
            version_id="v1.0",
            service="hp",
        )
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "boundary.download.begin"
        assert call_args[1]["level"] == "INFO"

    def test_emit_boundary_success(self, mock_emit):
        """Test boundary success event emission."""
        emit_boundary_success(
            boundary="extract",
            artifact_id="abc123",
            version_id="v1.0",
            duration_ms=1234.56,
            extra_payload={"files": 10},
        )
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "boundary.extract.success"
        payload = call_args[1]["payload"]
        assert payload["duration_ms"] == 1234.6  # Rounded to 1 decimal place
        assert payload["files"] == 10

    def test_emit_boundary_error(self, mock_emit):
        """Test boundary error event emission."""
        error = ValueError("Test error")
        emit_boundary_error(
            boundary="validate",
            artifact_id="abc123",
            version_id="v1.0",
            error=error,
            duration_ms=500,
        )
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "boundary.validate.error"
        assert call_args[1]["level"] == "ERROR"
        payload = call_args[1]["payload"]
        assert payload["error_type"] == "ValueError"


# ============================================================================
# DOCTOR EVENT TESTS (DOC)
# ============================================================================


class TestDoctorEvents:
    """Tests for doctor operation events."""

    def test_emit_doctor_begin(self, mock_emit):
        """Test doctor begin event."""
        emit_doctor_begin()
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "catalog.doctor.begin"

    def test_emit_doctor_issue_found(self, mock_emit):
        """Test doctor issue found event."""
        emit_doctor_issue_found(
            issue_type="missing_file",
            severity="error",
            affected_records=5,
            details={"path": "/some/path"},
        )
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["issue_type"] == "missing_file"
        assert payload["severity"] == "error"
        assert payload["affected_records"] == 5

    def test_emit_doctor_fixed(self, mock_emit):
        """Test doctor issue fixed event."""
        emit_doctor_fixed(issue_type="orphan_record", count=3)
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["fixed_count"] == 3

    def test_emit_doctor_complete(self, mock_emit):
        """Test doctor complete event."""
        emit_doctor_complete(total_issues=10, fixed=7, duration_ms=2500)
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["total_issues"] == 10
        assert payload["fixed"] == 7


# ============================================================================
# PRUNE EVENT TESTS (PRUNE)
# ============================================================================


class TestPruneEvents:
    """Tests for prune operation events."""

    def test_emit_prune_begin(self, mock_emit):
        """Test prune begin event."""
        emit_prune_begin(dry_run=True)
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["operation"] == "prune"
        assert payload["dry_run"] is True

    def test_emit_prune_orphan_found(self, mock_emit):
        """Test prune orphan found event."""
        emit_prune_orphan_found(
            path="some/orphan/file.txt",
            size_bytes=1024,
            age_days=30,
        )
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["path"] == "some/orphan/file.txt"
        assert payload["size_bytes"] == 1024
        assert payload["age_days"] == 30

    def test_emit_prune_deleted(self, mock_emit):
        """Test prune complete event."""
        emit_prune_deleted(count=5, total_bytes=10240, duration_ms=500)
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["deleted_count"] == 5
        assert payload["freed_bytes"] == 10240


# ============================================================================
# CLI EVENT TESTS (CLI)
# ============================================================================


class TestCliEvents:
    """Tests for CLI command events."""

    def test_emit_cli_command_begin(self, mock_emit):
        """Test CLI command begin event."""
        start_time = emit_cli_command_begin(command="migrate", args={"dry_run": True})
        assert isinstance(start_time, float)
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "cli.migrate.begin"
        payload = call_args[1]["payload"]
        assert payload["command"] == "migrate"

    def test_emit_cli_command_success(self, mock_emit):
        """Test CLI command success event."""
        emit_cli_command_success(
            command="latest",
            duration_ms=150,
            result_summary={"status": "ok"},
        )
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "cli.latest.success"
        payload = call_args[1]["payload"]
        assert payload["duration_ms"] == 150

    def test_emit_cli_command_error(self, mock_emit):
        """Test CLI command error event."""
        error = RuntimeError("Connection failed")
        emit_cli_command_error(
            command="versions",
            duration_ms=100,
            error=error,
        )
        assert mock_emit.called
        call_args = mock_emit.call_args
        assert call_args[1]["event_type"] == "cli.versions.error"
        assert call_args[1]["level"] == "ERROR"


# ============================================================================
# PERFORMANCE EVENT TESTS (PERF)
# ============================================================================


class TestPerformanceEvents:
    """Tests for performance monitoring events."""

    def test_emit_slow_operation_under_threshold(self, mock_emit):
        """Test no event emitted for fast operations."""
        emit_slow_operation("fast_op", duration_ms=100, threshold_ms=1000)
        # Should not emit
        assert not mock_emit.called

    def test_emit_slow_operation_over_threshold(self, mock_emit):
        """Test event emitted for slow operations."""
        emit_slow_operation("slow_op", duration_ms=1500, threshold_ms=1000)
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["operation"] == "slow_op"
        assert payload["duration_ms"] == 1500

    def test_emit_slow_query_under_threshold(self, mock_emit):
        """Test no event for fast queries."""
        emit_slow_query("select", duration_ms=100, rows_examined=1000)
        assert not mock_emit.called

    def test_emit_slow_query_over_threshold(self, mock_emit):
        """Test event for slow queries."""
        emit_slow_query("join", duration_ms=1000, rows_examined=50000, threshold_ms=500)
        assert mock_emit.called
        payload = mock_emit.call_args[1]["payload"]
        assert payload["query_type"] == "join"
        assert payload["rows_examined"] == 50000


# ============================================================================
# TIMED OPERATION TESTS
# ============================================================================


class TestTimedOperation:
    """Tests for TimedOperation context manager."""

    def test_timed_operation_basic(self, mock_emit):
        """Test basic timed operation."""
        with TimedOperation("test_op") as timer:
            time.sleep(0.01)  # Sleep for 10ms
            elapsed = timer.elapsed_ms
            assert elapsed >= 10

    def test_timed_operation_duration(self, mock_emit):
        """Test that timed operation records duration."""
        with TimedOperation("work") as timer:
            time.sleep(0.02)
            assert timer.elapsed_ms >= 20

    def test_timed_operation_no_emit_on_error(self, mock_emit):
        """Test no emit on exception."""
        try:
            with TimedOperation("error_op"):
                raise ValueError("Test error")
        except ValueError:
            pass
        # Should not emit on exception
        assert not mock_emit.called


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestEventIntegration:
    """Integration tests for event emission flow."""

    def test_boundary_event_sequence(self, mock_emit):
        """Test complete boundary event sequence."""
        # Begin
        emit_boundary_begin("download", "abc", "v1", "service")
        assert mock_emit.call_count == 1

        # Success
        emit_boundary_success("download", "abc", "v1", 500)
        assert mock_emit.call_count == 2

        # Verify sequence
        calls = [call[1]["event_type"] for call in mock_emit.call_args_list]
        assert calls[0] == "boundary.download.begin"
        assert calls[1] == "boundary.download.success"

    def test_doctor_operation_sequence(self, mock_emit):
        """Test complete doctor operation sequence."""
        emit_doctor_begin()
        emit_doctor_issue_found("missing_file", "error", 3)
        emit_doctor_fixed("missing_file", 2)
        emit_doctor_complete(3, 2, 1000)

        assert mock_emit.call_count == 4
        calls = [call[1]["event_type"] for call in mock_emit.call_args_list]
        assert "doctor.begin" in calls[0]
        assert "doctor.complete" in calls[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
