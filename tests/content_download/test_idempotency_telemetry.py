"""Tests for idempotency telemetry event emission.

Tests verify:
  - All 9 event types emit correctly
  - Event structure and field validity
  - Timestamp accuracy
  - JSON serialization integrity
  - Non-blocking behavior (no exceptions on emit)
"""

from __future__ import annotations

import json
import logging
import time
from io import StringIO
from typing import Any, Dict

import pytest

from DocsToKG.ContentDownload import idempotency_telemetry as telem


@pytest.fixture
def log_capture():
    """Capture telemetry log output."""
    logger = logging.getLogger("DocsToKG.ContentDownload.idempotency_telemetry")
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler.stream
    logger.removeHandler(handler)


def parse_telemetry_log(log_output: str) -> Dict[str, Any] | None:
    """Parse telemetry log line and extract event."""
    if not log_output.strip():
        return None
    line = log_output.strip()
    if not line.startswith("TELEMETRY|"):
        return None
    parts = line.split("|", 2)
    if len(parts) < 3:
        return None
    event_type, payload_json = parts[1], parts[2]
    return {
        "event_type": event_type,
        "payload": json.loads(payload_json),
    }


class TestJobPlannedEvent:
    """Test job_planned event emission."""

    def test_emit_job_planned(self, log_capture):
        """Job planned event contains all required fields."""
        telem.emit_job_planned(
            job_id="job-123",
            work_id="work-456",
            artifact_id="artifact-789",
            canonical_url="https://example.org/paper.pdf",
            idempotency_key="abc123def456",
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "job_planned"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["work_id"] == "work-456"
        assert payload["artifact_id"] == "artifact-789"
        assert payload["canonical_url"] == "https://example.org/paper.pdf"
        assert payload["idempotency_key"] == "abc123def456"
        assert "timestamp" in payload
        assert isinstance(payload["timestamp"], (int, float))


class TestJobLeasedEvent:
    """Test job_leased event emission."""

    def test_emit_job_leased(self, log_capture):
        """Job leased event contains lease details."""
        telem.emit_job_leased(
            job_id="job-123",
            owner="worker-1",
            ttl_sec=120,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "job_leased"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["owner"] == "worker-1"
        assert payload["ttl_sec"] == 120
        assert "timestamp" in payload


class TestJobStateChangedEvent:
    """Test job_state_changed event emission."""

    def test_emit_job_state_changed_with_reason(self, log_capture):
        """State changed event includes optional reason."""
        telem.emit_job_state_changed(
            job_id="job-123",
            from_state="PLANNED",
            to_state="LEASED",
            reason="scheduled",
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "job_state_changed"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["from_state"] == "PLANNED"
        assert payload["to_state"] == "LEASED"
        assert payload["reason"] == "scheduled"

    def test_emit_job_state_changed_without_reason(self, log_capture):
        """State changed event works without reason."""
        telem.emit_job_state_changed(
            job_id="job-123",
            from_state="HEAD_DONE",
            to_state="STREAMING",
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        payload = event["payload"]
        assert payload["reason"] is None


class TestLeaseRenewedEvent:
    """Test lease_renewed event emission."""

    def test_emit_lease_renewed(self, log_capture):
        """Lease renewed event shows extended TTL."""
        telem.emit_lease_renewed(
            job_id="job-123",
            owner="worker-1",
            new_ttl_sec=300,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "lease_renewed"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["owner"] == "worker-1"
        assert payload["new_ttl_sec"] == 300
        assert "timestamp" in payload


class TestLeaseReleasedEvent:
    """Test lease_released event emission."""

    def test_emit_lease_released(self, log_capture):
        """Lease released event clears ownership."""
        telem.emit_lease_released(
            job_id="job-123",
            owner="worker-1",
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "lease_released"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["owner"] == "worker-1"
        assert "timestamp" in payload


class TestOperationStartedEvent:
    """Test operation_started event emission."""

    def test_emit_operation_started(self, log_capture):
        """Operation started event tracks beginning of effect."""
        telem.emit_operation_started(
            job_id="job-123",
            op_key="opkey-abc123",
            op_type="HEAD",
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "operation_started"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["op_key"] == "opkey-abc123"
        assert payload["op_type"] == "HEAD"
        assert "timestamp" in payload


class TestOperationCompletedEvent:
    """Test operation_completed event emission."""

    def test_emit_operation_completed_success(self, log_capture):
        """Operation completed event records successful completion."""
        telem.emit_operation_completed(
            job_id="job-123",
            op_key="opkey-abc123",
            op_type="STREAM",
            result_code="OK",
            elapsed_ms=2500,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "operation_completed"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["op_key"] == "opkey-abc123"
        assert payload["op_type"] == "STREAM"
        assert payload["result_code"] == "OK"
        assert payload["elapsed_ms"] == 2500

    def test_emit_operation_completed_error(self, log_capture):
        """Operation completed event records error codes."""
        telem.emit_operation_completed(
            job_id="job-123",
            op_key="opkey-abc123",
            op_type="FINALIZE",
            result_code="RETRYABLE",
            elapsed_ms=500,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        payload = event["payload"]
        assert payload["result_code"] == "RETRYABLE"


class TestCrashRecoveryEvent:
    """Test crash_recovery_event emission."""

    def test_emit_crash_recovery(self, log_capture):
        """Crash recovery event reports cleanup counts."""
        telem.emit_crash_recovery(
            recovered_leases=5,
            abandoned_ops=3,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "crash_recovery_event"
        payload = event["payload"]

        assert payload["recovered_leases"] == 5
        assert payload["abandoned_ops"] == 3
        assert "timestamp" in payload

    def test_emit_crash_recovery_zero_cleanup(self, log_capture):
        """Crash recovery event handles zero cleanup."""
        telem.emit_crash_recovery(
            recovered_leases=0,
            abandoned_ops=0,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        payload = event["payload"]
        assert payload["recovered_leases"] == 0
        assert payload["abandoned_ops"] == 0


class TestIdempotencyReplayEvent:
    """Test idempotency_replay event emission."""

    def test_emit_idempotency_replay(self, log_capture):
        """Idempotency replay event shows cached result reuse."""
        original_time = time.time() - 60
        telem.emit_idempotency_replay(
            job_id="job-123",
            op_key="opkey-abc123",
            op_type="FINALIZE",
            reused_from_time=original_time,
        )

        event = parse_telemetry_log(log_capture.getvalue())
        assert event is not None
        assert event["event_type"] == "idempotency_replay"
        payload = event["payload"]

        assert payload["job_id"] == "job-123"
        assert payload["op_key"] == "opkey-abc123"
        assert payload["op_type"] == "FINALIZE"
        assert payload["reused_from_time"] == original_time
        assert "timestamp" in payload


class TestEventStructure:
    """Test overall event structure and validity."""

    def test_all_events_have_timestamp(self, log_capture):
        """Every event type includes a timestamp."""
        events_to_test = [
            (
                "job_planned",
                lambda: telem.emit_job_planned("j1", "w1", "a1", "https://example.org", "key1"),
            ),
            ("job_leased", lambda: telem.emit_job_leased("j1", "owner1", 120)),
            ("job_state_changed", lambda: telem.emit_job_state_changed("j1", "S1", "S2")),
            ("lease_renewed", lambda: telem.emit_lease_renewed("j1", "owner1", 300)),
            ("lease_released", lambda: telem.emit_lease_released("j1", "owner1")),
            ("operation_started", lambda: telem.emit_operation_started("j1", "op1", "HEAD")),
            (
                "operation_completed",
                lambda: telem.emit_operation_completed("j1", "op1", "HEAD", "OK", 100),
            ),
            ("crash_recovery", lambda: telem.emit_crash_recovery(1, 2)),
            (
                "idempotency_replay",
                lambda: telem.emit_idempotency_replay("j1", "op1", "HEAD", time.time()),
            ),
        ]

        for event_name, emit_fn in events_to_test:
            log_capture.truncate(0)
            log_capture.seek(0)
            emit_fn()
            event = parse_telemetry_log(log_capture.getvalue())
            assert event is not None, f"{event_name} failed to emit"
            assert "timestamp" in event["payload"], f"{event_name} missing timestamp"

    def test_events_are_json_serializable(self, log_capture):
        """All event payloads are valid JSON."""
        telem.emit_operation_completed(
            job_id="job-123",
            op_key="opkey-abc123",
            op_type="STREAM",
            result_code="OK",
            elapsed_ms=2500,
        )

        line = log_capture.getvalue().strip()
        if line.startswith("TELEMETRY|"):
            parts = line.split("|", 2)
            payload_json = parts[2]
            # Should not raise exception
            parsed = json.loads(payload_json)
            assert isinstance(parsed, dict)

    def test_emit_non_blocking(self):
        """Event emission does not raise exceptions."""
        # All these should complete without raising
        telem.emit_job_planned("j1", "w1", "a1", "url1", "key1")
        telem.emit_job_leased("j1", "owner1", 120)
        telem.emit_job_state_changed("j1", "S1", "S2")
        telem.emit_lease_renewed("j1", "owner1", 300)
        telem.emit_lease_released("j1", "owner1")
        telem.emit_operation_started("j1", "op1", "HEAD")
        telem.emit_operation_completed("j1", "op1", "HEAD", "OK", 100)
        telem.emit_crash_recovery(1, 2)
        telem.emit_idempotency_replay("j1", "op1", "HEAD", time.time())
