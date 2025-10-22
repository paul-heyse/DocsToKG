"""Tests for P1 Observability & Integrity: HTTP telemetry (unit + smoke).

Covers:
- HEAD request emission with content-type
- GET request emission with status/elapsed
- Retry/backoff visibility
- 304 Not Modified handling
- Error/timeout scenarios
- Telemetry threading through call chain
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional

from DocsToKG.ContentDownload.telemetry import (
    ATTEMPT_REASON_BACKOFF,
    ATTEMPT_REASON_CONN_ERROR,
    ATTEMPT_REASON_NOT_MODIFIED,
    ATTEMPT_REASON_TIMEOUT,
    ATTEMPT_STATUS_HTTP_200,
    ATTEMPT_STATUS_HTTP_304,
    ATTEMPT_STATUS_HTTP_HEAD,
    ATTEMPT_STATUS_RETRY,
    MultiSink,
    RunTelemetry,
    SimplifiedAttemptRecord,
)


class ListAttemptSink:
    """Test double for AttemptSink that collects events in-memory."""

    def __init__(self) -> None:
        self.io_attempts: list[SimplifiedAttemptRecord] = []
        self.manifests: list = []
        self.summaries: list = []

    def log_attempt(self, record, *, timestamp: Optional[str] = None) -> None:
        """No-op for resolver attempts (P1 uses log_io_attempt)."""
        pass

    def log_io_attempt(self, record: SimplifiedAttemptRecord) -> None:
        """Collect IO attempt records."""
        self.io_attempts.append(record)

    def log_manifest(self, entry) -> None:
        """No-op for manifest entries."""
        pass

    def log_summary(self, summary) -> None:
        """Collect summary records."""
        self.summaries.append(summary)

    def log_breaker_event(self, event) -> None:
        """No-op for breaker events."""
        pass

    def close(self) -> None:
        """No-op for cleanup."""
        pass


class TestSimplifiedAttemptRecord:
    """Tests for SimplifiedAttemptRecord construction and validation."""

    def test_construct_http_head(self) -> None:
        """Can construct a HEAD attempt record."""
        now = datetime.now(UTC)
        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=50,
        )

        assert rec.run_id == "run-1"
        assert rec.verb == "HEAD"
        assert rec.http_status == 200
        assert rec.elapsed_ms == 50

    def test_construct_with_extra_metadata(self) -> None:
        """Extra metadata is preserved."""
        now = datetime.now(UTC)
        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_RETRY,
            reason=ATTEMPT_REASON_BACKOFF,
            extra={"attempt": 2, "sleep_ms": 1000},
        )

        assert rec.extra["attempt"] == 2
        assert rec.extra["sleep_ms"] == 1000

    def test_construct_with_none_run_id_disabled(self) -> None:
        """run_id=None means telemetry is disabled."""
        now = datetime.now(UTC)
        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id=None,
            resolver=None,
            url="https://example.org/pdf",
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
        )

        assert rec.run_id is None


class TestListAttemptSink:
    """Tests for test double ListAttemptSink."""

    def test_collect_io_attempts(self) -> None:
        """Collects IO attempt records."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec1 = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
            http_status=200,
        )
        sink.log_io_attempt(rec1)

        rec2 = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_HTTP_200,
            http_status=200,
            bytes_written=50000,
        )
        sink.log_io_attempt(rec2)

        assert len(sink.io_attempts) == 2
        assert sink.io_attempts[0].verb == "HEAD"
        assert sink.io_attempts[1].verb == "GET"


def test_run_telemetry_log_io_attempt_fans_out() -> None:
    """RunTelemetry delegates IO attempts to all underlying sinks."""

    sink_a = ListAttemptSink()
    sink_b = ListAttemptSink()
    telemetry = RunTelemetry(MultiSink([sink_a, sink_b]))
    now = datetime.now(UTC)
    record = SimplifiedAttemptRecord(
        ts=now,
        run_id="run-1",
        resolver="resolver",
        url="https://example.org/pdf",
        verb="GET",
        status=ATTEMPT_STATUS_HTTP_200,
        http_status=200,
    )

    telemetry.log_io_attempt(record)

    assert sink_a.io_attempts == [record]
    assert sink_b.io_attempts == [record]


class TestHTTPHeadEmission:
    """Tests for HEAD request telemetry emission."""

    def test_head_success(self) -> None:
        """HEAD 200 is recorded with content-type."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
            http_status=200,
            content_type="application/pdf",
            elapsed_ms=45,
        )
        sink.log_io_attempt(rec)

        assert len(sink.io_attempts) == 1
        attempt = sink.io_attempts[0]
        assert attempt.verb == "HEAD"
        assert attempt.status == ATTEMPT_STATUS_HTTP_HEAD
        assert attempt.http_status == 200
        assert attempt.content_type == "application/pdf"

    def test_head_timeout(self) -> None:
        """HEAD timeout is recorded with error reason."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
            reason=ATTEMPT_REASON_TIMEOUT,
            http_status=None,
            elapsed_ms=5000,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.reason == ATTEMPT_REASON_TIMEOUT
        assert attempt.http_status is None


class TestHTTPGetEmission:
    """Tests for GET request telemetry emission."""

    def test_get_success_with_content_length(self) -> None:
        """GET 200 with Content-Length is recorded."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_HTTP_200,
            http_status=200,
            content_type="application/pdf",
            bytes_written=250000,
            content_length_hdr=250000,
            elapsed_ms=1200,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.verb == "GET"
        assert attempt.status == ATTEMPT_STATUS_HTTP_200
        assert attempt.bytes_written == 250000
        assert attempt.content_length_hdr == 250000

    def test_get_204_no_content(self) -> None:
        """GET 204 (no content) is recorded."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status="http-204",
            http_status=204,
            bytes_written=0,
            elapsed_ms=100,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.http_status == 204
        assert attempt.bytes_written == 0


class TestHTTPRetryEmission:
    """Tests for retry/backoff telemetry."""

    def test_retry_after_backoff(self) -> None:
        """Retry after backoff is recorded with sleep details."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_RETRY,
            reason=ATTEMPT_REASON_BACKOFF,
            extra={"attempt": 2, "sleep_ms": 1024},
            elapsed_ms=1050,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.status == ATTEMPT_STATUS_RETRY
        assert attempt.reason == ATTEMPT_REASON_BACKOFF
        assert attempt.extra["sleep_ms"] == 1024

    def test_multiple_retries_visible(self) -> None:
        """Multiple retry attempts are each recorded separately."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        # First attempt fails with 429
        rec1 = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status="http-429",
            http_status=429,
            elapsed_ms=100,
        )
        sink.log_io_attempt(rec1)

        # Backoff sleep (recorded as retry with sleep_ms)
        rec2 = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_RETRY,
            reason=ATTEMPT_REASON_BACKOFF,
            extra={"attempt": 2, "sleep_ms": 1000},
            elapsed_ms=1005,
        )
        sink.log_io_attempt(rec2)

        # Retry succeeds with 200
        rec3 = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_HTTP_200,
            http_status=200,
            bytes_written=100000,
            elapsed_ms=500,
        )
        sink.log_io_attempt(rec3)

        assert len(sink.io_attempts) == 3
        assert sink.io_attempts[0].http_status == 429
        assert sink.io_attempts[1].status == ATTEMPT_STATUS_RETRY
        assert sink.io_attempts[2].http_status == 200


class TestHTTP304NotModified:
    """Tests for 304 Not Modified (conditional request) handling."""

    def test_304_not_modified_recorded(self) -> None:
        """304 with not-modified reason."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_HTTP_304,
            reason=ATTEMPT_REASON_NOT_MODIFIED,
            http_status=304,
            elapsed_ms=80,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.http_status == 304
        assert attempt.reason == ATTEMPT_REASON_NOT_MODIFIED


class TestHTTPErrorEmission:
    """Tests for error scenarios (timeouts, connection errors, etc)."""

    def test_connection_error(self) -> None:
        """Connection error recorded with reason."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status="download-error",
            reason=ATTEMPT_REASON_CONN_ERROR,
            http_status=None,
            elapsed_ms=3000,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.reason == ATTEMPT_REASON_CONN_ERROR
        assert attempt.http_status is None

    def test_http_500_error(self) -> None:
        """HTTP 500 error recorded."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status="http-500",
            http_status=500,
            elapsed_ms=150,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.http_status == 500


class TestTelemetryDisabledBehavior:
    """Tests for no-op behavior when telemetry is None."""

    def test_telemetry_none_still_constructs(self) -> None:
        """Records can be constructed with run_id=None (disabled)."""
        now = datetime.now(UTC)
        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id=None,  # Disabled
            resolver=None,
            url="https://example.org/pdf",
            verb="HEAD",
            status=ATTEMPT_STATUS_HTTP_HEAD,
        )

        assert rec.run_id is None


class TestElapsedTimeMeasurement:
    """Tests for accurate elapsed time recording."""

    def test_elapsed_ms_is_integer_milliseconds(self) -> None:
        """elapsed_ms is recorded in milliseconds as int."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        # Simulate 1.234 seconds
        elapsed_ms_int = 1234

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_HTTP_200,
            http_status=200,
            elapsed_ms=elapsed_ms_int,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert isinstance(attempt.elapsed_ms, int)
        assert attempt.elapsed_ms == 1234


class TestBytesWrittenTracking:
    """Tests for Content-Length and bytes-written tracking."""

    def test_bytes_written_without_content_length(self) -> None:
        """bytes_written can be recorded without Content-Length."""
        sink = ListAttemptSink()
        now = datetime.now(UTC)

        rec = SimplifiedAttemptRecord(
            ts=now,
            run_id="run-1",
            resolver="unpaywall",
            url="https://example.org/pdf",
            verb="GET",
            status=ATTEMPT_STATUS_HTTP_200,
            http_status=200,
            bytes_written=100000,
            content_length_hdr=None,  # No Content-Length header
            elapsed_ms=500,
        )
        sink.log_io_attempt(rec)

        attempt = sink.io_attempts[0]
        assert attempt.bytes_written == 100000
        assert attempt.content_length_hdr is None
