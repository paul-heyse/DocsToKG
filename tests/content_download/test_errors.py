"""Unit tests for DocsToKG.ContentDownload.errors module."""

import logging
from unittest.mock import Mock, patch

import pytest

from DocsToKG.ContentDownload.errors import (
    ContentPolicyError,
    DownloadError,
    NetworkError,
    RateLimitError,
    format_download_summary,
    get_actionable_error_message,
    log_download_failure,
)


class TestDownloadError:
    """Test DownloadError dataclass."""

    def test_init_with_all_fields(self):
        """Test initialization with all fields provided."""
        error = DownloadError(
            error_type="network",
            message="Connection failed",
            url="https://example.org/test.pdf",
            work_id="W12345",
            http_status=500,
            reason_code="connection_error",
            suggestion="Check network connectivity",
            metadata={"retry_count": 3},
        )
        assert error.error_type == "network"
        assert error.message == "Connection failed"
        assert error.url == "https://example.org/test.pdf"
        assert error.work_id == "W12345"
        assert error.http_status == 500
        assert error.reason_code == "connection_error"
        assert error.suggestion == "Check network connectivity"
        assert error.metadata["retry_count"] == 3

    def test_init_with_minimal_fields(self):
        """Test initialization with only required fields."""
        error = DownloadError(error_type="unknown", message="Something went wrong")
        assert error.error_type == "unknown"
        assert error.message == "Something went wrong"
        assert error.url is None
        assert error.work_id is None
        assert error.http_status is None
        assert error.reason_code is None
        assert error.suggestion is None
        assert error.metadata == {}

    def test_metadata_initialization(self):
        """Test that metadata is properly initialized when None."""
        error = DownloadError(error_type="test", message="test", metadata=None)
        assert error.metadata == {}


class TestNetworkError:
    """Test NetworkError exception."""

    def test_init_with_url_and_details(self):
        """Test initialization with URL and details."""
        exc = NetworkError("Connection timeout", url="https://example.org", details={"timeout": 30})
        assert str(exc) == "Connection timeout"
        assert exc.url == "https://example.org"
        assert exc.details == {"timeout": 30}

    def test_init_minimal(self):
        """Test initialization with only message."""
        exc = NetworkError("Network error")
        assert str(exc) == "Network error"
        assert exc.url is None
        assert exc.details == {}


class TestContentPolicyError:
    """Test ContentPolicyError exception."""

    def test_init_with_violation_and_policy(self):
        """Test initialization with violation and policy."""
        exc = ContentPolicyError(
            "Content type not allowed",
            violation="content-type",
            policy={"allowed_types": ["application/pdf"]},
        )
        assert str(exc) == "Content type not allowed"
        assert exc.violation == "content-type"
        assert exc.policy == {"allowed_types": ["application/pdf"]}

    def test_init_minimal(self):
        """Test initialization with only message and violation."""
        exc = ContentPolicyError("Policy violation", violation="max-bytes")
        assert str(exc) == "Policy violation"
        assert exc.violation == "max-bytes"
        assert exc.policy == {}


class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_init_with_retry_after_and_domain(self):
        """Test initialization with retry_after and domain."""
        exc = RateLimitError("Rate limited", retry_after=60.0, domain="example.org")
        assert str(exc) == "Rate limited"
        assert exc.retry_after == 60.0
        assert exc.domain == "example.org"

    def test_init_minimal(self):
        """Test initialization with only message."""
        exc = RateLimitError("Too many requests")
        assert str(exc) == "Too many requests"
        assert exc.retry_after is None
        assert exc.domain is None


class TestGetActionableErrorMessage:
    """Test get_actionable_error_message function."""

    def test_http_401(self):
        """Test 401 authentication error."""
        msg, suggestion = get_actionable_error_message(401, None)
        assert "Authentication required" in msg
        assert "authentication credentials" in suggestion.lower()

    def test_http_403(self):
        """Test 403 forbidden error."""
        msg, suggestion = get_actionable_error_message(403, None)
        assert "Access forbidden" in msg
        assert "authentication credentials" in suggestion.lower()

    def test_http_404(self):
        """Test 404 not found error."""
        msg, suggestion = get_actionable_error_message(404, None)
        assert "Resource not found" in msg
        assert "alternative resolvers" in suggestion.lower()

    def test_http_429(self):
        """Test 429 rate limit error."""
        msg, suggestion = get_actionable_error_message(429, None)
        assert "Rate limit exceeded" in msg
        assert "rate limiting" in suggestion.lower()

    def test_http_500(self):
        """Test 500 server error."""
        msg, suggestion = get_actionable_error_message(500, None)
        assert "Server error" in msg
        assert "retry" in suggestion.lower()

    def test_http_502_503(self):
        """Test 502/503 service unavailable."""
        msg502, suggestion502 = get_actionable_error_message(502, None)
        msg503, suggestion503 = get_actionable_error_message(503, None)
        assert "temporarily unavailable" in msg502.lower()
        assert "temporarily unavailable" in msg503.lower()
        assert "exponential backoff" in suggestion502.lower()

    def test_http_504(self):
        """Test 504 gateway timeout."""
        msg, suggestion = get_actionable_error_message(504, None)
        assert "Gateway timeout" in msg
        assert "timeout" in suggestion.lower()

    def test_generic_http_error(self):
        """Test generic HTTP error."""
        msg, suggestion = get_actionable_error_message(418, None)  # I'm a teapot
        assert "HTTP error 418" in msg
        assert "network connectivity" in suggestion.lower()

    def test_robots_disallowed(self):
        """Test robots.txt blocked error."""
        msg, suggestion = get_actionable_error_message(None, "robots_disallowed")
        assert "robots.txt" in msg.lower()
        assert "robots.txt" in suggestion.lower()

    def test_max_bytes_header(self):
        """Test max bytes exceeded error."""
        msg, suggestion = get_actionable_error_message(None, "max_bytes_header")
        assert "maximum size limit" in msg.lower()
        assert "max_bytes" in suggestion.lower()

    def test_domain_max_bytes(self):
        """Test domain bandwidth budget exceeded."""
        msg, suggestion = get_actionable_error_message(None, "domain_max_bytes")
        assert "bandwidth budget exceeded" in msg.lower()
        assert "domain_bytes_budget" in suggestion.lower()

    def test_request_exception(self):
        """Test general request exception."""
        msg, suggestion = get_actionable_error_message(None, "request_exception")
        assert "Network request failed" in msg
        assert "network connectivity" in suggestion.lower()

    def test_timeout(self):
        """Test timeout error."""
        msg, suggestion = get_actionable_error_message(None, "timeout")
        assert "timed out" in msg.lower()
        assert "timeout value" in suggestion.lower()

    def test_connection_error(self):
        """Test connection error."""
        msg, suggestion = get_actionable_error_message(None, "connection_error")
        assert "establish connection" in msg.lower()
        assert "network connectivity" in suggestion.lower()

    def test_circuit_breaker_errors(self):
        """Test circuit breaker open errors."""
        msg1, suggestion1 = get_actionable_error_message(None, "resolver_breaker_open")
        msg2, suggestion2 = get_actionable_error_message(None, "domain_breaker_open")
        assert "circuit breaker" in msg1.lower()
        assert "circuit breaker" in msg2.lower()
        assert "cooldown" in suggestion1.lower()
        assert "cooldown" in suggestion2.lower()

    def test_pdf_corruption_errors(self):
        """Test PDF corruption detection errors."""
        msg1, suggestion1 = get_actionable_error_message(None, "pdf_too_small")
        msg2, suggestion2 = get_actionable_error_message(None, "html_tail_detected")
        msg3, suggestion3 = get_actionable_error_message(None, "pdf_eof_missing")
        assert "too small" in msg1.lower()
        assert "html" in msg2.lower()
        assert "eof" in msg3.lower()
        for sugg in [suggestion1, suggestion2, suggestion3]:
            assert "alternative" in sugg.lower() or "corrupted" in sugg.lower()

    def test_generic_fallback(self):
        """Test generic fallback for unknown errors."""
        msg, suggestion = get_actionable_error_message(None, "unknown_reason")
        assert "Download failed" in msg
        assert "alternative resolvers" in suggestion.lower()

    def test_with_url_parameter(self):
        """Test that URL parameter is accepted (even if not used in current implementation)."""
        msg, suggestion = get_actionable_error_message(
            404, None, url="https://example.org/test.pdf"
        )
        assert "Resource not found" in msg


class TestLogDownloadFailure:
    """Test log_download_failure function."""

    @patch("DocsToKG.ContentDownload.errors.get_actionable_error_message")
    def test_log_with_all_parameters(self, mock_get_error):
        """Test logging with all parameters provided."""
        mock_get_error.return_value = ("Error message", "Suggestion")
        logger = Mock(spec=logging.Logger)

        log_download_failure(
            logger,
            "https://example.org/test.pdf",
            "W12345",
            http_status=403,
            reason_code="request_exception",
            error_details="Connection refused",
            exception=ValueError("Test exception"),
        )

        # Verify get_actionable_error_message was called
        mock_get_error.assert_called_once_with(
            403, "request_exception", "https://example.org/test.pdf"
        )

        # Verify error log was called
        assert logger.error.called
        error_call = logger.error.call_args
        assert "Error message" in str(error_call)

        # Verify info log was called for suggestion
        assert logger.info.called
        info_call = logger.info.call_args
        assert "Suggestion" in str(info_call)

    @patch("DocsToKG.ContentDownload.errors.get_actionable_error_message")
    def test_log_minimal_parameters(self, mock_get_error):
        """Test logging with minimal parameters."""
        mock_get_error.return_value = ("Error message", None)
        logger = Mock(spec=logging.Logger)

        log_download_failure(logger, "https://example.org/test.pdf", "W12345")

        # Verify error log was called
        assert logger.error.called

        # Verify info log was NOT called (no suggestion)
        assert not logger.info.called

    @patch("DocsToKG.ContentDownload.errors.get_actionable_error_message")
    def test_log_with_exception(self, mock_get_error):
        """Test that exception details are included in log."""
        mock_get_error.return_value = ("Error message", "Suggestion")
        logger = Mock(spec=logging.Logger)

        test_exc = ValueError("Test exception")
        log_download_failure(logger, "https://example.org/test.pdf", "W12345", exception=test_exc)

        # Check that exception information is in the logged data
        error_call_args = logger.error.call_args
        log_entry = error_call_args[1]["extra"]["extra_fields"]
        assert log_entry["exception_type"] == "ValueError"
        assert log_entry["exception_message"] == "Test exception"


class TestFormatDownloadSummary:
    """Test format_download_summary function."""

    def test_basic_summary(self):
        """Test basic summary generation."""
        summary = format_download_summary(total_attempts=100, successes=85, failures_by_reason={})
        assert "Total attempts: 100" in summary
        assert "Successes: 85 (85.0%)" in summary
        assert "Failures: 15 (15.0%)" in summary

    def test_summary_with_failures(self):
        """Test summary with failure reasons."""
        summary = format_download_summary(
            total_attempts=100,
            successes=85,
            failures_by_reason={"timeout": 10, "http_error": 5},
        )
        assert "timeout: 10 occurrences" in summary
        assert "http_error: 5 occurrences" in summary
        assert "66.7% of failures" in summary
        assert "33.3% of failures" in summary

    def test_summary_with_recommendations(self):
        """Test that recommendations are included."""
        summary = format_download_summary(
            total_attempts=100, successes=85, failures_by_reason={"timeout": 10}
        )
        assert "Recommendations:" in summary
        assert "timeout:" in summary

    def test_empty_failures(self):
        """Test summary with no failures."""
        summary = format_download_summary(total_attempts=100, successes=100, failures_by_reason={})
        assert "Total attempts: 100" in summary
        assert "Successes: 100 (100.0%)" in summary
        assert "Failures: 0 (0.0%)" in summary

    def test_zero_attempts(self):
        """Test summary with zero attempts."""
        summary = format_download_summary(total_attempts=0, successes=0, failures_by_reason={})
        assert "Total attempts: 0" in summary
        assert "Successes: 0 (0.0%)" in summary

    def test_many_failure_reasons(self):
        """Test that only top 5 failure reasons are shown."""
        failures = {f"reason_{i}": 10 - i for i in range(10)}
        summary = format_download_summary(
            total_attempts=100, successes=45, failures_by_reason=failures
        )
        # Should show top 5
        assert "reason_0" in summary
        assert "reason_4" in summary
        # Should not show beyond top 5 in the main list
        lines = summary.split("\n")
        reason_lines = [line for line in lines if "reason_" in line and "occurrence" in line]
        assert len(reason_lines) == 5

    def test_recommendations_limited_to_top_3(self):
        """Test that recommendations are limited to top 3 reasons."""
        failures = {f"timeout": 30, "http_error": 20, "connection_error": 10, "other": 5}
        summary = format_download_summary(
            total_attempts=100, successes=35, failures_by_reason=failures
        )
        recommendation_section = (
            summary.split("Recommendations:")[1] if "Recommendations:" in summary else ""
        )
        # Count recommendation lines (those with '-' prefix)
        rec_lines = [
            line for line in recommendation_section.split("\n") if line.strip().startswith("-")
        ]
        assert len(rec_lines) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
