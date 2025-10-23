# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.errors",
#   "purpose": "Actionable error taxonomy and logging helpers for content downloads.",
#   "sections": [
#     {
#       "id": "downloaderror",
#       "name": "DownloadError",
#       "anchor": "class-downloaderror",
#       "kind": "class"
#     },
#     {
#       "id": "networkerror",
#       "name": "NetworkError",
#       "anchor": "class-networkerror",
#       "kind": "class"
#     },
#     {
#       "id": "contentpolicyerror",
#       "name": "ContentPolicyError",
#       "anchor": "class-contentpolicyerror",
#       "kind": "class"
#     },
#     {
#       "id": "ratelimiterror",
#       "name": "RateLimitError",
#       "anchor": "class-ratelimiterror",
#       "kind": "class"
#     },
#     {
#       "id": "get-actionable-error-message",
#       "name": "get_actionable_error_message",
#       "anchor": "function-get-actionable-error-message",
#       "kind": "function"
#     },
#     {
#       "id": "log-download-failure",
#       "name": "log_download_failure",
#       "anchor": "function-log-download-failure",
#       "kind": "function"
#     },
#     {
#       "id": "format-download-summary",
#       "name": "format_download_summary",
#       "anchor": "function-format-download-summary",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Actionable error taxonomy and logging helpers for content downloads.

Responsibilities
----------------
- Define lightweight exception types (``NetworkError``, ``ContentPolicyError``,
  ``RateLimitError``) that retain enough metadata for telemetry sinks and user
  feedback without leaking resolver internals.
- Provide the :class:`DownloadError` dataclass used by manifests and console
  summaries to capture failure context consistently across resolvers.
- Translate HTTP status codes and :class:`ReasonCode` values into user-friendly
  remediation hints via :func:`get_actionable_error_message`.
- Centralise structured logging through :func:`log_download_failure`, ensuring
  failures emit the fields required by observability dashboards.

Design Notes
------------
- The helpers avoid importing heavy dependencies so they can be used from
  within retry handlers and error paths without risk of further exceptions.
- Metadata dictionaries default to empty mappings to simplify serialisation
  into telemetry payloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = (
    "DownloadError",
    "NetworkError",
    "ContentPolicyError",
    "RateLimitError",
    "get_actionable_error_message",
    "log_download_failure",
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DownloadError:
    """Structured download error with diagnostic information."""

    error_type: str
    message: str
    url: str | None = None
    work_id: str | None = None
    http_status: int | None = None
    reason_code: str | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class NetworkError(Exception):
    """Raised when network-related download failures occur."""

    def __init__(
        self, message: str, *, url: str | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.url = url
        self.details = details or {}


class ContentPolicyError(Exception):
    """Raised when content policy violations prevent downloads."""

    def __init__(self, message: str, *, violation: str, policy: dict[str, Any] | None = None):
        super().__init__(message)
        self.violation = violation
        self.policy = policy or {}


class RateLimitError(Exception):
    """Raised when rate limiting prevents downloads."""

    def __init__(
        self,
        message: str,
        *,
        host: str | None = None,
        role: str | None = None,
        waited_ms: int | None = None,
        next_allowed_at: datetime | None = None,
        backend: str | None = None,
        mode: str | None = None,
        retry_after: float | None = None,
        domain: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.host = host or domain
        self.domain = domain or host
        self.role = role
        self.waited_ms = waited_ms
        self.next_allowed_at = next_allowed_at
        self.backend = backend
        self.mode = mode
        self.retry_after = retry_after
        self.details = details or {}


def get_actionable_error_message(
    http_status: int | None,
    reason_code: str | None,
    url: str | None = None,
) -> tuple[str, str | None]:
    """Generate user-friendly error message with actionable suggestions.

    Args:
        http_status: HTTP status code from failed request
        reason_code: Internal reason code describing failure
        url: URL that failed to download

    Returns:
        Tuple of (error_message, suggestion) where suggestion may be None

    Examples:
        >>> msg, suggestion = get_actionable_error_message(403, "request_exception")
        >>> print(msg)
        Access forbidden (HTTP 403)
        >>> print(suggestion)
        Check authentication credentials or access permissions for this resource
    """

    # HTTP status-based messages
    if http_status == 401:
        return (
            "Authentication required (HTTP 401)",
            "Add authentication credentials in resolver configuration or check API keys",
        )
    elif http_status == 403:
        return (
            "Access forbidden (HTTP 403)",
            "Check authentication credentials or access permissions for this resource",
        )
    elif http_status == 404:
        return (
            "Resource not found (HTTP 404)",
            "The requested document may have been moved or deleted. Try alternative resolvers.",
        )
    elif http_status == 429:
        return (
            "Rate limit exceeded (HTTP 429)",
            "Slow down requests or increase resolver rate limiting intervals. Check Retry-After header.",
        )
    elif http_status == 500:
        return (
            "Server error (HTTP 500)",
            "The upstream server encountered an error. Retry later or skip this resource.",
        )
    elif http_status == 502 or http_status == 503:
        return (
            f"Service temporarily unavailable (HTTP {http_status})",
            "The server is temporarily overloaded. Retry with exponential backoff.",
        )
    elif http_status == 504:
        return (
            "Gateway timeout (HTTP 504)",
            "The server took too long to respond. Try increasing timeout values or retry later.",
        )
    elif http_status and http_status >= 400:
        return (
            f"HTTP error {http_status}",
            "Check server logs or network connectivity for more details",
        )

    # Reason code-based messages
    if reason_code == "robots_disallowed":
        return (
            "Download blocked by robots.txt",
            "Respect the site's robots.txt. Consider disabling robot checking if you have permission.",
        )
    elif reason_code == "domain_disallowed_mime":
        return (
            "Content type not allowed by domain policy",
            "Update domain_content_rules to allow this MIME type or disable content policy",
        )
    elif reason_code == "request_exception":
        return (
            "Network request failed",
            "Check network connectivity, firewall rules, or proxy configuration",
        )
    elif reason_code == "timeout":
        return (
            "Request timed out",
            "Increase timeout value or check network latency. May indicate slow server.",
        )
    elif reason_code == "connection_error":
        return (
            "Failed to establish connection",
            "Check network connectivity, DNS resolution, or firewall rules",
        )
    # Legacy breaker error handling removed - now handled by pybreaker-based BreakerRegistry
    elif reason_code == "pdf_too_small":
        return (
            "Downloaded PDF is too small (likely corrupt)",
            "File may be truncated or corrupted. Try alternative URLs or resolvers.",
        )
    elif reason_code == "html_tail_detected":
        return (
            "HTML content detected in PDF tail",
            "File appears to be HTML masquerading as PDF. Try alternative resolvers.",
        )
    elif reason_code == "pdf_eof_missing":
        return (
            "PDF missing EOF marker (likely corrupt)",
            "File is incomplete or corrupted. Try re-downloading or alternative resolvers.",
        )

    # Generic fallback
    return (
        "Download failed",
        "Check logs for detailed error information. Try alternative resolvers or retry later.",
    )


def log_download_failure(
    logger: logging.Logger,
    url: str,
    work_id: str,
    http_status: int | None = None,
    reason_code: str | None = None,
    error_details: str | None = None,
    exception: Exception | None = None,
) -> None:
    """Log download failure with structured context and actionable suggestions.

    Args:
        logger: Logger instance to use for output
        url: URL that failed to download
        work_id: Work identifier for correlation
        http_status: HTTP status code if available
        reason_code: Internal reason code describing failure
        error_details: Additional error context
        exception: Original exception if available

    Examples:
        >>> log_download_failure(
        ...     LOGGER,
        ...     "https://example.org/paper.pdf",
        ...     "W123",
        ...     http_status=403,
        ...     reason_code="request_exception"
        ... )
    """

    error_msg, suggestion = get_actionable_error_message(http_status, reason_code, url)

    log_entry = {
        "url": url,
        "work_id": work_id,
        "http_status": http_status,
        "reason_code": reason_code,
        "error_message": error_msg,
    }

    if error_details:
        log_entry["details"] = error_details

    if exception:
        log_entry["exception_type"] = type(exception).__name__
        log_entry["exception_message"] = str(exception)
        if isinstance(exception, RateLimitError):
            if exception.host:
                log_entry["rate_limit_host"] = exception.host
            if exception.role:
                log_entry["rate_limit_role"] = exception.role
            if exception.backend:
                log_entry["rate_limit_backend"] = exception.backend
            if exception.mode:
                log_entry["rate_limit_mode"] = exception.mode
            if exception.waited_ms is not None:
                log_entry["rate_limit_waited_ms"] = exception.waited_ms
            if exception.retry_after is not None:
                log_entry["retry_after"] = exception.retry_after
            if exception.next_allowed_at is not None:
                log_entry["rate_limit_next_allowed"] = exception.next_allowed_at.isoformat()
            if exception.details:
                log_entry["rate_limit_details"] = exception.details

    logger.error("Download failed: %s", error_msg, extra={"extra_fields": log_entry})

    if suggestion:
        logger.info(
            "Suggestion: %s", suggestion, extra={"extra_fields": {"url": url, "work_id": work_id}}
        )


def format_download_summary(
    total_attempts: int,
    successes: int,
    failures_by_reason: dict[str, int],
) -> str:
    """Format a human-readable download summary with recommendations.

    Args:
        total_attempts: Total number of download attempts
        successes: Number of successful downloads
        failures_by_reason: Dictionary mapping reason codes to failure counts

    Returns:
        Formatted summary string with recommendations

    Examples:
        >>> summary = format_download_summary(
        ...     total_attempts=100,
        ...     successes=85,
        ...     failures_by_reason={"http_error": 10, "timeout": 5}
        ... )
        >>> print(summary)
        Download Summary:
        - Total attempts: 100
        - Successes: 85 (85.0%)
        - Failures: 15 (15.0%)

        Top failure reasons:
        1. http_error: 10 occurrences (66.7% of failures)
        2. timeout: 5 occurrences (33.3% of failures)

        Recommendations:
        - http_error: Check server logs or network connectivity for more details
        - timeout: Increase timeout value or check network latency. May indicate slow server.
    """

    failures = total_attempts - successes
    success_rate = (successes / total_attempts * 100) if total_attempts > 0 else 0
    failure_rate = (failures / total_attempts * 100) if total_attempts > 0 else 0

    lines = [
        "Download Summary:",
        f"- Total attempts: {total_attempts}",
        f"- Successes: {successes} ({success_rate:.1f}%)",
        f"- Failures: {failures} ({failure_rate:.1f}%)",
        "",
    ]

    if failures_by_reason:
        lines.append("Top failure reasons:")
        sorted_reasons = sorted(failures_by_reason.items(), key=lambda x: x[1], reverse=True)

        for idx, (reason, count) in enumerate(sorted_reasons[:5], 1):
            pct = (count / failures * 100) if failures > 0 else 0
            lines.append(f"{idx}. {reason}: {count} occurrences ({pct:.1f}% of failures)")

        lines.append("")
        lines.append("Recommendations:")

        for reason, count in sorted_reasons[:3]:
            _, suggestion = get_actionable_error_message(None, reason)
            if suggestion:
                lines.append(f"- {reason}: {suggestion}")

    return "\n".join(lines)
