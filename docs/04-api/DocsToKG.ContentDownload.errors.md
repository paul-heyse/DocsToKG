# 1. Module: errors

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.errors``.

## 1. Overview

Enhanced error handling and actionable error messages for content downloads.

This module provides structured error messages with diagnostic information and
suggested remediation steps to help users troubleshoot download failures.

## 2. Functions

### `get_actionable_error_message(http_status, reason_code, url)`

Generate user-friendly error message with actionable suggestions.

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

### `log_download_failure(logger, url, work_id, http_status, reason_code, error_details, exception)`

Log download failure with structured context and actionable suggestions.

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

### `format_download_summary(total_attempts, successes, failures_by_reason)`

Format a human-readable download summary with recommendations.

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

### `__post_init__(self)`

*No documentation available.*

## 3. Classes

### `DownloadError`

Structured download error with diagnostic information.

### `NetworkError`

Raised when network-related download failures occur.

### `ContentPolicyError`

Raised when content policy violations prevent downloads.

### `RateLimitError`

Raised when rate limiting prevents downloads.
