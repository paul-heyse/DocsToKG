"""
Canonical Exception Types for Download Pipeline

Thin signal types for prepare/stream/finalize functions.
Raised to signal skip (robots/policy) or error conditions;
the pipeline catches and converts to DownloadOutcome.

These exceptions keep function signatures pure (return types stable)
while allowing clean short-circuit logic.
"""

from __future__ import annotations

from typing import Optional

from .types import ReasonCode


class SkipDownload(Exception):
    """
    Raise when a download should be skipped without error.

    Caught by the pipeline and converted to:
        DownloadOutcome(ok=False, classification="skip", reason=reason)

    Common reasons:
        - "robots": robots.txt disallowed
        - "policy-type": content type not in allowed list
        - "policy-size": payload too large

    Example:
        def prepare_candidate_download(plan, ...):
            if not allowed_by_robots(plan.url):
                raise SkipDownload("robots")
            return plan
    """

    def __init__(self, reason: ReasonCode, message: Optional[str] = None) -> None:
        """
        Initialize skip signal.

        Args:
            reason: Normalized skip reason code
            message: Optional human-readable message
        """
        self.reason = reason
        super().__init__(message or f"Download skipped: {reason}")


class DownloadError(Exception):
    """
    Raise when an unrecoverable error occurs during download.

    Caught by the pipeline and converted to:
        DownloadOutcome(ok=False, classification="error", reason=reason)

    Common reasons:
        - "conn-error": network connection failure
        - "timeout": request timeout
        - "too-large": content exceeded max_bytes
        - "unexpected-ct": content type mismatch

    Unlike SkipDownload (transient/policy), this signals a true error.

    Example:
        def stream_candidate_payload(plan, ...):
            if bytes_so_far > max_bytes:
                raise DownloadError("too-large")
            return result
    """

    def __init__(self, reason: ReasonCode, message: Optional[str] = None) -> None:
        """
        Initialize error signal.

        Args:
            reason: Normalized error reason code
            message: Optional human-readable message
        """
        self.reason = reason
        super().__init__(message or f"Download error: {reason}")
