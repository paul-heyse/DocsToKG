"""
Privacy and security helpers for Wayback telemetry.

This module provides utilities for masking sensitive data in telemetry events,
such as sanitizing URLs, removing query strings, and hashing identifiers.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional
from urllib.parse import urlparse, urlunparse

LOGGER = logging.getLogger(__name__)


def mask_url_query_string(url: str, placeholder: str = "[REDACTED]") -> str:
    """Remove query string from URL for privacy.

    Args:
        url: Original URL.
        placeholder: String to show instead of query string.

    Returns:
        URL with query string removed or replaced.
    """
    try:
        parsed = urlparse(url)
        if parsed.query:
            # Reconstruct URL without query string
            redacted = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
            if placeholder:
                return f"{redacted}?{placeholder}"
            return redacted
        return url
    except Exception as e:
        LOGGER.warning(f"Failed to mask query string in URL: {e}")
        return url


def hash_sensitive_value(value: str, prefix: str = "hash_") -> str:
    """Hash a sensitive value (e.g., internal ID) for privacy.

    Args:
        value: Sensitive value to hash.
        prefix: Prefix for the hash (e.g., "hash_", "id_").

    Returns:
        Hashed value with prefix (e.g., "hash_abc123...").
    """
    try:
        hash_digest = hashlib.sha256(value.encode()).hexdigest()[:12]
        return f"{prefix}{hash_digest}"
    except Exception as e:
        LOGGER.warning(f"Failed to hash sensitive value: {e}")
        return value


def sanitize_details_string(details: str, max_length: int = 256) -> str:
    """Sanitize a details/error string for logging (truncate and mask).

    Args:
        details: Original details string.
        max_length: Maximum length before truncation.

    Returns:
        Sanitized details string.
    """
    if not details:
        return ""

    # Truncate to max length
    if len(details) > max_length:
        details = details[: max_length - 3] + "..."

    # Mask any URLs in the details
    import re

    url_pattern = r"https?://[^\s]+"
    details = re.sub(url_pattern, lambda m: mask_url_query_string(m.group(0)), details)

    return details


def should_log_details(policy: str = "default") -> bool:
    """Determine if detailed logging should be enabled based on policy.

    Args:
        policy: Policy name ("strict", "default", "permissive").
                "strict": never log details/URLs
                "default": log URLs but mask query strings
                "permissive": log everything

    Returns:
        True if details should be logged.
    """
    return policy in ("default", "permissive")


def mask_event_for_logging(event: dict, policy: str = "default") -> dict:
    """Apply privacy policy to an event dict before logging.

    Args:
        event: Telemetry event to mask.
        policy: Privacy policy ("strict", "default", "permissive").

    Returns:
        New dict with masked fields.
    """
    masked = dict(event)

    # Field names that might contain sensitive data
    url_fields = [
        "original_url",
        "canonical_url",
        "archive_url",
        "emitted_url",
        "query_url",
        "discovered_pdf_url",
        "archive_html_url",
        "archive_pdf_url",
    ]
    detail_fields = ["details", "error", "error_type"]

    if policy == "strict":
        # Mask all URLs and details
        for field in url_fields:
            if field in masked:
                masked[field] = "[REDACTED_URL]"
        for field in detail_fields:
            if field in masked:
                masked[field] = "[REDACTED]"

    elif policy == "default":
        # Mask query strings and truncate details
        for field in url_fields:
            if field in masked and isinstance(masked[field], str):
                masked[field] = mask_url_query_string(masked[field])
        for field in detail_fields:
            if field in masked and isinstance(masked[field], str):
                masked[field] = sanitize_details_string(masked[field])

    # "permissive" policy: log everything as-is

    return masked
