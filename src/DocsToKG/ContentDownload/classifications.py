"""Classification enums shared across the content download pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Union


class Classification(Enum):
    """Canonical classification codes for download outcomes."""

    PDF = "pdf"
    HTML = "html"
    MISS = "miss"
    UNKNOWN = "unknown"
    HTTP_ERROR = "http_error"
    CACHED = "cached"
    SKIPPED = "skipped"
    HTML_TOO_LARGE = "html_too_large"
    PAYLOAD_TOO_LARGE = "payload_too_large"

    @classmethod
    def from_wire(cls, value: Union[str, "Classification", None]) -> "Classification":
        """Return the enum member when ``value`` matches a known code."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        if not text:
            return cls.UNKNOWN

        legacy_map = {
            "pdf_unknown": cls.PDF,
            "pdf_corrupt": cls.MISS,
            "request_error": cls.HTTP_ERROR,
            "exists": cls.CACHED,
        }
        if text in legacy_map:
            return legacy_map[text]

        for member in cls:
            if member.value == text:
                return member
        return cls.UNKNOWN


PDF_LIKE = frozenset({Classification.PDF, Classification.CACHED})


class ReasonCode(Enum):
    """Machine-readable reason taxonomy for download outcomes."""

    UNKNOWN = "unknown"
    PDF_SNIFF_UNKNOWN = "pdf_sniff_unknown"
    PDF_TOO_SMALL = "pdf_too_small"
    HTML_TAIL_DETECTED = "html_tail_detected"
    PDF_EOF_MISSING = "pdf_eof_missing"
    ROBOTS_DISALLOWED = "robots_disallowed"
    UNEXPECTED_304 = "unexpected_304"
    RESUME_COMPLETE = "resume_complete"
    ALREADY_DOWNLOADED = "already_downloaded"
    CONDITIONAL_CACHE_INVALID = "conditional_cache_invalid"
    CONDITIONAL_NOT_MODIFIED = "conditional_not_modified"
    MAX_BYTES_HEADER = "max_bytes_header"
    MAX_BYTES_STREAM = "max_bytes_stream"
    HEAD_PRECHECK_FAILED = "head_precheck_failed"
    HTTP_STATUS = "http_status"
    REQUEST_EXCEPTION = "request_exception"
    RESOLVER_MISSING = "resolver_missing"
    RESOLVER_DISABLED = "resolver_disabled"
    RESOLVER_NOT_APPLICABLE = "resolver_not_applicable"
    DUPLICATE_URL = "duplicate_url"
    DUPLICATE_URL_GLOBAL = "duplicate_url_global"
    LIST_ONLY = "list_only"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    RESOLVER_BREAKER_OPEN = "resolver_breaker_open"
    DOMAIN_BREAKER_OPEN = "domain_breaker_open"

    @classmethod
    def from_wire(cls, value: Union[str, "ReasonCode", None]) -> "ReasonCode":
        """Return the matching enum member or ``UNKNOWN``."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        if not text:
            return cls.UNKNOWN
        for member in cls:
            if member.value == text:
                return member
        return cls.UNKNOWN


__all__ = ("Classification", "PDF_LIKE", "ReasonCode")
