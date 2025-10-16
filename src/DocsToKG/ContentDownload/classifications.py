"""Classification enums shared across the content download pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class Classification(Enum):
    """Canonical classification codes for download outcomes."""

    PDF = "pdf"
    PDF_UNKNOWN = "pdf_unknown"
    PDF_CORRUPT = "pdf_corrupt"
    HTML = "html"
    MISS = "miss"
    HTTP_ERROR = "http_error"
    REQUEST_ERROR = "request_error"
    CACHED = "cached"
    EXISTS = "exists"
    SKIPPED = "skipped"
    HTML_TOO_LARGE = "html_too_large"
    PAYLOAD_TOO_LARGE = "payload_too_large"

    @classmethod
    def from_wire(cls, value: Union[str, "Classification", None]) -> "Classification":
        """Return the enum member when ``value`` matches a known code."""

        if isinstance(value, cls):
            return value
        if value is None:
            return cls.MISS
        text = str(value).strip().lower()
        if not text:
            return cls.MISS
        for member in cls:
            if member.value == text:
                return member
        return cls.MISS


PDF_LIKE = frozenset({Classification.PDF, Classification.PDF_UNKNOWN, Classification.CACHED})

__all__ = ("Classification", "PDF_LIKE")
