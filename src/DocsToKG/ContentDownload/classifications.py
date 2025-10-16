"""Classification enums shared across the content download pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class Classification(Enum):
    """Canonical classification codes for download outcomes."""

    PDF = "pdf"
    HTML = "html"
    MISS = "miss"
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
            return cls.MISS
        text = str(value).strip().lower()
        if not text:
            return cls.MISS

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
        return cls.MISS


PDF_LIKE = frozenset({Classification.PDF, Classification.CACHED})

__all__ = ("Classification", "PDF_LIKE")
