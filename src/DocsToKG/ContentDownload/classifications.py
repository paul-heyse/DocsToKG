"""Classification enums shared across the content download pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Optional, Union


class Classification(StrEnum):
    """Canonical classification codes for download outcomes."""

    PDF = "pdf"
    PDF_UNKNOWN = "pdf_unknown"
    PDF_CORRUPT = "pdf_corrupt"
    HTML = "html"
    MISS = "miss"
    HTTP_ERROR = "http_error"
    CACHED = "cached"
    EXISTS = "exists"
    SKIPPED = "skipped"
    HTML_TOO_LARGE = "html_too_large"
    PAYLOAD_TOO_LARGE = "payload_too_large"

    @classmethod
    def from_wire(cls, value: Union[str, "Classification", None]) -> Optional["Classification"]:
        """Return the enum member when ``value`` matches a known code."""

        if value is None:
            return None
        if isinstance(value, cls):
            return value
        text = value.strip().lower()
        if not text:
            return None
        try:
            return cls(text)
        except ValueError:
            return None


PDF_LIKE = frozenset({Classification.PDF, Classification.PDF_UNKNOWN})

__all__ = ("Classification", "PDF_LIKE")
