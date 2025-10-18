from __future__ import annotations

from typing import Any

from .base import SerializationResult

__all__ = ["create_ser_result"]


def create_ser_result(*, text: str, span_source: Any | None = None) -> SerializationResult:
    return SerializationResult(text=text, span_source=span_source)
