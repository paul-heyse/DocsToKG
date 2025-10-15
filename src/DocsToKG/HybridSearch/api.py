"""Compatibility shim for the legacy `api` module."""

from __future__ import annotations

from .service import HybridSearchAPI, HybridSearchService, RequestValidationError
from .types import HybridSearchRequest

__all__ = [
    "HybridSearchAPI",
    "HybridSearchService",
    "HybridSearchRequest",
    "RequestValidationError",
]

