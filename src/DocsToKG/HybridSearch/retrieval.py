"""Compatibility shim for the legacy `retrieval` module."""

from __future__ import annotations

from .service import ChannelResults, HybridSearchService, RequestValidationError

__all__ = ["ChannelResults", "HybridSearchService", "RequestValidationError"]

