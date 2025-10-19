"""Minimal pyalex fake used by the content download tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from . import config as config  # re-export module for attribute access

__all__ = ["Topics", "Works", "config"]


@dataclass
class Topics:
    """Stub topic collection returning canned responses."""

    data: Dict[str, Any] | None = None

    def topics(self, **_kwargs: Any) -> Dict[str, Any]:
        return self.data or {"results": []}


@dataclass
class Works:
    """Stub work collection returning canned responses."""

    data: Dict[str, Any] | None = None

    def works(self, **_kwargs: Any) -> Dict[str, Any]:
        return self.data or {"results": []}

