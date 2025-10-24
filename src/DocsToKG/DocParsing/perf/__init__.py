"""Performance monitoring helpers for DocParsing."""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Lazily expose the Typer application to avoid heavy imports during tests."""

    if name == "app":
        from DocsToKG.DocParsing.perf.cli import app as perf_app

        return perf_app
    raise AttributeError(name)
