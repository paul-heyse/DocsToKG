"""
Lightweight stand-ins for optional DocParsing dependencies.

This package mirrors the public surface of a subset of third-party libraries so
integration tests and type checkers can rely on deterministic behaviour without
installing heavyweight extras (vLLM, docling, etc.).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PACKAGE_ROOT"]

PACKAGE_ROOT = Path(__file__).resolve().parent
