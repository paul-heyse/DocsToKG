"""Static fake modules for content download tests.

This package complements :mod:`tests.docparsing.fake_deps` by providing
lightweight shims for optional dependencies that appear only in the content
download integration tests. The modules defined here are importable so both
pytest and mypy can observe the same API surface without relying on dynamic
``ModuleType`` hacks inside the test files themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent

__all__ = ["PACKAGE_ROOT"]
