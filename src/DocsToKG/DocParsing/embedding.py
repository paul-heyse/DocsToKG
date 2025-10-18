"""Compatibility shim and CLI wrapper for the embedding stage.

This module re-exports the public API from :mod:`DocsToKG.DocParsing._embedding`
so existing imports work unchanged. When executed as a script it forwards to
the unified DocParsing CLI, maintaining support for the historical
``python DocsToKG/DocParsing/embedding.py`` entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent.parent
    cleaned: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() == script_dir:
                continue
        except Exception:
            pass
        cleaned.append(entry)
    if str(package_root) not in cleaned:
        cleaned.insert(0, str(package_root))
    sys.path = cleaned
    __package__ = "DocsToKG.DocParsing"

from DocsToKG.DocParsing.core import main as _core_main
from DocsToKG.DocParsing.env import (
    data_chunks,
    data_doctags,
    data_html,
    data_pdfs,
    data_vectors,
    detect_data_root,
)

from . import _embedding as _backend  # type: ignore[import]
from ._embedding import *  # noqa: F401,F403 - re-export legacy symbols

__all__ = list(getattr(_backend, "__all__", []))


def _format_argv(argv: Iterable[str]) -> list[str]:
    """Return a normalised list of CLI arguments."""

    return list(argv)


def _script_main(argv: Sequence[str] | None = None) -> int:
    """Execute the unified DocParsing CLI while preserving legacy semantics."""

    args = _format_argv(argv if argv is not None else sys.argv[1:])
    root = detect_data_root()
    data_doctags(root, ensure=True)
    data_chunks(root, ensure=True)
    data_vectors(root, ensure=True)
    data_html(root, ensure=True)
    data_pdfs(root, ensure=True)
    return _core_main(["embed", *args])


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    sys.exit(_script_main())
