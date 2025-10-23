# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.cli_utils",
#   "purpose": "Shared CLI building blocks for DocParsing subcommands.",
#   "sections": [
#     {
#       "id": "clioption",
#       "name": "CLIOption",
#       "anchor": "class-clioption",
#       "kind": "class"
#     },
#     {
#       "id": "build-subcommand",
#       "name": "build_subcommand",
#       "anchor": "function-build-subcommand",
#       "kind": "function"
#     },
#     {
#       "id": "preview-list",
#       "name": "preview_list",
#       "anchor": "function-preview-list",
#       "kind": "function"
#     },
#     {
#       "id": "merge-args",
#       "name": "merge_args",
#       "anchor": "function-merge-args",
#       "kind": "function"
#     },
#     {
#       "id": "scan-pdf-html",
#       "name": "scan_pdf_html",
#       "anchor": "function-scan-pdf-html",
#       "kind": "function"
#     },
#     {
#       "id": "directory-contains-suffixes",
#       "name": "directory_contains_suffixes",
#       "anchor": "function-directory-contains-suffixes",
#       "kind": "function"
#     },
#     {
#       "id": "detect-mode",
#       "name": "detect_mode",
#       "anchor": "function-detect-mode",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Shared CLI building blocks for DocParsing subcommands.

Controllers across DocTags, chunking, and embedding rely on the same option
shapes, directory heuristics, and preview formatting. This module packages
those behaviours into small helpers—such as declarative ``CLIOption`` records,
mode detection utilities, and directory scanners—so each stage can compose
parsers quickly while still delivering consistent UX and validation.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PDF_SUFFIXES: tuple[str, ...] = (".pdf",)
HTML_SUFFIXES: tuple[str, ...] = (".html", ".htm")

__all__ = [
    "CLIOption",
    "HTML_SUFFIXES",
    "PDF_SUFFIXES",
    "build_subcommand",
    "detect_mode",
    "directory_contains_suffixes",
    "merge_args",
    "preview_list",
    "scan_pdf_html",
]


@dataclass(frozen=True)
class CLIOption:
    """Declarative CLI argument specification used by ``build_subcommand``."""

    flags: Tuple[str, ...]
    kwargs: Dict[str, Any]


def build_subcommand(
    parser: argparse.ArgumentParser, options: Sequence[CLIOption]
) -> argparse.ArgumentParser:
    """Attach CLI options described by ``options`` to ``parser``."""

    for option in options:
        parser.add_argument(*option.flags, **option.kwargs)
    return parser


def preview_list(items: List[str], limit: int = 5) -> List[str]:
    """Return a truncated preview list with remainder hint."""

    if len(items) <= limit:
        return list(items)
    preview = list(items[:limit])
    preview.append(f"... (+{len(items) - limit} more)")
    return preview


def merge_args(parser: argparse.ArgumentParser, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Merge override values into the default parser namespace."""

    base = parser.parse_args([])
    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    return base


def scan_pdf_html(input_dir: Path) -> tuple[bool, bool]:
    """Return booleans indicating whether PDFs or HTML files exist beneath ``input_dir``."""

    has_pdf = False
    has_html = False

    if not input_dir.exists():
        return has_pdf, has_html

    for root, _dirs, files in os.walk(input_dir):
        if not files:
            continue
        for name in files:
            lower = name.lower()
            if not has_pdf and lower.endswith(PDF_SUFFIXES):
                has_pdf = True
            elif not has_html and lower.endswith(HTML_SUFFIXES):
                has_html = True
            if has_pdf and has_html:
                return has_pdf, has_html
    return has_pdf, has_html


def directory_contains_suffixes(directory: Path, suffixes: tuple[str, ...]) -> bool:
    """Return True when ``directory`` contains at least one file ending with ``suffixes``."""

    if not directory.exists():
        return False
    suffixes_lower = tuple(s.lower() for s in (suffixes or ()))
    for _root, _dirs, files in os.walk(directory):
        if not files:
            continue
        for name in files:
            if name.lower().endswith(suffixes_lower):
                return True
    return False


def detect_mode(input_dir: Path) -> str:
    """Infer conversion mode based on the contents of ``input_dir``."""

    if not input_dir.exists():
        raise ValueError(f"Cannot auto-detect mode in {input_dir}: directory not found")

    has_pdf, has_html = scan_pdf_html(input_dir)
    if has_pdf and not has_html:
        return "pdf"
    if has_html and not has_pdf:
        return "html"
    if has_pdf and has_html:
        raise ValueError(f"Cannot auto-detect mode in {input_dir}: found both PDF and HTML files")
    raise ValueError(f"Cannot auto-detect mode in {input_dir}: no PDF or HTML files found")
