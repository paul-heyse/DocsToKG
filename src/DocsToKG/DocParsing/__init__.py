"""High-level facade exposing CLI entry points for DocsToKG DocParsing."""

from __future__ import annotations

from . import pipelines as _pipelines

__all__ = [
    "pipelines",
    "pdf_build_parser",
    "pdf_parse_args",
    "pdf_main",
    "html_build_parser",
    "html_parse_args",
    "html_main",
]

pdf_build_parser = _pipelines.pdf_build_parser
pdf_parse_args = _pipelines.pdf_parse_args
pdf_main = _pipelines.pdf_main
html_build_parser = _pipelines.html_build_parser
html_parse_args = _pipelines.html_parse_args
html_main = _pipelines.html_main

pipelines = _pipelines
