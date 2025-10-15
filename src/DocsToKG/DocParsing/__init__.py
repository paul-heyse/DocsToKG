"""High-level facade exposing CLI entry points for DocsToKG DocParsing."""

from __future__ import annotations

import sys
from types import ModuleType

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

_pdf_module = ModuleType(__name__ + ".pdf_pipeline")
_pdf_module.__dict__.update(
    {
        "build_parser": pdf_build_parser,
        "parse_args": pdf_parse_args,
        "main": pdf_main,
        "convert_one": _pipelines.pdf_convert_one,
        "list_pdfs": _pipelines.list_pdfs,
        "PdfTask": _pipelines.PdfTask,
        "PdfConversionResult": _pipelines.PdfConversionResult,
    }
)

_html_module = ModuleType(__name__ + ".html_pipeline")
_html_module.__dict__.update(
    {
        "build_parser": html_build_parser,
        "parse_args": html_parse_args,
        "main": html_main,
        "convert_one": _pipelines.html_convert_one,
        "list_htmls": _pipelines.list_htmls,
        "HtmlTask": _pipelines.HtmlTask,
        "HtmlConversionResult": _pipelines.HtmlConversionResult,
    }
)

sys.modules.setdefault(__name__ + ".pdf_pipeline", _pdf_module)
sys.modules.setdefault(__name__ + ".html_pipeline", _html_module)
