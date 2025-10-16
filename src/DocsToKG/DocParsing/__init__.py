"""High-level facade exposing CLI entry points for DocsToKG DocParsing."""

from __future__ import annotations

import sys
import warnings
from types import ModuleType
from typing import Any, Dict

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


class _DeprecatedModule(ModuleType):
    """Proxy module that emits a deprecation warning on first attribute access."""

    def __init__(self, name: str, mapping: Dict[str, Any], message: str) -> None:
        super().__init__(name)
        self._mapping = mapping
        self._message = message
        self._warned = False
        self.__all__ = tuple(mapping.keys())

    def _emit_warning(self) -> None:
        if not self._warned:
            warnings.warn(self._message, DeprecationWarning, stacklevel=3)
            self._warned = True

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        if item in self._mapping:
            self._emit_warning()
            return self._mapping[item]
        raise AttributeError(item)


_pdf_module = _DeprecatedModule(
    __name__ + ".pdf_pipeline",
    {
        "build_parser": pdf_build_parser,
        "parse_args": pdf_parse_args,
        "main": pdf_main,
        "convert_one": _pipelines.pdf_convert_one,
        "list_pdfs": _pipelines.list_pdfs,
        "PdfTask": _pipelines.PdfTask,
        "PdfConversionResult": _pipelines.PdfConversionResult,
    },
    "DocsToKG.DocParsing.pdf_pipeline is deprecated; import DocsToKG.DocParsing.pipelines instead.",
)

_html_module = _DeprecatedModule(
    __name__ + ".html_pipeline",
    {
        "build_parser": html_build_parser,
        "parse_args": html_parse_args,
        "main": html_main,
        "convert_one": _pipelines.html_convert_one,
        "list_htmls": _pipelines.list_htmls,
        "HtmlTask": _pipelines.HtmlTask,
        "HtmlConversionResult": _pipelines.HtmlConversionResult,
    },
    "DocsToKG.DocParsing.html_pipeline is deprecated; import DocsToKG.DocParsing.pipelines instead.",
)

sys.modules.setdefault(__name__ + ".pdf_pipeline", _pdf_module)
sys.modules.setdefault(__name__ + ".html_pipeline", _html_module)
