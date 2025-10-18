"""High-level facade exposing consolidated DocParsing modules."""

from __future__ import annotations

from . import chunking, embedding
from . import core as _core
from . import doctags as _doctags
from . import formats as _formats
from . import token_profiles as _token_profiles

__all__ = [
    "core",
    "formats",
    "doctags",
    "chunking",
    "embedding",
    "token_profiles",
    "plan",
    "manifest",
    "pdf_build_parser",
    "pdf_parse_args",
    "pdf_main",
    "html_build_parser",
    "html_parse_args",
    "html_main",
]

core = _core
formats = _formats
doctags = _doctags
token_profiles = _token_profiles

plan = _core.plan
manifest = _core.manifest

pdf_build_parser = _doctags.pdf_build_parser
pdf_parse_args = _doctags.pdf_parse_args
pdf_main = _doctags.pdf_main
html_build_parser = _doctags.html_build_parser
html_parse_args = _doctags.html_parse_args
html_main = _doctags.html_main
