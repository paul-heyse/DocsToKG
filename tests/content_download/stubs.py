"""Content download dependency stubs used by integration tests."""

from __future__ import annotations

import importlib
import sys
from typing import Tuple

from tests.docparsing.stubs import dependency_stubs as _docparsing_dependency_stubs

_CONTENT_FAKE_MODULES: Tuple[Tuple[str, str, bool], ...] = (
    ("pyalex", "tests.content_download.fakes.pyalex", True),
    ("pyalex.config", "tests.content_download.fakes.pyalex.config", True),
    ("pydantic_core", "tests.content_download.fakes.pydantic_core", True),
    (
        "pydantic_settings",
        "tests.content_download.fakes.pydantic_settings",
        True,
    ),
    (
        "docling_core.persistence",
        "tests.content_download.fakes.docling_core.persistence",
        True,
    ),
    (
        "docling_core.serializers",
        "tests.content_download.fakes.docling_core.serializers",
        True,
    ),
    ("pooch", "tests.content_download.fakes.pooch", True),
)


def dependency_stubs() -> None:
    """Install optional dependency stubs needed by content download tests."""

    _docparsing_dependency_stubs()

    for module_name, fake_path, force in _CONTENT_FAKE_MODULES:
        if not force and module_name in sys.modules:
            continue
        module = importlib.import_module(fake_path)
        sys.modules[module_name] = module
