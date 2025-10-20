"""Regression coverage for the deprecated DocParsing.schemas shim.

The canonical schema surface now lives in :mod:`DocsToKG.DocParsing.formats`.
This test ensures importing representative DocParsing modules no longer pulls in
the deprecated shim implicitly; doing so would reintroduce circular imports and
optional-dependency drift.
"""

from __future__ import annotations

import importlib
import sys

from tests.docparsing.stubs import dependency_stubs


def test_docparsing_modules_do_not_import_legacy_schema_module():
    """Ensure top-level DocParsing modules avoid the deprecated shim."""

    dependency_stubs()

    sys.modules.pop("DocsToKG.DocParsing.schemas", None)

    module_names = (
        "DocsToKG.DocParsing.formats",
    )

    for name in module_names:
        sys.modules.pop(name, None)
        importlib.import_module(name)

    assert "DocsToKG.DocParsing.schemas" not in sys.modules
