"""Verify the lazy-import facade for DocsToKG.DocParsing behaves correctly.

These smoke tests confirm that the package-level lazy loader does not import
heavy submodules prematurely, that missing optional dependencies raise helpful
errors, and that successful imports are cached for reuse. This protects the CLI
start-up path and keeps optional extras truly optional.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


def test_import_docparsing_without_optional_dependencies(patcher):
    """Importing the facade should not eagerly import optional submodules."""

    sys.modules.pop("DocsToKG.DocParsing", None)
    sys.modules.pop("DocsToKG.DocParsing.chunking", None)
    module = __import__("DocsToKG.DocParsing", fromlist=["DocParsing"])

    assert "DocsToKG.DocParsing.chunking" not in sys.modules
    assert "core" not in module.__dict__
    assert "doctags" not in module.__dict__

    original_import = module._import_module

    def raising_import(name: str):
        if name == "DocsToKG.DocParsing.chunking":
            raise ModuleNotFoundError("No module named 'docling'", name="docling")
        return original_import(name)

    patcher.setattr(module, "_import_module", raising_import)

    with pytest.raises(ImportError) as excinfo:
        _ = module.chunking  # noqa: F841  (exercise lazy loader)

    assert "docling" in str(excinfo.value)
    assert "DocsToKG.DocParsing.chunking" in str(excinfo.value)
    assert "chunking" not in module.__dict__, "failed imports must not be cached"


def test_lazy_import_caches_modules(patcher):
    """Lazy imports should only invoke the importer once per submodule."""

    sys.modules.pop("DocsToKG.DocParsing", None)
    sys.modules.pop("DocsToKG.DocParsing.embedding", None)
    module = __import__("DocsToKG.DocParsing", fromlist=["DocParsing"])

    fake_embedding = ModuleType("DocsToKG.DocParsing.embedding")
    import_called = {}

    def tracking_import(name: str):
        import_called[name] = import_called.get(name, 0) + 1
        if name == "DocsToKG.DocParsing.embedding":
            return fake_embedding
        return __import__(name, fromlist=[name.rsplit(".", 1)[-1]])

    patcher.setattr(module, "_import_module", tracking_import)

    embedding_module = module.embedding
    assert embedding_module is module.embedding is fake_embedding
    assert import_called["DocsToKG.DocParsing.embedding"] == 1
