# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.stubs",
#   "purpose": "Pytest coverage for docparsing stubs scenarios",
#   "sections": [
#     {
#       "id": "dependency-stubs",
#       "name": "dependency_stubs",
#       "anchor": "function-dependency-stubs",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DocParsing dependency stubs used by integration tests.

The dynamic `ModuleType` stubs that previously lived here now exist as a static
package under :mod:`tests.docparsing.fake_deps`. This helper simply mounts the
package on ``sys.path`` and registers each fake module under the production
module name so imports succeed when optional dependencies are absent.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Tuple

from tests.docparsing.fake_deps import PACKAGE_ROOT

__all__ = ["dependency_stubs"]

_FAKE_MODULES: Tuple[Tuple[str, str, bool], ...] = (
    ("sentence_transformers", "tests.docparsing.fake_deps.sentence_transformers", True),
    ("vllm", "tests.docparsing.fake_deps.vllm", True),
    ("tqdm", "tests.docparsing.fake_deps.tqdm", True),
    ("transformers", "tests.docparsing.fake_deps.transformers", True),
    ("docling_core", "tests.docparsing.fake_deps.docling_core", True),
    ("docling_core.transforms", "tests.docparsing.fake_deps.docling_core.transforms", True),
    (
        "docling_core.transforms.chunker",
        "tests.docparsing.fake_deps.docling_core.transforms.chunker",
        True,
    ),
    (
        "docling_core.transforms.chunker.base",
        "tests.docparsing.fake_deps.docling_core.transforms.chunker.base",
        True,
    ),
    (
        "docling_core.transforms.chunker.hybrid_chunker",
        "tests.docparsing.fake_deps.docling_core.transforms.chunker.hybrid_chunker",
        True,
    ),
    (
        "docling_core.transforms.chunker.hierarchical_chunker",
        "tests.docparsing.fake_deps.docling_core.transforms.chunker.hierarchical_chunker",
        True,
    ),
    (
        "docling_core.transforms.chunker.tokenizer",
        "tests.docparsing.fake_deps.docling_core.transforms.chunker.tokenizer",
        True,
    ),
    (
        "docling_core.transforms.chunker.tokenizer.huggingface",
        "tests.docparsing.fake_deps.docling_core.transforms.chunker.tokenizer.huggingface",
        True,
    ),
    (
        "docling_core.transforms.serializer",
        "tests.docparsing.fake_deps.docling_core.transforms.serializer",
        True,
    ),
    (
        "docling_core.transforms.serializer.base",
        "tests.docparsing.fake_deps.docling_core.transforms.serializer.base",
        True,
    ),
    (
        "docling_core.transforms.serializer.common",
        "tests.docparsing.fake_deps.docling_core.transforms.serializer.common",
        True,
    ),
    (
        "docling_core.transforms.serializer.markdown",
        "tests.docparsing.fake_deps.docling_core.transforms.serializer.markdown",
        True,
    ),
    ("docling_core.types", "tests.docparsing.fake_deps.docling_core.types", True),
    ("docling_core.types.doc", "tests.docparsing.fake_deps.docling_core.types.doc", True),
    (
        "docling_core.types.doc.document",
        "tests.docparsing.fake_deps.docling_core.types.doc.document",
        True,
    ),
)


def _ensure_sys_path(paths: Iterable[Path]) -> None:
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def dependency_stubs(dense_dim: int = 2560) -> None:
    """Install lightweight optional dependency stubs for integration tests."""

    tests_root = PACKAGE_ROOT.parent.parent
    project_root = tests_root.parent
    _ensure_sys_path([tests_root, project_root])

    fake_vllm = importlib.import_module("tests.docparsing.fake_deps.vllm")
    setattr(fake_vllm, "DEFAULT_DENSE_DIM", dense_dim)

    for module_name, fake_path, force in _FAKE_MODULES:
        if not force:
            if importlib.util.find_spec(module_name) is not None:
                continue
            if module_name in sys.modules:
                continue
        module = importlib.import_module(fake_path)
        sys.modules[module_name] = module
