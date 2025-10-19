# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_docstrings",
#   "purpose": "Pytest coverage for docparsing docstrings scenarios",
#   "sections": [
#     {
#       "id": "iter-python-modules",
#       "name": "_iter_python_modules",
#       "anchor": "function-iter-python-modules",
#       "kind": "function"
#     },
#     {
#       "id": "iter-defs",
#       "name": "_iter_defs",
#       "anchor": "function-iter-defs",
#       "kind": "function"
#     },
#     {
#       "id": "test-module-and-definitions-have-docstrings",
#       "name": "test_module_and_definitions_have_docstrings",
#       "anchor": "function-test-module-and-definitions-have-docstrings",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Enforce docstring coverage across DocParsing modules.

This file walks every Python module beneath `DocsToKG.DocParsing` to confirm that
modules, classes, and functions expose descriptive docstrings that power the
documentation tooling. It acts as a guardrail whenever new code is added or
existing helpers are refactored without updating their annotations.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# --- Globals ---

DOC_PARSING_ROOT = Path(__file__).resolve().parents[2] / "src" / "DocsToKG" / "DocParsing"
# --- Helper Functions ---


def _iter_python_modules(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and not path.name.startswith(".")
    ]


def _iter_defs(node: ast.AST) -> list[ast.AST]:
    """Yield all class and function definitions within *node* (including nested)."""

    defs: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append(child)
        defs.extend(_iter_defs(child))
    return defs


@pytest.mark.parametrize(
    "module_path",
    [
        path
        for path in _iter_python_modules(DOC_PARSING_ROOT)
        if path.name != "__init__.py" or path.parent.name != "testing"
    ],
)
# --- Test Cases ---


def test_module_and_definitions_have_docstrings(module_path: Path) -> None:
    source = module_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(module_path))

    module_doc = ast.get_docstring(module, clean=False)
    assert (
        module_doc and module_doc.strip()
    ), f"Missing module docstring in {module_path.relative_to(DOC_PARSING_ROOT)}"

    for definition in _iter_defs(module):
        doc = ast.get_docstring(definition, clean=False)
        assert doc and doc.strip(), (
            "Missing docstring in "
            f"{module_path.relative_to(DOC_PARSING_ROOT)}:{definition.lineno} "
            f"({getattr(definition, 'name', '<anonymous>')})"
        )
