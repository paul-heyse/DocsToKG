from __future__ import annotations

import ast
from pathlib import Path

import pytest


DOC_PARSING_ROOT = Path(__file__).resolve().parents[2] / "src" / "DocsToKG" / "DocParsing"


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


@pytest.mark.parametrize("module_path", _iter_python_modules(DOC_PARSING_ROOT))
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
