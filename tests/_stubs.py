# === NAVMAP v1 ===
# {
#   "module": "tests._stubs",
#   "purpose": "Pytest coverage for  stubs scenarios",
#   "sections": [
#     {
#       "id": "promote-simple-namespace-modules",
#       "name": "promote_simple_namespace_modules",
#       "anchor": "function-promote-simple-namespace-modules",
#       "kind": "function"
#     },
#     {
#       "id": "dependency-stubs",
#       "name": "dependency_stubs",
#       "anchor": "function-dependency-stubs",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Test stubs and utilities for DocParsing-related unit tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict
# --- Globals ---

__all__ = ["promote_simple_namespace_modules", "dependency_stubs"]
# --- Test Cases ---


def promote_simple_namespace_modules() -> None:
    """Convert SimpleNamespace test stubs in :data:`sys.modules` to real modules."""

    for module_name, module_obj in list(sys.modules.items()):
        if isinstance(module_obj, SimpleNamespace):
            promoted = ModuleType(module_name)
            promoted.__dict__.update(vars(module_obj))
            sys.modules[module_name] = promoted


def dependency_stubs(**stubs: Dict[str, Any]) -> None:
    """Install test stubs for optional dependencies."""

    for name, stub in stubs.items():
        if callable(stub):
            sys.modules[name] = stub()
        elif isinstance(stub, dict):
            sys.modules[name] = SimpleNamespace(**stub)
        else:
            sys.modules[name] = stub
