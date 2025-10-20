# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_settings_import_error",
#   "purpose": "Regression tests for settings import-time guidance.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Regression tests for settings import-time guidance."""

from __future__ import annotations

import builtins
import importlib
import sys
from unittest.mock import patch

import pytest


def test_missing_pyyaml_prompts_managed_environment_message() -> None:
    """ImportError should point contributors to the managed .venv bootstrap."""

    module_name = "DocsToKG.OntologyDownload.settings"
    original_module = sys.modules.pop(module_name, None)
    original_yaml = sys.modules.pop("yaml", None)
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, globals, locals, fromlist, level)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError) as excinfo:
                importlib.import_module(module_name)

        message = str(excinfo.value)
        assert "PyYAML is required for configuration parsing." in message
        assert "Ensure the project-managed .venv is set up" in message
        assert "bootstrap" in message
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        else:
            importlib.import_module(module_name)

        sys.modules.pop("yaml", None)
        if original_yaml is not None:
            sys.modules["yaml"] = original_yaml
