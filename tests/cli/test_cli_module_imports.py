from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from typing import Iterable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

CLI_MODULE_EXPECTATIONS: dict[str, tuple[str, ...]] = {
    "DocsToKG.ContentDownload.cli_orchestrator": ("app",),
    "DocsToKG.ContentDownload.cli_v2": ("app",),
    "DocsToKG.ContentDownload.cli_breakers": ("install_breaker_cli",),
    "DocsToKG.ContentDownload.cli_breaker_advisor": ("install_breaker_advisor_cli",),
    "DocsToKG.ContentDownload.cli_telemetry": ("install_telemetry_cli",),
    "DocsToKG.ContentDownload.cli_telemetry_summary": ("summarize", "load_one"),
    "DocsToKG.ContentDownload.cli_config": ("register_config_commands",),
}

CLI_MODULE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "DocsToKG.ContentDownload.cli_orchestrator": ("typer",),
    "DocsToKG.ContentDownload.cli_v2": ("typer",),
    "DocsToKG.ContentDownload.cli_config": ("typer", "pydantic"),
}


@pytest.mark.parametrize("module_name", sorted(CLI_MODULE_EXPECTATIONS))
def test_cli_module_imports_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    expected = CLI_MODULE_EXPECTATIONS[module_name]

    for dependency in CLI_MODULE_DEPENDENCIES.get(module_name, ()):  # pragma: no branch - small list
        pytest.importorskip(dependency)

    with monkeypatch.context() as ctx:
        ctx.delitem(sys.modules, module_name, raising=False)
        ctx.delitem(sys.modules, "DocsToKG.ContentDownload.ratelimit", raising=False)

        original_import = builtins.__import__

        def fake_import(
            name: str,
            globals: dict | None = None,
            locals: dict | None = None,
            fromlist: tuple | None = None,
            level: int = 0,
        ):
            if name.startswith("pyrate_limiter"):
                raise ModuleNotFoundError("No module named 'pyrate_limiter'")
            return original_import(name, globals, locals, fromlist or (), level)

        ctx.setattr(builtins, "__import__", fake_import)
        module = importlib.import_module(module_name)

    _assert_exports(module, expected)

    # Reload ratelimit with the real dependency for other tests
    try:
        pytest.importorskip("httpx")
    except pytest.skip.Exception:
        pass
    else:
        sys.modules.pop("DocsToKG.ContentDownload.ratelimit", None)
        importlib.import_module("DocsToKG.ContentDownload.ratelimit")
    sys.modules.pop(module_name, None)


def _assert_exports(module: object, expected: Iterable[str]) -> None:
    for attr in expected:
        assert hasattr(module, attr), f"Module {module.__name__} should export {attr}"
