from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

pytest.importorskip("httpx")


def test_orchestrator_reexports() -> None:
    orchestrator = importlib.import_module("DocsToKG.ContentDownload.orchestrator")

    for attr in [
        "Orchestrator",
        "OrchestratorConfig",
        "WorkQueue",
        "JobState",
        "JobResult",
        "KeyedLimiter",
        "host_key",
        "Worker",
    ]:
        assert hasattr(orchestrator, attr), f"Expected orchestrator to export {attr}"


def test_ratelimit_import_without_pyrate(monkeypatch: pytest.MonkeyPatch) -> None:
    target = "DocsToKG.ContentDownload.ratelimit"
    with monkeypatch.context() as ctx:
        ctx.delitem(sys.modules, target, raising=False)

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
        module = importlib.import_module(target)

    assert hasattr(module, "RateLimitExceeded")

    with pytest.raises(RuntimeError) as excinfo:
        module.Limiter([])  # type: ignore[arg-type]
    assert "pyrate-limiter is required" in str(excinfo.value)

    # Restore the real implementation for subsequent tests
    sys.modules.pop(target, None)
    importlib.import_module(target)
