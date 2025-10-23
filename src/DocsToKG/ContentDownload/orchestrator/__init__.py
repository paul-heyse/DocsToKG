"""Orchestration subpackage for ContentDownload."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

_ATTRIBUTE_EXPORTS: dict[str, tuple[str, str]] = {
    "Orchestrator": (".scheduler", "Orchestrator"),
    "OrchestratorConfig": (".scheduler", "OrchestratorConfig"),
    "WorkQueue": (".queue", "WorkQueue"),
    "JobState": (".models", "JobState"),
    "JobResult": (".models", "JobResult"),
    "KeyedLimiter": (".limits", "KeyedLimiter"),
    "host_key": (".limits", "host_key"),
    "Worker": (".workers", "Worker"),
}

_MODULE_EXPORTS: dict[str, str] = {
    "feature_flags": ".feature_flags",
    "limits": ".limits",
    "models": ".models",
    "queue": ".queue",
    "scheduler": ".scheduler",
    "workers": ".workers",
}

__all__ = sorted({*_ATTRIBUTE_EXPORTS, *_MODULE_EXPORTS})


def _load_module(name: str, module_path: str) -> ModuleType:
    module = importlib.import_module(f"{__name__}{module_path}")
    setattr(sys.modules[__name__], name, module)
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised via tests
    if name in _ATTRIBUTE_EXPORTS:
        module_path, attr_name = _ATTRIBUTE_EXPORTS[name]
        module = importlib.import_module(f"{__name__}{module_path}")
        value = getattr(module, attr_name)
        setattr(sys.modules[__name__], name, value)
        return value
    if name in _MODULE_EXPORTS:
        return _load_module(name, _MODULE_EXPORTS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - tooling helper
    return sorted(set(globals()) | set(__all__))
