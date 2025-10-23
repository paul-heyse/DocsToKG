"""Public error exports for ContentDownload."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from DocsToKG.ContentDownload.errors.tenacity_policies import (
    OperationType,
    create_contextual_retry_policy,
)

__all__ = [
    "DownloadError",
    "NetworkError",
    "ContentPolicyError",
    "RateLimitError",
    "get_actionable_error_message",
    "log_download_failure",
    "OperationType",
    "create_contextual_retry_policy",
]


def _load_core_module() -> ModuleType:
    """Load the legacy ``errors.py`` module and memoise the result."""

    module_name = "DocsToKG.ContentDownload._errors_core"
    if module_name in sys.modules:
        cached = sys.modules[module_name]
        assert isinstance(cached, ModuleType)
        return cached

    module_path = Path(__file__).resolve().parent.parent / "errors.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise ImportError(f"Unable to load error module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_core = _load_core_module()

for _name in (
    "DownloadError",
    "NetworkError",
    "ContentPolicyError",
    "RateLimitError",
    "get_actionable_error_message",
    "log_download_failure",
):
    _value = getattr(_core, _name)
    if hasattr(_value, "__module__"):
        _value.__module__ = __name__
    globals()[_name] = _value

