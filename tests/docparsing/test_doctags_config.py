"""Validation tests for :mod:`DocsToKG.DocParsing.doctags` configuration objects."""

from __future__ import annotations

import pytest

from DocsToKG.DocParsing.doctags import DoctagsCfg


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_doctags_cfg_rejects_out_of_range_gpu_memory(tmp_path, value):
    """``DoctagsCfg.finalize`` should reject GPU utilization outside the [0, 1] range."""

    cfg = DoctagsCfg(
        data_root=tmp_path,
        input=tmp_path / "input",
        output=tmp_path / "output",
        gpu_memory_utilization=value,
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.finalize()

    message = str(excinfo.value)
    assert "gpu_memory_utilization must be between 0.0 and 1.0 (inclusive)" in message
    assert f"received {float(value)}" in message
import importlib
import logging
import os
import sys
import types

import pytest


@pytest.mark.parametrize("cpu_count_value", [0, 1, 2, 3])
def test_doctags_workers_never_drop_below_one(
    monkeypatch: pytest.MonkeyPatch, cpu_count_value: int
) -> None:
    """DocTags workers derived from CPU count should never fall below one."""

    # Clear env overrides so defaults are exercised.
    for env_var in (
        "DOCSTOKG_DOCTAGS_WORKERS",
        "DOCSTOKG_DOCTAGS_PROFILE",
    ):
        monkeypatch.delenv(env_var, raising=False)

    original_cpu_count = os.cpu_count

    logging_utils_name = "DocsToKG.OntologyDownload.logging_utils"
    logging_stub = types.ModuleType(logging_utils_name)
    logging_stub.JSONFormatter = logging.Formatter
    monkeypatch.setitem(sys.modules, logging_utils_name, logging_stub)

    def _cpu_count_stub() -> int:
        return cpu_count_value

    monkeypatch.setattr("os.cpu_count", _cpu_count_stub)

    module_name = "DocsToKG.DocParsing.doctags"
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)

    try:
        cfg = module.DoctagsCfg.from_env()
        assert cfg.workers >= 1
        assert module.DEFAULT_WORKERS >= 1

        gpu_default_workers = module.PROFILE_PRESETS["gpu-default"]["workers"]
        gpu_max_workers = module.PROFILE_PRESETS["gpu-max"]["workers"]

        assert gpu_default_workers >= 1
        assert gpu_max_workers >= 1
    finally:
        # Restore the real CPU count and module state for other tests.
        monkeypatch.setattr("os.cpu_count", original_cpu_count)
        importlib.reload(module)
