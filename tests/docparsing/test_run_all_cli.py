"""Run-all CLI regression tests."""

from __future__ import annotations

import importlib
import types
from typing import Dict, List

import pytest

from DocsToKG.DocParsing.core import cli as core_cli


def _reload_core_cli() -> types.ModuleType:
    """Reload the core CLI module to reset patched state between tests."""

    return importlib.reload(core_cli)


def test_run_all_forwards_sparsity_warn_threshold_pct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run-all orchestrator forwards sparsity threshold flags to the embed stage."""

    module = _reload_core_cli()
    call_order: List[str] = []
    stage_args: Dict[str, List[str]] = {}

    def _stage(name: str):
        def _runner(args: List[str]) -> int:
            call_order.append(name)
            stage_args[name] = args
            return 0

        return _runner

    monkeypatch.setattr(module, "doctags", _stage("doctags"))
    monkeypatch.setattr(module, "chunk", _stage("chunk"))
    monkeypatch.setattr(module, "embed", _stage("embed"))

    exit_code = module.run_all([
        "--sparsity-warn-threshold-pct",
        "12.5",
    ])

    assert exit_code == 0
    assert call_order == ["doctags", "chunk", "embed"]
    embed_args = stage_args["embed"]
    assert "--sparsity-warn-threshold-pct" in embed_args
    flag_index = embed_args.index("--sparsity-warn-threshold-pct")
    assert embed_args[flag_index + 1] == "12.5"
