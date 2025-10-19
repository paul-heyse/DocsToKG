"""Run-all CLI regression tests."""

from __future__ import annotations

import argparse
import importlib
import types
from pathlib import Path
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


def test_doctags_forwards_vllm_wait_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DocTags CLI forwards custom vLLM wait timeout into the PDF pipeline."""

    module = _reload_core_cli()

    from DocsToKG.DocParsing import doctags as doctags_module

    captured: Dict[str, argparse.Namespace] = {}

    def fake_pdf_main(args: argparse.Namespace) -> int:
        captured["namespace"] = args
        return 0

    monkeypatch.setattr(doctags_module, "pdf_main", fake_pdf_main)

    data_root = tmp_path / "Data"
    pdf_dir = tmp_path / "pdf"
    out_dir = tmp_path / "doctags"

    data_root.mkdir()
    pdf_dir.mkdir()
    out_dir.mkdir()

    timeout_s = 432
    exit_code = module.doctags(
        [
            "--mode",
            "pdf",
            "--data-root",
            str(data_root),
            "--in-dir",
            str(pdf_dir),
            "--out-dir",
            str(out_dir),
            "--vllm-wait-timeout",
            str(timeout_s),
        ]
    )

    assert exit_code == 0
    assert "namespace" in captured
    assert captured["namespace"].vllm_wait_timeout == timeout_s
