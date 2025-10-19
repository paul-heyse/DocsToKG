"""Run-all CLI regression tests."""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Dict, List

import pytest

from DocsToKG.DocParsing.core import cli as core_cli
from DocsToKG.DocParsing.cli_errors import ChunkingCLIValidationError


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


def test_run_all_chunk_workers_zero_triggers_validation(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Zero chunk workers are forwarded and rejected by the chunk stage."""

    module = _reload_core_cli()
    stage_args: Dict[str, List[str]] = {}

    def _doctags(argv: List[str]) -> int:
        stage_args["doctags"] = list(argv)
        return 0

    stub_chunking = types.ModuleType("DocsToKG.DocParsing.chunking")

    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--log-level", default="INFO")
        parser.add_argument("--data-root")
        parser.add_argument("--in-dir")
        parser.add_argument("--out-dir")
        parser.add_argument("--workers", type=int, default=1)
        return parser

    def _main(args: argparse.Namespace) -> int:
        if args.workers < 1:
            raise ChunkingCLIValidationError(
                option="--workers", message="must be >= 1"
            )
        return 0

    stub_chunking.build_parser = _build_parser  # type: ignore[attr-defined]
    stub_chunking.main = _main  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "DocsToKG.DocParsing.chunking", stub_chunking)

    original_chunk = module.chunk

    def _chunk(argv: List[str]) -> int:
        stage_args["chunk"] = list(argv)
        return original_chunk(argv)

    monkeypatch.setattr(module, "doctags", _doctags)
    monkeypatch.setattr(module, "chunk", _chunk)

    data_root = tmp_path / "Data"
    doctags_dir = data_root / "DocTagsFiles"
    chunks_dir = data_root / "ChunkedDocTagFiles"
    chunks_dir.mkdir(parents=True)
    doctags_dir.mkdir(parents=True)

    exit_code = module.run_all(
        [
            "--data-root",
            str(data_root),
            "--doctags-out-dir",
            str(doctags_dir),
            "--chunk-out-dir",
            str(chunks_dir),
            "--chunk-workers",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "chunk" in stage_args
    chunk_args = stage_args["chunk"]
    assert "--workers" in chunk_args
    assert chunk_args[chunk_args.index("--workers") + 1] == "0"
    assert "--workers" in captured.err
    assert "must be >= 1" in captured.err
    assert "embed" not in stage_args
