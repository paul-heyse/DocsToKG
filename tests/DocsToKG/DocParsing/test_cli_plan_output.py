from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from DocsToKG.DocParsing.core import planning
from DocsToKG.DocParsing.core.cli import app


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner()


def _make_bucket(count: int, preview: list[str]) -> dict[str, Any]:
    return {"count": count, "preview": preview}


def test_plan_cli_json_output(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    doctags_plan = {
        "stage": "doctags",
        "mode": "html",
        "input_dir": "/tmp/html",
        "output_dir": "/tmp/doctags",
        "process": _make_bucket(2, ["doc-1", "doc-2"]),
        "skip": _make_bucket(1, ["doc-3"]),
        "notes": ["HTML sanitiser disabled"],
        "log_level": "DEBUG",
    }
    chunk_plan = {
        "stage": "chunk",
        "input_dir": "/tmp/doctags",
        "output_dir": "/tmp/chunks",
        "process": _make_bucket(3, ["doc-1"]),
        "skip": _make_bucket(0, []),
        "notes": [],
    }
    embed_plan = {
        "stage": "embed",
        "action": "generate",
        "chunks_dir": "/tmp/chunks",
        "vectors_dir": "/tmp/vectors",
        "process": _make_bucket(3, ["doc-1", "doc-2", "doc-3"]),
        "skip": _make_bucket(0, []),
        "notes": ["Vectors directory not found; outputs will be created during generation"],
        "vector_format": "jsonl",
    }

    captured_args: dict[str, list[str]] = {}

    def _capture(name: str, argv: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        captured_args[name] = list(argv)
        return payload

    monkeypatch.setattr(planning, "plan_doctags", lambda argv: _capture("doctags", argv, doctags_plan))
    monkeypatch.setattr(planning, "plan_chunk", lambda argv: _capture("chunk", argv, chunk_plan))
    monkeypatch.setattr(planning, "plan_embed", lambda argv: _capture("embed", argv, embed_plan))

    result = cli_runner.invoke(app, ["--output", "json"])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    assert payload["docparse"]["stages"][0]["stage"] == "doctags"
    assert payload["docparse"]["stages"][0]["buckets"]["process"]["count"] == 2
    assert payload["docparse"]["stages"][0]["notes"] == doctags_plan["notes"]
    assert payload["docparse"]["stages"][1]["stage"] == "chunk"
    assert payload["docparse"]["stages"][2]["vector_format"] == "jsonl"
    totals = payload["docparse"]["totals"]
    assert totals["doctags"]["process"] == 2
    assert totals["embed"]["process"] == 3

    assert "--output" not in captured_args["doctags"]
    assert "--output" not in captured_args["chunk"]
    assert "--output" not in captured_args["embed"]


def test_plan_cli_default_pretty(cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planning, "plan_doctags", lambda argv: {"stage": "doctags", "process": _make_bucket(0, [])})
    monkeypatch.setattr(planning, "plan_chunk", lambda argv: {"stage": "chunk", "process": _make_bucket(0, [])})
    monkeypatch.setattr(planning, "plan_embed", lambda argv: {"stage": "embed", "process": _make_bucket(0, [])})

    result = cli_runner.invoke(app, [])
    assert result.exit_code == 0
    assert "docparse all plan" in result.output
    assert "process 0" in result.output


def test_plan_cli_rejects_unknown_output(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["--output", "yaml"])
    assert result.exit_code != 0
    assert "Output must be either" in result.output
