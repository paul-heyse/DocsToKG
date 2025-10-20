# === NAVMAP v1 ===
# {
#   "module": "tests.docparsing.test_typer_cli",
#   "purpose": "Regression-spec for the Typer CLI wiring ensuring legacy behaviour parity.",
#   "sections": [
#     {"id": "module-overview", "name": "Module Overview", "anchor": "module-overview"},
#     {"id": "root-help", "name": "Root Help Coverage", "anchor": "test_docparse_root_help_lists_subcommands"},
#     {"id": "doctags-behaviour", "name": "DocTags Command Behaviour", "anchor": "test_docparse_doctags_help_preserves_legacy_flags"},
#     {"id": "global-flags", "name": "Global Flag Behaviour", "anchor": "test_docparse_root_version_flag"},
#     {"id": "plan-forwarding", "name": "Plan Forwarding", "anchor": "test_docparse_plan_receives_none_when_no_args"}
#   ]
# }
# === /NAVMAP ===

"""Regression tests for the Typer-backed DocParsing CLI dispatcher.

Each test asserts that the Typer application surfaces the expected help text,
forwards argv to the legacy stage shims unchanged, and honours the new global
flags introduced during the migration (e.g. `--version`, root-level
`--data-root`). The suite is designed to detect behavioural regressions
without depending on the heavy downstream pipelines.
"""

from __future__ import annotations

from typing import Sequence

from typer.testing import CliRunner

from DocsToKG.DocParsing.core import cli as core_cli

runner = CliRunner()


def test_docparse_root_help_lists_subcommands() -> None:
    """The Typer root help should enumerate all known commands."""

    result = runner.invoke(core_cli.app, ["--help"])
    assert result.exit_code == 0
    stdout = result.stdout
    assert "doctags" in stdout
    assert "chunk" in stdout
    assert "embed" in stdout
    assert "token-profiles" in stdout
    assert "manifest" in stdout
    assert "all" in stdout


def test_docparse_doctags_help_preserves_legacy_flags() -> None:
    """`docparse doctags --help` continues to surface the argparse options."""

    result = runner.invoke(core_cli.app, ["doctags", "--help"])
    assert result.exit_code == 0
    stdout = result.stdout
    # Spot-check a few legacy flags to confirm the passthrough help is rendered.
    assert "--mode" in stdout
    assert "[auto|html|pdf]" in stdout
    assert "--served-model-name" in stdout
    assert "--vllm-wait-timeout" in stdout


def test_docparse_doctags_forwards_arguments(monkeypatch) -> None:
    """Typer should forward argv to the legacy handler unchanged."""

    captured: dict[str, Sequence[str] | None] = {}

    def fake_doctags(argv: Sequence[str] | None = None) -> int:
        captured["argv"] = argv
        return 5

    monkeypatch.setattr(core_cli, "_execute_doctags", fake_doctags)
    result = runner.invoke(core_cli.app, ["doctags", "--mode", "pdf", "--resume"])
    assert result.exit_code == 5
    assert captured["argv"] == ["--mode", "pdf", "--resume"]


def test_docparse_root_version_flag() -> None:
    """The global --version flag should print the package version and exit."""

    result = runner.invoke(core_cli.app, ["--version"])
    assert result.exit_code == 0
    assert "DocsToKG" in result.stdout


def test_docparse_root_data_root_option_applies_to_subcommand(
    tmp_path, monkeypatch
) -> None:
    """Global --data-root should populate subcommands when omitted locally."""

    captured: dict[str, Sequence[str] | None] = {}

    def fake_chunk(argv: Sequence[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(core_cli, "_execute_chunk", fake_chunk)
    result = runner.invoke(core_cli.app, ["--data-root", str(tmp_path), "chunk"])
    assert result.exit_code == 0
    assert captured["argv"] is not None
    assert "--data-root" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--data-root") + 1] == str(tmp_path)


def test_docparse_plan_receives_none_when_no_args(monkeypatch) -> None:
    """The plan command should forward None when no explicit flags are given."""

    captured: dict[str, Sequence[str] | None] = {}

    def fake_plan(argv: Sequence[str] | None = None) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(core_cli, "_execute_run_all", fake_plan)
    result = runner.invoke(core_cli.app, ["plan"])
    assert result.exit_code == 0
    assert captured["argv"] == ["--plan"]
