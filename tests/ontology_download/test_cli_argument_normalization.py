# === NAVMAP v1 ===
# {
#   "module": "tests.ontology_download.test_cli_argument_normalization",
#   "purpose": "Regression tests for CLI argument normalization behaviour.",
#   "sections": [
#     {"id": "tests", "name": "Test Cases", "anchor": "TST", "kind": "tests"}
#   ]
# }
# === /NAVMAP ===

"""Regression tests for CLI argument normalization behaviour."""

from __future__ import annotations

import pytest

from DocsToKG.OntologyDownload.cli import _normalize_plan_args


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["hp"], ["pull", "hp"]),
        (["--log-level", "DEBUG", "hp"], ["--log-level", "DEBUG", "pull", "hp"]),
        (["--log-level=DEBUG", "hp"], ["--log-level=DEBUG", "pull", "hp"]),
    ],
)
def test_normalize_plan_args_inserts_default_subcommand(argv: list[str], expected: list[str]) -> None:
    """Positional ontology identifiers should trigger insertion of the default command."""

    assert _normalize_plan_args(argv) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["pull", "hp"],
        ["plan", "hp"],
        ["config", "validate"],
        ["--log-level", "DEBUG", "pull", "hp"],
    ],
)
def test_normalize_plan_args_preserves_explicit_commands(argv: list[str]) -> None:
    """Argument vectors with explicit subcommands should pass through untouched."""

    assert _normalize_plan_args(argv) == argv
