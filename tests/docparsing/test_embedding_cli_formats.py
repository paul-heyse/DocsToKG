"""Ensure the embedding CLI exposes and validates supported vector formats.

These tests interrogate the argparse surface for `docparse embed`, checking that
all advertised `--vector-format` choices parse successfully and that unexpected
values trigger helpful errors. Keeping this coverage prevents regressions when
new vector serialization backends are introduced.
"""

from __future__ import annotations

import argparse

import pytest

from DocsToKG.DocParsing.embedding import cli as embedding_cli


def _vector_format_action(parser: argparse.ArgumentParser) -> argparse.Action:
    """Return the argparse action that controls ``--vector-format``."""

    for action in parser._actions:
        if "--vector-format" in action.option_strings:
            return action
    raise AssertionError("--vector-format option is not registered on the parser")


def test_embedding_cli_accepts_advertised_formats() -> None:
    """Ensure every advertised vector format parses successfully."""

    parser = embedding_cli.build_parser()
    format_action = _vector_format_action(parser)

    for fmt in format_action.choices:
        args = parser.parse_args(["--data-root", "Data", "--vector-format", fmt])
        assert args.vector_format == fmt


@pytest.mark.parametrize("unsupported", ["parquet", "csv", "yaml"])
def test_embedding_cli_rejects_unsupported_formats(unsupported: str) -> None:
    """Unsupported formats should trigger ``SystemExit`` from argparse choices."""

    parser = embedding_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--data-root", "Data", "--vector-format", unsupported])
