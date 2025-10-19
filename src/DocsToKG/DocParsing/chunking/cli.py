"""CLI builder utilities dedicated to the DocParsing chunking stage.

This module combines shared CLI option metadata, argument parsers, and
subcommand wiring so that both the standalone ``docparse`` CLI and automation
agents can consistently configure chunking runs. It plugs the chunker-specific
flags into the core CLI scaffolding, supports preset profiles for common worker
and tokenizer combinations, and delegates config hydration to the shared
``parse_args_with_overrides`` helper for parity with other DocParsing stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from DocsToKG.DocParsing.config import parse_args_with_overrides
from DocsToKG.DocParsing.core import CLIOption, build_subcommand
from DocsToKG.DocParsing.doctags import add_data_root_option, add_resume_force_options

from .config import CHUNK_PROFILE_PRESETS, SOFT_BARRIER_MARGIN

CHUNK_CLI_OPTIONS: Tuple[CLIOption, ...] = (
    CLIOption(
        ("--config",),
        {"type": Path, "default": None, "help": "Path to stage config file (JSON/YAML/TOML)."},
    ),
    CLIOption(
        ("--profile",),
        {
            "type": str,
            "default": None,
            "choices": sorted(CHUNK_PROFILE_PRESETS),
            "help": "Preset for workers/token windows/tokenizer (cpu-small, gpu-default, gpu-max, bert-compat).",
        },
    ),
    CLIOption(
        ("--in-dir",),
        {
            "type": Path,
            "default": None,
            "help": "DocTags input directory (defaults to data_root/DocTagsFiles).",
        },
    ),
    CLIOption(
        ("--out-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Chunk output directory (defaults to data_root/ChunkedDocTagFiles).",
        },
    ),
    CLIOption(("--min-tokens",), {"type": int, "default": 256}),
    CLIOption(("--max-tokens",), {"type": int, "default": 512}),
    CLIOption(
        ("--log-level",),
        {
            "type": lambda value: str(value).upper(),
            "default": "INFO",
            "choices": ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
            "help": "Logging verbosity for console output (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--tokenizer-model",),
        {
            "type": str,
            "default": None,
            "help": "Tokenizer identifier used to compute token windows (defaults to profile/default).",
        },
    ),
    CLIOption(
        ("--soft-barrier-margin",),
        {
            "type": int,
            "default": SOFT_BARRIER_MARGIN,
            "help": "Token margin to retain around soft barriers (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--structural-markers",),
        {
            "type": Path,
            "default": None,
            "help": "Optional path to structural marker overrides JSON.",
        },
    ),
    CLIOption(
        ("--heading-markers",),
        {
            "type": Path,
            "default": None,
            "dest": "structural_markers",
            "help": "Deprecated alias for --structural-markers maintained for compatibility.",
        },
    ),
    CLIOption(
        ("--serializer-provider",),
        {
            "type": str,
            "default": None,
            "help": "Serializer implementation to persist chunk outputs (default provider).",
        },
    ),
    CLIOption(("--workers",), {"type": int, "default": 1}),
    CLIOption(("--shard-count",), {"type": int, "default": 1}),
    CLIOption(("--shard-index",), {"type": int, "default": 0}),
    CLIOption(("--validate-only",), {"action": "store_true"}),
    CLIOption(("--inject-anchors",), {"action": "store_true"}),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the chunking pipeline."""

    parser = argparse.ArgumentParser()
    add_data_root_option(parser)
    build_subcommand(parser, CHUNK_CLI_OPTIONS)
    add_resume_force_options(
        parser,
        resume_help="Skip DocTags whose chunk outputs already exist with matching hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the chunking stage."""

    parser = build_parser()
    return parse_args_with_overrides(parser, argv)


__all__ = ["CHUNK_CLI_OPTIONS", "build_parser", "parse_args"]
