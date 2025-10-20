"""CLI parser utilities for configuring the DocParsing embedding stage.

The embedding CLI mirrors the chunking interface but introduces additional
flags for dense/sparse backends, shard coordination, and cache management. This
module defines the reusable option tuples and parser constructors consumed by
``docparse embed`` and orchestration scripts so vector generation runs are
configured consistently across environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from DocsToKG.DocParsing.config import parse_args_with_overrides
from DocsToKG.DocParsing.core import CLIOption, build_subcommand
from DocsToKG.DocParsing.doctags import add_data_root_option, add_resume_force_options

from .config import EMBED_PROFILE_PRESETS, SPLADE_SPARSITY_WARN_THRESHOLD_PCT

EMBED_CLI_OPTIONS: Tuple[CLIOption, ...] = (
    CLIOption(
        ("--config",),
        {"type": Path, "default": None, "help": "Path to stage config file (JSON/YAML/TOML)."},
    ),
    CLIOption(
        ("--profile",),
        {
            "type": str,
            "default": None,
            "choices": sorted(EMBED_PROFILE_PRESETS),
            "help": "Preset controlling batch sizes, SPLADE backend, and Qwen settings.",
        },
    ),
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
        ("--no-cache",),
        {"action": "store_true", "help": "Disable Qwen LLM caching between batches (debug)."},
    ),
    CLIOption(
        ("--shard-count",),
        {
            "type": int,
            "default": 1,
            "help": "Total number of shards for distributed runs (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--shard-index",),
        {
            "type": int,
            "default": 0,
            "help": "Zero-based shard index to process (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--chunks-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Override path to chunk files (auto-detected relative to data root).",
        },
    ),
    CLIOption(
        ("--out-dir", "--vectors-dir"),
        {
            "type": Path,
            "default": None,
            "help": "Directory where vector outputs will be written (auto-detected).",
        },
    ),
    CLIOption(
        ("--vector-format", "--format"),
        {
            "type": str,
            "default": "jsonl",
            "choices": ["jsonl", "parquet"],
            "help": (
                "Vector output format (default: %(default)s). "
                "Parquet requires the pyarrow dependency."
            ),
        },
    ),
    CLIOption(("--bm25-k1",), {"type": float, "default": 1.5}),
    CLIOption(("--bm25-b",), {"type": float, "default": 0.75}),
    CLIOption(("--batch-size-splade",), {"type": int, "default": 32}),
    CLIOption(("--batch-size-qwen",), {"type": int, "default": 64}),
    CLIOption(("--splade-max-active-dims",), {"type": int, "default": None}),
    CLIOption(
        ("--splade-model-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Explicit path to the SPLADE model directory (defaults to DocsToKG cache).",
        },
    ),
    CLIOption(
        ("--splade-attn",),
        {
            "type": str,
            "default": "auto",
            "choices": ["auto", "flash", "sdpa", "eager"],
            "help": (
                "SPLADE attention backend preference order (default: %(default)s). "
                "Auto first attempts FlashAttention 2, then scaled dot-product attention (SDPA) "
                "before falling back to eager/standard attention."
            ),
        },
    ),
    CLIOption(
        ("--qwen-dtype",),
        {
            "type": str,
            "default": "bfloat16",
            "choices": ["float32", "bfloat16", "float16", "int8"],
            "help": "Qwen inference dtype (default: %(default)s).",
        },
    ),
    CLIOption(
        ("--qwen-quant",),
        {
            "type": str,
            "default": None,
            "help": "Optional Qwen quantisation preset (nf4, awq, ...).",
        },
    ),
    CLIOption(
        ("--qwen-model-dir",),
        {
            "type": Path,
            "default": None,
            "help": "Explicit path to the Qwen model directory (defaults to DocsToKG cache).",
        },
    ),
    CLIOption(("--qwen-dim",), {"type": int, "default": 2560}),
    CLIOption(("--tp", "--tensor-parallel"), {"type": int, "default": 1}),
    CLIOption(
        ("--sparsity-warn-threshold-pct",),
        {
            "type": float,
            "default": SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
            "help": (
                "Warn when SPLADE sparsity exceeds this percent of active dims "
                f"(default: {SPLADE_SPARSITY_WARN_THRESHOLD_PCT})."
            ),
        },
    ),
    CLIOption(("--sparsity-report-top-n",), {"type": int, "default": 10}),
    CLIOption(
        ("--files-parallel",),
        {
            "type": int,
            "default": 1,
            "help": "Process up to N chunk files concurrently during embedding (default: 1 for serial runs).",
        },
    ),
    CLIOption(
        ("--validate-only",),
        {"action": "store_true", "help": "Validate existing vectors in --out-dir and exit."},
    ),
    CLIOption(
        ("--plan-only",),
        {
            "action": "store_true",
            "help": "Show resume/skip plan and exit without generating embeddings.",
        },
    ),
    CLIOption(
        ("--offline",),
        {
            "action": "store_true",
            "help": (
                "Disable network access by setting TRANSFORMERS_OFFLINE=1. "
                "All models must already exist in local caches."
            ),
        },
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the embedding pipeline."""

    parser = argparse.ArgumentParser()
    add_data_root_option(parser)
    build_subcommand(parser, EMBED_CLI_OPTIONS)
    add_resume_force_options(
        parser,
        resume_help="Skip chunk files whose vector outputs already exist with matching hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone embedding execution."""

    parser = build_parser()
    return parse_args_with_overrides(parser, argv)


__all__ = ["EMBED_CLI_OPTIONS", "build_parser", "parse_args"]
