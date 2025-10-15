#!/usr/bin/env python3
"""
DocParsing Embedding Benchmark CLI

This command-line interface estimates the performance benefits of the streaming
embedding pipeline relative to the legacy whole-corpus workflow. It relies on
synthetic, deterministic inputs to provide fast feedback without requiring the
full DocsToKG environment.

Key Features:
- Parameterise chunk counts, token lengths, and dense vector dimensionality
- Leverage testing utilities to model runtime and memory characteristics
- Output a ready-to-share textual summary for performance reports

Usage:
    python -m DocsToKG.DocParsing.cli.benchmark_embeddings --chunks 1024 --tokens 512

Dependencies:
- argparse: Parse command-line options exposed by the CLI.
- DocsToKG.DocParsing.testing: Provides simulation primitives used under the hood.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from DocsToKG.DocParsing.testing import (
    format_benchmark_summary,
    simulate_embedding_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the synthetic benchmark harness.

    Args:
        None: Parser creation does not require inputs.

    Returns:
        :class:`argparse.ArgumentParser` configured with benchmark options.

    Raises:
        None
    """

    parser = argparse.ArgumentParser(
        description=(
            "Estimate streaming embedding performance using deterministic "
            "synthetic inputs. The benchmark models the relative speedups and "
            "memory savings achieved by the new streaming pipeline."
        )
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=512,
        help="Number of synthetic chunks to model (default: 512)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=384,
        help="Average tokens per chunk in the synthetic corpus (default: 384)",
    )
    parser.add_argument(
        "--dense-dim",
        type=int,
        default=2560,
        help="Dense embedding dimension to model (default: 2560 for Qwen3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the synthetic benchmark and emit a human-friendly summary.

    Args:
        argv: Optional sequence of command-line arguments. When ``None`` the
            values from :data:`sys.argv` are used.

    Returns:
        Exit code where ``0`` indicates the benchmark completed successfully.

    Raises:
        SystemExit: Propagated if argument parsing fails.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    result = simulate_embedding_benchmark(
        num_chunks=args.chunks,
        chunk_tokens=args.tokens,
        dense_dim=args.dense_dim,
    )
    print(format_benchmark_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
