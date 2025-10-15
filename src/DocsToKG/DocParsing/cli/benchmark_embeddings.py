#!/usr/bin/env python3
"""Synthetic benchmark harness for the DocParsing embedding pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from DocsToKG.DocParsing.testing import (
    format_benchmark_summary,
    simulate_embedding_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the synthetic benchmark harness."""

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
    """Execute the synthetic benchmark and emit a human-friendly summary."""

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
