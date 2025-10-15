#!/usr/bin/env python3
"""
Embedding CLI Wrapper

This CLI exposes the hybrid embedding pipeline that generates BM25, SPLADE, and
Qwen vectors for DocsToKG chunk files. It reuses the core parser and runtime
logic from ``DocsToKG.DocParsing.EmbeddingV2`` while enhancing descriptions for
operator-facing documentation.

Key Features:
- Share argument definitions with the primary embedding module
- Provide concise messaging suited for orchestration scripts
- Support ``python -m`` invocation without altering defaults

Usage:
    python -m DocsToKG.DocParsing.cli.embed_vectors --resume
"""

from __future__ import annotations

import argparse
from typing import Sequence

from DocsToKG.DocParsing.EmbeddingV2 import build_parser as embed_build_parser
from DocsToKG.DocParsing.EmbeddingV2 import main as embed_main


def build_parser() -> argparse.ArgumentParser:
    """Return the embedding parser with a concise description.

    Args:
        None

    Returns:
        :class:`argparse.ArgumentParser` configured for embedding CLI usage.

    Raises:
        None
    """

    parser = embed_build_parser()
    parser.description = "Generate BM25, SPLADE, and Qwen vectors for chunk files"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and invoke the embedding pipeline.

    Args:
        argv: Optional sequence of command-line arguments.

    Returns:
        Exit code from the embedding pipeline.

    Raises:
        SystemExit: Propagated when argument parsing fails.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return embed_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
