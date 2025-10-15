#!/usr/bin/env python3
"""Convenience CLI wrapper for the hybrid embedding pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from DocsToKG.DocParsing.EmbeddingV2 import build_parser as embed_build_parser
from DocsToKG.DocParsing.EmbeddingV2 import main as embed_main


def build_parser() -> argparse.ArgumentParser:
    """Return the embedding parser with a concise description."""

    parser = embed_build_parser()
    parser.description = "Generate BM25, SPLADE, and Qwen vectors for chunk files"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and invoke the embedding pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return embed_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
