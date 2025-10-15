#!/usr/bin/env python3
"""Convenience CLI wrapper for the DocTags chunking pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
    build_parser as chunk_build_parser,
    main as chunk_main,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose the chunker parser with an enhanced description."""

    parser = chunk_build_parser()
    parser.description = "Chunk DocTags corpora with topic-aware coalescence"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and invoke the chunking pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return chunk_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
