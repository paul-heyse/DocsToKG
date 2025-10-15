#!/usr/bin/env python3
"""
Chunking CLI Wrapper

This lightweight CLI exposes the Docling hybrid chunker with DocsToKG-specific
defaults. It delegates substantive work to
``DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin`` while presenting a
user-friendly interface aligned with the wider DocParsing toolchain.

Key Features:
- Share the same argument surface as the standalone chunking script
- Provide descriptive help text for DocsToKG operators
- Enable orchestration scripts to call the chunker via ``python -m``

Usage:
    python -m DocsToKG.DocParsing.cli.chunk_and_coalesce --in-dir Data/DocTagsFiles
"""

from __future__ import annotations

import argparse
from typing import Sequence

from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
    build_parser as chunk_build_parser,
)
from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
    main as chunk_main,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose the chunker parser with an enhanced description.

    Args:
        None: Parser creation does not require inputs.

    Returns:
        :class:`argparse.ArgumentParser` configured for chunking CLI usage.

    Raises:
        None
    """

    parser = chunk_build_parser()
    parser.description = "Chunk DocTags corpora with topic-aware coalescence"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and invoke the chunking pipeline.

    Args:
        argv: Optional sequence of command-line arguments. When ``None`` the
            values from :data:`sys.argv` are used.

    Returns:
        Exit code returned by the underlying chunking pipeline.

    Raises:
        SystemExit: Propagated when argument parsing fails.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    return chunk_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
