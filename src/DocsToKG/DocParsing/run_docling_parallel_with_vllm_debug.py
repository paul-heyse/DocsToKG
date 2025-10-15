#!/usr/bin/env python3
"""Legacy PDF → DocTags Converter with vLLM (DEPRECATED).

⚠️  This script is deprecated. Use the unified CLI instead:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf

This shim forwards invocations to the unified CLI for backward compatibility
and will be removed in a future release.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser (deprecated, forwards to unified CLI)."""
    warnings.warn(
        "run_docling_parallel_with_vllm_debug.py is deprecated. "
        "Use: python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf",
        DeprecationWarning,
        stacklevel=2,
    )
    from DocsToKG.DocParsing.cli.doctags_convert import build_parser as unified_build_parser

    return unified_build_parser()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (deprecated)."""
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Forward to unified CLI with PDF mode forced."""
    warnings.warn(
        "\n"
        + "=" * 70
        + "\nDEPRECATION WARNING: run_docling_parallel_with_vllm_debug.py\n"
        + "=" * 70
        + "\nThis script is deprecated. Please update your code to use:\n"
        + "  python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf\n\n"
        + "This shim will be removed in the next major release.\n"
        + "=" * 70,
        DeprecationWarning,
        stacklevel=2,
    )

    from DocsToKG.DocParsing.cli.doctags_convert import main as unified_main

    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--mode" not in argv_list:
        argv_list = ["--mode", "pdf", *argv_list]
    return unified_main(argv_list)


if __name__ == "__main__":
    raise SystemExit(main())
