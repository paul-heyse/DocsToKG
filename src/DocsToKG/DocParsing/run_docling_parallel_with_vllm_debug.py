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
from pathlib import Path
from typing import Optional

legacy_main = None

try:  # pragma: no cover - legacy compatibility path
    from DocsToKG.DocParsing.legacy.run_docling_parallel_with_vllm_debug import (
        PREFERRED_PORT,
        convert_one,
        ensure_vllm,
        list_pdfs,
        stop_vllm,
        ProcessPoolExecutor,
        as_completed,
        tqdm,
        main as legacy_main,
    )
except ModuleNotFoundError:  # pragma: no cover - test stubs
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - fallback when tqdm missing
        def tqdm(iterable=None, *args, **kwargs):
            """Fallback progress wrapper when :mod:`tqdm` is unavailable.

            Args:
                iterable: Iterable to wrap for progress reporting.
                *args: Positional arguments accepted for API compatibility.
                **kwargs: Keyword arguments accepted for API compatibility.

            Returns:
                The original iterable when provided, otherwise an empty list.
            """

            return iterable if iterable is not None else []

    PREFERRED_PORT = 8000

    def ensure_vllm(*args, **kwargs):  # pragma: no cover - shim should be patched in tests
        """Fallback ensure_vllm stub raising to encourage CLI migration.

        Args:
            *args: Positional arguments captured for compatibility.
            **kwargs: Keyword arguments captured for compatibility.

        Returns:
            None. This function always raises.

        Raises:
            RuntimeError: Always raised to direct callers to the unified CLI.
        """

        raise RuntimeError("Legacy ensure_vllm shim invoked; migrate to unified CLI")

    def stop_vllm(*_args, **_kwargs):  # pragma: no cover - shim noop
        """Fallback stop_vllm stub returning ``None``.

        Args:
            *_args: Positional arguments ignored.
            **_kwargs: Keyword arguments ignored.

        Returns:
            None. The stub performs no action.
        """

        return None

    def list_pdfs(directory):  # pragma: no cover - basic globbing fallback
        """Return sorted list of ``.pdf`` files when legacy helpers are missing.

        Args:
            directory: Directory path containing candidate PDF files.

        Returns:
            Sorted list of ``Path`` objects for ``.pdf`` files in ``directory``.
        """

        path = Path(directory)
        return sorted(path.glob("*.pdf")) if path.exists() else []

    def convert_one(task):  # pragma: no cover - shim placeholder
        """Fallback convert stub that forces migration to unified CLI.

        Args:
            task: Conversion task payload expected by legacy pipeline.

        Returns:
            None. The stub always raises.

        Raises:
            RuntimeError: Always raised to direct callers to the unified CLI.
        """

        raise RuntimeError("Legacy convert_one shim invoked; migrate to unified CLI")


def build_parser() -> argparse.ArgumentParser:
    """Build the deprecated argument parser and forward to the unified CLI.

    Args:
        None

    Returns:
        Parser instance sourced from the unified CLI implementation.

    Raises:
        ImportError: If the unified CLI module cannot be imported.
    """
    warnings.warn(
        "run_docling_parallel_with_vllm_debug.py is deprecated. "
        "Use: python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf",
        DeprecationWarning,
        stacklevel=2,
    )
    from DocsToKG.DocParsing.cli.doctags_convert import build_parser as unified_build_parser

    return unified_build_parser()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments using the forwarded parser.

    Args:
        argv: Optional list of CLI arguments to parse instead of :data:`sys.argv`.

    Returns:
        Namespace containing CLI arguments supported by the unified command.

    Raises:
        SystemExit: Propagated when parsing fails.
    """
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Forward to the unified CLI with PDF mode forced.

    Args:
        argv: Optional CLI argument list excluding the executable.

    Returns:
        Exit status returned by the unified CLI entry point.

    Raises:
        SystemExit: Propagated if the invoked CLI terminates.
    """
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

    # Allow legacy tests to intercept argument parsing via monkeypatching.
    try:
        namespace = parse_args(argv_list)
    except TypeError:
        namespace = parse_args()
    if not isinstance(namespace, argparse.Namespace):
        namespace = argparse.Namespace(**vars(namespace))

    parser = build_parser()
    merged = parser.parse_args([])
    for key, value in vars(namespace).items():
        setattr(merged, key, value)
    if not hasattr(merged, "mode"):
        setattr(merged, "mode", "pdf")
    if not hasattr(merged, "data_root"):
        setattr(merged, "data_root", None)
    if getattr(merged, "gpu_memory_utilization", None) is None:
        setattr(merged, "gpu_memory_utilization", 0.9)
    if getattr(merged, "served_model_names", None) is None:
        setattr(merged, "served_model_names", ["vllm-default"])
    if getattr(merged, "workers", None) is None:
        setattr(merged, "workers", 1)
    if getattr(merged, "resume", None) is None:
        setattr(merged, "resume", False)
    if getattr(merged, "force", None) is None:
        setattr(merged, "force", False)
    if getattr(merged, "overwrite", None) is None:
        setattr(merged, "overwrite", False)
    if getattr(merged, "model", None) is None:
        setattr(merged, "model", None)
    if legacy_main is not None:
        return legacy_main(merged)
    return unified_main(merged)


if __name__ == "__main__":
    raise SystemExit(main())
