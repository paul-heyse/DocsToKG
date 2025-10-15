#!/usr/bin/env python3
"""
Unified DocTags Conversion CLI

This command-line interface orchestrates HTML and PDF conversions to DocTags
using Docling backends. It consolidates disparate scripts into a single entry
point that auto-detects the appropriate backend, manages manifests, and shares
DocsToKG-wide defaults.

Key Features:
- Auto-detect the conversion backend from input directory contents
- Forward CLI options to specialised HTML and PDF pipelines
- Integrate with DocsToKG resume/force semantics for idempotent runs

Usage:
    python -m DocsToKG.DocParsing.cli.doctags_convert --mode auto --input Data/HTML
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from DocsToKG.DocParsing.legacy import run_docling_html_to_doctags_parallel as html_backend
from DocsToKG.DocParsing.legacy import run_docling_parallel_with_vllm_debug as pdf_backend
from DocsToKG.DocParsing._common import (
    data_doctags,
    data_html,
    data_pdfs,
    detect_data_root,
    get_logger,
)

_EXAMPLES = """Examples:
  # Auto-detect mode from input directory contents
  python -m DocsToKG.DocParsing.cli.doctags_convert --input Data/HTML

  # Force PDF conversion with explicit workers and output directory
  python -m DocsToKG.DocParsing.cli.doctags_convert --mode pdf \
      --input corpora/pdfs --output Data/DocTagsFiles --workers 4

  # Convert HTML while overwriting existing DocTags files
  python -m DocsToKG.DocParsing.cli.doctags_convert --mode html --overwrite
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the unified DocTags conversion argument parser.

    Args:
        None: Parser creation does not require inputs.

    Returns:
        :class:`argparse.ArgumentParser` populated with DocTags CLI options.

    Raises:
        None
    """

    parser = argparse.ArgumentParser(
        description="Convert HTML or PDF corpora to DocTags using Docling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "pdf"],
        default="auto",
        help="Select conversion backend; auto infers from input directory",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or "
            "$DOCSTOKG_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Directory containing HTML or PDF sources (defaults vary by mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for generated .doctags files (defaults vary by mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes to launch; backend defaults used when omitted",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override vLLM model path or identifier for PDF conversion",
    )
    parser.add_argument(
        "--served-model-name",
        dest="served_model_names",
        action="append",
        nargs="+",
        default=None,
        help="Model alias to expose from vLLM (repeatable)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory allocated to the vLLM server",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip documents whose outputs already exist with matching content hash",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even when resume criteria are satisfied",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DocTags files (HTML mode only)",
    )
    return parser


def detect_mode(input_dir: Path) -> str:
    """Inspect ``input_dir`` and infer conversion mode based on file types.

    Args:
        input_dir: Directory whose contents determine the appropriate backend.

    Returns:
        ``"pdf"`` or ``"html"`` depending on the detected file extensions.

    Raises:
        ValueError: If both PDF and HTML files are present (or neither).

    Examples:
        >>> tmp = Path("/tmp/docstokg-cli-examples")
        >>> _ = tmp.mkdir(exist_ok=True)
        >>> _ = (tmp / "example.html").write_text("<html></html>", encoding="utf-8")
        >>> detect_mode(tmp)
        'html'
    """

    pdf_count = sum(1 for _ in input_dir.rglob("*.pdf"))
    html_count = sum(1 for _ in input_dir.rglob("*.html")) + sum(
        1 for _ in input_dir.rglob("*.htm")
    )
    if pdf_count > 0 and html_count == 0:
        return "pdf"
    if html_count > 0 and pdf_count == 0:
        return "html"
    raise ValueError(
        f"Cannot auto-detect mode in {input_dir}: found {pdf_count} PDFs and {html_count} HTML files"
    )


def _merge_args(parser: argparse.ArgumentParser, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Return a parser namespace seeded with override values.

    Args:
        parser: Parser whose default values should seed the namespace.
        overrides: Mapping of argument names to explicit override values.

    Returns:
        :class:`argparse.Namespace` with defaults populated and overrides applied.
    """

    base = parser.parse_args([])
    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    return base


def main(args: argparse.Namespace | list[str] | None = None) -> int:
    """Dispatch conversion to the HTML or PDF backend based on requested mode.

    Args:
        args: Either an :class:`argparse.Namespace`, a list of CLI arguments, or ``None``.

    Returns:
        Exit code from the selected conversion backend.

    Raises:
        ValueError: If conversion mode cannot be determined automatically.
    """

    parser = build_parser()
    parsed_args = args if isinstance(args, argparse.Namespace) else parser.parse_args(args)
    args = parsed_args
    logger = get_logger(__name__)

    resolved_root = (
        detect_data_root(args.data_root) if args.data_root is not None else detect_data_root()
    )

    html_default_in = data_html(resolved_root)
    pdf_default_in = data_pdfs(resolved_root)
    doctags_default_out = data_doctags(resolved_root)

    mode = args.mode
    input_dir: Path
    if args.input is not None:
        input_dir = args.input.resolve()
        if mode == "auto":
            mode = detect_mode(input_dir)
    else:
        if mode == "auto":
            html_count = sum(1 for _ in html_default_in.rglob("*.html")) + sum(
                1 for _ in html_default_in.rglob("*.htm")
            )
            pdf_count = sum(1 for _ in pdf_default_in.rglob("*.pdf"))
            if html_count > 0 and pdf_count == 0:
                mode = "html"
            elif pdf_count > 0 and html_count == 0:
                mode = "pdf"
            else:
                raise ValueError("Cannot auto-detect mode: specify --mode or --input explicitly")
        input_dir = html_default_in if mode == "html" else pdf_default_in

    output_dir = args.output.resolve() if args.output is not None else doctags_default_out

    logger.info(
        "Unified DocTags conversion",
        extra={
            "extra_fields": {
                "mode": mode,
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "workers": args.workers,
                "resume": args.resume,
                "force": args.force,
                "overwrite": args.overwrite,
                "model": args.model,
                "served_model_names": args.served_model_names,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        },
    )

    if mode == "html":
        overrides = {
            "data_root": args.data_root,
            "input": input_dir,
            "output": output_dir,
            "workers": args.workers,
            "resume": args.resume,
            "force": args.force,
            "overwrite": args.overwrite,
        }
        html_args = _merge_args(html_backend.build_parser(), overrides)
        return html_backend.main(html_args)

    overrides = {
        "data_root": args.data_root,
        "input": input_dir,
        "output": output_dir,
        "workers": args.workers,
        "resume": args.resume,
        "force": args.force,
        "model": args.model,
        "served_model_names": args.served_model_names,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    pdf_args = _merge_args(pdf_backend.build_parser(), overrides)
    return pdf_backend.main(pdf_args)


if __name__ == "__main__":
    raise SystemExit(main())
