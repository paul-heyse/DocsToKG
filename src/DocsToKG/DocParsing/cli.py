# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.cli",
#   "purpose": "CLI entry points for DocParsing workflows",
#   "sections": [
#     {
#       "id": "run-chunk",
#       "name": "_run_chunk",
#       "anchor": "function-run-chunk",
#       "kind": "function"
#     },
#     {
#       "id": "run-embed",
#       "name": "_run_embed",
#       "anchor": "function-run-embed",
#       "kind": "function"
#     },
#     {
#       "id": "build-doctags-parser",
#       "name": "_build_doctags_parser",
#       "anchor": "function-build-doctags-parser",
#       "kind": "function"
#     },
#     {
#       "id": "detect-mode",
#       "name": "_detect_mode",
#       "anchor": "function-detect-mode",
#       "kind": "function"
#     },
#     {
#       "id": "merge-args",
#       "name": "_merge_args",
#       "anchor": "function-merge-args",
#       "kind": "function"
#     },
#     {
#       "id": "run-doctags",
#       "name": "_run_doctags",
#       "anchor": "function-run-doctags",
#       "kind": "function"
#     },
#     {
#       "id": "command",
#       "name": "_Command",
#       "anchor": "class-command",
#       "kind": "class"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     },
#     {
#       "id": "chunk",
#       "name": "chunk",
#       "anchor": "function-chunk",
#       "kind": "function"
#     },
#     {
#       "id": "embed",
#       "name": "embed",
#       "anchor": "function-embed",
#       "kind": "function"
#     },
#     {
#       "id": "doctags",
#       "name": "doctags",
#       "anchor": "function-doctags",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Unified DocParsing command-line interface.

This module consolidates the individual DocParsing CLIs into a single entry
point with subcommands. Invoke it with:

    python -m DocsToKG.DocParsing.cli <command> [options...]

Available commands:
    - chunk:     Run the Docling hybrid chunker.
    - embed:     Generate BM25, SPLADE, and dense vectors for chunks.
    - doctags:   Convert HTML/PDF corpora into DocTags.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from DocsToKG.DocParsing import pipelines as pipeline_backend
from DocsToKG.DocParsing._common import (
    data_doctags,
    data_html,
    data_pdfs,
    detect_data_root,
    get_logger,
)
from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
    build_parser as chunk_build_parser,
)
from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
    main as chunk_pipeline_main,
)
from DocsToKG.DocParsing.EmbeddingV2 import build_parser as embed_build_parser
from DocsToKG.DocParsing.EmbeddingV2 import main as embed_pipeline_main


# --- Globals ---

CommandHandler = Callable[[Sequence[str]], int]

CLI_DESCRIPTION = """\
Unified DocParsing CLI

Examples:
  python -m DocsToKG.DocParsing.cli chunk --data-root Data
  python -m DocsToKG.DocParsing.cli embed --resume
  python -m DocsToKG.DocParsing.cli doctags --mode pdf --workers 2
"""

__all__ = ["main", "chunk", "embed", "doctags"]


# --- Chunk Command ---

def _run_chunk(argv: Sequence[str]) -> int:
    """Execute the Docling chunker subcommand.

    Args:
        argv: Argument vector forwarded to the chunker parser.

    Returns:
        Process exit code produced by the Docling chunker pipeline.
    """
    parser = chunk_build_parser()
    parser.prog = "docparse chunk"
    args = parser.parse_args(argv)
    return chunk_pipeline_main(args)


# --- Embed Command ---

def _run_embed(argv: Sequence[str]) -> int:
    """Execute the embedding pipeline subcommand.

    Args:
        argv: Argument vector forwarded to the embedding parser.

    Returns:
        Process exit code produced by the embedding pipeline.
    """
    parser = embed_build_parser()
    parser.prog = "docparse embed"
    args = parser.parse_args(argv)
    return embed_pipeline_main(args)


# --- Doctags Command ---

def _build_doctags_parser(prog: str = "docparse doctags") -> argparse.ArgumentParser:
    """Create an :mod:`argparse` parser configured for DocTags conversion.

    Args:
        prog: Program name displayed in help output.

    Returns:
        Argument parser instance for the ``doctags`` subcommand.
    """
    examples = """Examples:
  docparse doctags --input Data/HTML
  docparse doctags --mode pdf --workers 4
  docparse doctags --mode html --overwrite
"""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert HTML or PDF corpora to DocTags using Docling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "pdf"],
        default="auto",
        help="Select conversion backend; auto infers from input directory",
    )
    pipeline_backend.add_data_root_option(parser)
    parser.add_argument(
        "--in-dir",
        "--input",
        dest="in_dir",
        type=Path,
        default=None,
        help="Directory containing HTML or PDF sources (defaults vary by mode)",
    )
    parser.add_argument(
        "--out-dir",
        "--output",
        dest="out_dir",
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
    pipeline_backend.add_resume_force_options(
        parser,
        resume_help="Skip documents whose outputs already exist with matching content hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DocTags files (HTML mode only)",
    )
    return parser


_PDF_SUFFIXES: tuple[str, ...] = (".pdf",)
_HTML_SUFFIXES: tuple[str, ...] = (".html", ".htm")


def _scan_pdf_html(input_dir: Path) -> tuple[bool, bool]:
    """Return booleans indicating whether PDFs or HTML files exist beneath ``input_dir``."""

    has_pdf = False
    has_html = False

    if not input_dir.exists():
        return has_pdf, has_html

    for root, _dirs, files in os.walk(input_dir):
        if not files:
            continue
        for name in files:
            lower = name.lower()
            if not has_pdf and lower.endswith(_PDF_SUFFIXES):
                has_pdf = True
            elif not has_html and lower.endswith(_HTML_SUFFIXES):
                has_html = True
            if has_pdf and has_html:
                return has_pdf, has_html
    return has_pdf, has_html


def _directory_contains_suffixes(directory: Path, suffixes: tuple[str, ...]) -> bool:
    """Return True when ``directory`` contains at least one file ending with ``suffixes``."""

    if not directory.exists():
        return False
    suffixes_lower = tuple(s.lower() for s in suffixes)
    for root, _dirs, files in os.walk(directory):
        if not files:
            continue
        for name in files:
            if name.lower().endswith(suffixes_lower):
                return True
    return False


def _detect_mode(input_dir: Path) -> str:
    """Infer conversion mode based on the contents of ``input_dir``.

    Args:
        input_dir: Directory searched for PDF and HTML inputs.

    Returns:
        ``"pdf"`` when only PDFs are present, ``"html"`` when only HTML files exist.

    Raises:
        ValueError: If both formats are present, neither type can be detected, or the directory is missing.
    """
    if not input_dir.exists():
        raise ValueError(f"Cannot auto-detect mode in {input_dir}: directory not found")

    has_pdf, has_html = _scan_pdf_html(input_dir)
    if has_pdf and not has_html:
        return "pdf"
    if has_html and not has_pdf:
        return "html"
    if has_pdf and has_html:
        raise ValueError(
            f"Cannot auto-detect mode in {input_dir}: found both PDF and HTML files"
        )
    raise ValueError(f"Cannot auto-detect mode in {input_dir}: no PDF or HTML files found")


def _merge_args(parser: argparse.ArgumentParser, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Merge override values into the default parser namespace.

    Args:
        parser: Parser providing default argument values.
        overrides: Mapping of argument names to replacement values.

    Returns:
        Namespace populated with defaults and supplied overrides.
    """
    base = parser.parse_args([])
    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    return base


def _run_doctags(argv: Sequence[str]) -> int:
    """Execute the DocTags conversion subcommand.

    Args:
        argv: Argument vector provided by the CLI dispatcher.

    Returns:
        Process exit code from the selected DocTags backend.
    """
    parser = _build_doctags_parser()
    args = parser.parse_args(argv)
    logger = get_logger(__name__)

    resolved_root = (
        detect_data_root(args.data_root) if args.data_root is not None else detect_data_root()
    )

    html_default_in = data_html(resolved_root)
    pdf_default_in = data_pdfs(resolved_root)
    doctags_default_out = data_doctags(resolved_root)

    mode = args.mode
    if args.in_dir is not None:
        input_dir = args.in_dir.resolve()
        if mode == "auto":
            mode = _detect_mode(input_dir)
    else:
        if mode == "auto":
            html_present = _directory_contains_suffixes(html_default_in, _HTML_SUFFIXES)
            pdf_present = _directory_contains_suffixes(pdf_default_in, _PDF_SUFFIXES)
            if html_present and not pdf_present:
                mode = "html"
            elif pdf_present and not html_present:
                mode = "pdf"
            else:
                raise ValueError("Cannot auto-detect mode: specify --mode or --input explicitly")
        input_dir = html_default_in if mode == "html" else pdf_default_in

    output_dir = args.out_dir.resolve() if args.out_dir is not None else doctags_default_out

    args.in_dir = input_dir
    args.out_dir = output_dir

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

    base_overrides = {
        "data_root": args.data_root,
        "input": input_dir,
        "output": output_dir,
        "workers": args.workers,
        "resume": args.resume,
        "force": args.force,
    }

    if mode == "html":
        html_overrides = {
            **base_overrides,
            "overwrite": args.overwrite,
        }
        html_args = _merge_args(pipeline_backend.html_build_parser(), html_overrides)
        return pipeline_backend.html_main(html_args)

    overrides = {
        **base_overrides,
        "model": args.model,
        "served_model_names": args.served_model_names,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    pdf_args = _merge_args(pipeline_backend.pdf_build_parser(), overrides)
    return pipeline_backend.pdf_main(pdf_args)


# --- Dispatcher ---

class _Command:
    """Callable wrapper storing handler metadata for subcommands.

    Attributes:
        handler: Callable invoked with the subcommand argument vector.
        help: Short help text displayed in CLI usage.

    Examples:
        >>> cmd = _Command(_run_chunk, \"Run the chunker\")
        >>> cmd.handler([])  # doctest: +SKIP
        0
    """

    __slots__ = ("handler", "help")

    def __init__(self, handler: CommandHandler, help: str) -> None:
        """Initialize a command wrapper.

        Args:
            handler: Callable that executes the subcommand.
            help: Short help text displayed in CLI usage.

        Returns:
            None
        """
        self.handler = handler
        self.help = help


COMMANDS: Dict[str, _Command] = {
    "chunk": _Command(_run_chunk, "Run the Docling hybrid chunker"),
    "embed": _Command(_run_embed, "Generate BM25/SPLADE/dense vectors"),
    "doctags": _Command(_run_doctags, "Convert HTML/PDF corpora into DocTags"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to one of the DocParsing subcommands.

    Args:
        argv: Optional argument vector supplied programmatically.

    Returns:
        Process exit code returned by the selected subcommand.
    """

    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS.keys(), help="CLI to execute")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to the selected command",
    )

    parsed = parser.parse_args(argv)
    command_args: List[str] = list(parsed.args)
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]

    command = COMMANDS[parsed.command]
    return command.handler(command_args)


def chunk(argv: Sequence[str] | None = None) -> int:
    """Programmatic helper mirroring ``docparse chunk``.

    Args:
        argv: Optional argument vector supplied for testing.

    Returns:
        Process exit code returned by the chunker pipeline.
    """

    return _run_chunk([] if argv is None else list(argv))


def embed(argv: Sequence[str] | None = None) -> int:
    """Programmatic helper mirroring ``docparse embed``.

    Args:
        argv: Optional argument vector supplied for testing.

    Returns:
        Process exit code returned by the embedding pipeline.
    """

    return _run_embed([] if argv is None else list(argv))


def doctags(argv: Sequence[str] | None = None) -> int:
    """Programmatic helper mirroring ``docparse doctags``.

    Args:
        argv: Optional argument vector supplied for testing.

    Returns:
        Process exit code returned by the DocTags conversion pipeline.
    """

    return _run_doctags([] if argv is None else list(argv))


if __name__ == "__main__":
    raise SystemExit(main())
