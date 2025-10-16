# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.cli",
#   "purpose": "CLI entry points for DocsToKG.DocParsing.cli workflows",
#   "sections": [
#     {
#       "id": "_run_chunk",
#       "name": "_run_chunk",
#       "anchor": "RC",
#       "kind": "function"
#     },
#     {
#       "id": "_run_embed",
#       "name": "_run_embed",
#       "anchor": "RE",
#       "kind": "function"
#     },
#     {
#       "id": "_build_doctags_parser",
#       "name": "_build_doctags_parser",
#       "anchor": "BDP",
#       "kind": "function"
#     },
#     {
#       "id": "_detect_mode",
#       "name": "_detect_mode",
#       "anchor": "DM",
#       "kind": "function"
#     },
#     {
#       "id": "_merge_args",
#       "name": "_merge_args",
#       "anchor": "MA",
#       "kind": "function"
#     },
#     {
#       "id": "_run_doctags",
#       "name": "_run_doctags",
#       "anchor": "RD",
#       "kind": "function"
#     },
#     {
#       "id": "_command",
#       "name": "_Command",
#       "anchor": "COMM",
#       "kind": "class"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "MAIN",
#       "kind": "function"
#     },
#     {
#       "id": "chunk",
#       "name": "chunk",
#       "anchor": "CHUN",
#       "kind": "function"
#     },
#     {
#       "id": "embed",
#       "name": "embed",
#       "anchor": "EMBE",
#       "kind": "function"
#     },
#     {
#       "id": "doctags",
#       "name": "doctags",
#       "anchor": "DOCT",
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

CommandHandler = Callable[[Sequence[str]], int]

CLI_DESCRIPTION = """\
Unified DocParsing CLI

Examples:
  python -m DocsToKG.DocParsing.cli chunk --data-root Data
  python -m DocsToKG.DocParsing.cli embed --resume
  python -m DocsToKG.DocParsing.cli doctags --mode pdf --workers 2
"""


# --------------------------------------------------------------------------- #
# Chunk command
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Embed command
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Doctags command
# --------------------------------------------------------------------------- #


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


def _detect_mode(input_dir: Path) -> str:
    """Infer conversion mode based on the contents of ``input_dir``.

    Args:
        input_dir: Directory searched for PDF and HTML inputs.

    Returns:
        ``"pdf"`` when only PDFs are present, ``"html"`` when only HTML files exist.

    Raises:
        ValueError: If both formats are present or neither can be detected.
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
        f"Cannot auto-detect mode in {input_dir}: "
        f"found {pdf_count} PDFs and {html_count} HTML files"
    )


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
    if args.input is not None:
        input_dir = args.input.resolve()
        if mode == "auto":
            mode = _detect_mode(input_dir)
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
        html_args = _merge_args(pipeline_backend.html_build_parser(), overrides)
        return pipeline_backend.html_main(html_args)

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
    pdf_args = _merge_args(pipeline_backend.pdf_build_parser(), overrides)
    return pipeline_backend.pdf_main(pdf_args)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


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


__all__ = ["main", "chunk", "embed", "doctags"]


if __name__ == "__main__":
    raise SystemExit(main())
