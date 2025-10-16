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
#       "id": "scan-pdf-html",
#       "name": "_scan_pdf_html",
#       "anchor": "function-scan-pdf-html",
#       "kind": "function"
#     },
#     {
#       "id": "directory-contains-suffixes",
#       "name": "_directory_contains_suffixes",
#       "anchor": "function-directory-contains-suffixes",
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
#       "id": "run-all",
#       "name": "_run_all",
#       "anchor": "function-run-all",
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
#       "id": "run-all",
#       "name": "run_all",
#       "anchor": "function-run-all",
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
    - all:       Convert HTML/PDF, chunk, and embed sequentially.
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

# --- Globals ---

CommandHandler = Callable[[Sequence[str]], int]

CLI_DESCRIPTION = """\
Unified DocParsing CLI

Examples:
  python -m DocsToKG.DocParsing.cli all --resume
  python -m DocsToKG.DocParsing.cli chunk --data-root Data
  python -m DocsToKG.DocParsing.cli embed --resume
  python -m DocsToKG.DocParsing.cli doctags --mode pdf --workers 2
"""

__all__ = ["main", "run_all", "chunk", "embed", "doctags"]


# --- Chunk Command ---


def _run_chunk(argv: Sequence[str]) -> int:
    """Execute the Docling chunker subcommand.

    Args:
        argv: Argument vector forwarded to the chunker parser.

    Returns:
        Process exit code produced by the Docling chunker pipeline.
    """
    from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
        build_parser as chunk_build_parser,
    )
    from DocsToKG.DocParsing.DoclingHybridChunkerPipelineWithMin import (
        main as chunk_pipeline_main,
    )

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
    from DocsToKG.DocParsing.EmbeddingV2 import build_parser as embed_build_parser
    from DocsToKG.DocParsing.EmbeddingV2 import main as embed_pipeline_main

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
        raise ValueError(f"Cannot auto-detect mode in {input_dir}: found both PDF and HTML files")
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


# --- All-In-One Command ---


def _run_all(argv: Sequence[str]) -> int:
    """Execute DocTags conversion, chunking, and embedding sequentially.

    Args:
        argv: Argument vector supplied by the CLI dispatcher.

    Returns:
        Exit code from the final stage executed. Non-zero codes surface immediately.
    """

    parser = argparse.ArgumentParser(
        prog="docparse all",
        description="Run doctags → chunk → embed sequentially while preserving manifests.",
    )
    pipeline_backend.add_data_root_option(parser)
    pipeline_backend.add_resume_force_options(
        parser,
        resume_help="Apply resume semantics to every stage",
        force_help="Force reprocessing at every stage even if outputs exist",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "html", "pdf"],
        default="auto",
        help="DocTags conversion mode override passed to the doctags stage",
    )
    parser.add_argument(
        "--doctags-in-dir",
        type=Path,
        default=None,
        help="Input directory override for the doctags stage",
    )
    parser.add_argument(
        "--doctags-out-dir",
        type=Path,
        default=None,
        help="Output directory override for generated DocTags files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit overwriting existing DocTags files when running in HTML mode",
    )
    parser.add_argument(
        "--chunk-out-dir",
        type=Path,
        default=None,
        help="Output directory override for chunk JSONL files",
    )
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=None,
        help="Worker processes for the chunk stage",
    )
    parser.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=None,
        help="Minimum tokens per chunk passed to the chunk stage",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per chunk passed to the chunk stage",
    )
    parser.add_argument(
        "--structural-markers",
        type=Path,
        default=None,
        help="Structural marker configuration forwarded to the chunk stage",
    )
    parser.add_argument(
        "--embed-out-dir",
        type=Path,
        default=None,
        help="Output directory override for embedding JSONL files",
    )
    parser.add_argument(
        "--embed-offline",
        action="store_true",
        help="Run the embedding stage with TRANSFORMERS_OFFLINE=1",
    )
    parser.add_argument(
        "--embed-validate-only",
        action="store_true",
        help="Skip embedding generation and only validate existing vectors",
    )
    parser.add_argument(
        "--splade-zero-pct-warn-threshold",
        type=float,
        default=None,
        help="Override SPLADE sparsity warning threshold for the embed stage",
    )

    args = parser.parse_args(argv)
    logger = get_logger(__name__)

    extra = {
        "resume": bool(args.resume),
        "force": bool(args.force),
        "mode": args.mode,
    }
    if args.data_root:
        extra["data_root"] = str(args.data_root)
    logger.info("docparse all starting", extra={"extra_fields": extra})

    doctags_args: List[str] = []
    if args.data_root:
        doctags_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        doctags_args.append("--resume")
    if args.force:
        doctags_args.append("--force")
    if args.mode != "auto":
        doctags_args.extend(["--mode", args.mode])
    if args.doctags_in_dir:
        doctags_args.extend(["--in-dir", str(args.doctags_in_dir)])
    if args.doctags_out_dir:
        doctags_args.extend(["--out-dir", str(args.doctags_out_dir)])
    if args.overwrite:
        doctags_args.append("--overwrite")

    exit_code = _run_doctags(doctags_args)
    if exit_code != 0:
        logger.error(
            "DocTags stage failed",
            extra={"extra_fields": {"exit_code": exit_code}},
        )
        return exit_code

    chunk_args: List[str] = []
    if args.data_root:
        chunk_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        chunk_args.append("--resume")
    if args.force:
        chunk_args.append("--force")
    if args.doctags_out_dir:
        chunk_args.extend(["--in-dir", str(args.doctags_out_dir)])
    if args.chunk_out_dir:
        chunk_args.extend(["--out-dir", str(args.chunk_out_dir)])
    if args.chunk_workers:
        chunk_args.extend(["--workers", str(args.chunk_workers)])
    if args.chunk_min_tokens:
        chunk_args.extend(["--min-tokens", str(args.chunk_min_tokens)])
    if args.chunk_max_tokens:
        chunk_args.extend(["--max-tokens", str(args.chunk_max_tokens)])
    if args.structural_markers:
        chunk_args.extend(["--structural-markers", str(args.structural_markers)])

    exit_code = _run_chunk(chunk_args)
    if exit_code != 0:
        logger.error(
            "Chunk stage failed",
            extra={"extra_fields": {"exit_code": exit_code}},
        )
        return exit_code

    embed_args: List[str] = []
    if args.data_root:
        embed_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        embed_args.append("--resume")
    if args.force:
        embed_args.append("--force")
    if args.chunk_out_dir:
        embed_args.extend(["--chunks-dir", str(args.chunk_out_dir)])
    if args.embed_out_dir:
        embed_args.extend(["--out-dir", str(args.embed_out_dir)])
    if args.embed_offline:
        embed_args.append("--offline")
    if args.embed_validate_only:
        embed_args.append("--validate-only")
    if args.splade_zero_pct_warn_threshold is not None:
        embed_args.extend(
            ["--splade-zero-pct-warn-threshold", str(args.splade_zero_pct_warn_threshold)]
        )

    exit_code = _run_embed(embed_args)
    if exit_code != 0:
        logger.error(
            "Embedding stage failed",
            extra={"extra_fields": {"exit_code": exit_code}},
        )
        return exit_code

    logger.info("docparse all completed", extra={"extra_fields": {"status": "success"}})
    return 0


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
    "all": _Command(_run_all, "Run doctags → chunk → embed sequentially"),
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


def run_all(argv: Sequence[str] | None = None) -> int:
    """Programmatic helper mirroring ``docparse all``.

    Args:
        argv: Optional argument vector supplied for orchestration.

    Returns:
        Process exit code returned by the pipeline orchestrator.
    """

    return _run_all([] if argv is None else list(argv))


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
