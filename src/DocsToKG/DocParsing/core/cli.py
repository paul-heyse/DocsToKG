# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.core.cli",
#   "purpose": "Typer entry points for DocParsing core tools (planning preview).",
#   "sections": [
#     {
#       "id": "build-doctags-parser",
#       "name": "build_doctags_parser",
#       "anchor": "function-build-doctags-parser",
#       "kind": "function"
#     },
#     {
#       "id": "resolve-doctags-paths",
#       "name": "_resolve_doctags_paths",
#       "anchor": "function-resolve-doctags-paths",
#       "kind": "function"
#     },
#     {
#       "id": "plan-command",
#       "name": "plan",
#       "anchor": "function-plan",
#       "kind": "function"
#     },
#     {
#       "id": "main",
#       "name": "main",
#       "anchor": "function-main",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Typer CLI surfaces for DocParsing core helpers.

This module currently focuses on the ``docparse plan`` subcommand so operators
can preview DocTags, chunking, and embedding work in either the traditional
pretty-printed form or a structured JSON payload. The helpers also expose
``build_doctags_parser`` and ``_resolve_doctags_paths`` for reuse inside the
planning module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Annotated

import typer

from DocsToKG.DocParsing.cli_errors import DoctagsCLIValidationError

from .cli_utils import detect_mode

app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    help="DocParsing planning helpers and structured outputs.",
)

__all__ = [
    "app",
    "build_doctags_parser",
    "plan",
    "_resolve_doctags_paths",
    "main",
]


def build_doctags_parser() -> argparse.ArgumentParser:
    """Return a lightweight parser for DocTags planning options."""

    parser = argparse.ArgumentParser(description="DocTags planning options")
    parser.add_argument(
        "--mode",
        choices=("auto", "pdf", "html"),
        default="auto",
        help="Conversion mode to plan (auto|pdf|html)",
    )
    parser.add_argument(
        "--input",
        dest="in_dir",
        type=Path,
        default=None,
        help="DocTags input directory override",
    )
    parser.add_argument(
        "--output",
        dest="out_dir",
        type=Path,
        default=None,
        help="DocTags output directory override",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base data directory containing DocTags/Chunk/Vector folders",
    )
    parser.add_argument("--resume", action="store_true", help="Respect existing manifests")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Recompute input hashes when resuming to validate manifests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite HTML outputs when planning in html mode",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="DocTags log level override (INFO by default)",
    )
    return parser


def _resolve_doctags_paths(args: argparse.Namespace) -> tuple[str, Path, Path, str]:
    """Resolve DocTags mode and directories for planning."""

    from DocsToKG.DocParsing.env import (
        data_doctags,
        data_html,
        data_pdfs,
        detect_data_root,
        prepare_data_root,
        resolve_pipeline_path,
    )

    resolved_root = prepare_data_root(args.data_root, detect_data_root())
    data_root_overridden = args.data_root is not None

    pdf_default = data_pdfs(resolved_root, ensure=False)
    html_default = data_html(resolved_root, ensure=False)
    doctags_default = data_doctags(resolved_root, ensure=False)

    pdf_candidate = resolve_pipeline_path(
        cli_value=args.in_dir,
        default_path=pdf_default,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_pdfs(root, ensure=False),
    ).resolve()
    html_candidate = resolve_pipeline_path(
        cli_value=args.in_dir,
        default_path=html_default,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_html(root, ensure=False),
    ).resolve()

    mode = str(args.mode or "auto").lower()
    input_dir: Path
    if mode == "auto":
        candidates: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for candidate_mode, candidate_path in (("pdf", pdf_candidate), ("html", html_candidate)):
            if candidate_path not in seen:
                candidates.append((candidate_mode, candidate_path))
                seen.add(candidate_path)
        last_error: Exception | None = None
        input_dir = pdf_candidate
        for _candidate_mode, candidate_path in candidates:
            try:
                detected = detect_mode(candidate_path)
            except ValueError as exc:  # directory missing or ambiguous
                last_error = exc
                continue
            else:
                mode = detected
                input_dir = candidate_path
                break
        else:
            hint = (
                "Specify --mode explicitly or ensure the input directory contains "
                "either PDF or HTML files"
            )
            message = str(last_error) if last_error else "Unable to detect DocTags mode"
            raise DoctagsCLIValidationError(option="--mode", message=message, hint=hint)
    elif mode in {"pdf", "html"}:
        input_dir = pdf_candidate if mode == "pdf" else html_candidate
    else:
        raise DoctagsCLIValidationError(
            option="--mode",
            message=f"Unsupported mode '{mode}' (expected auto, pdf, or html)",
        )

    output_dir = resolve_pipeline_path(
        cli_value=args.out_dir,
        default_path=doctags_default,
        resolved_data_root=resolved_root,
        data_root_overridden=data_root_overridden,
        resolver=lambda root: data_doctags(root, ensure=False),
    ).resolve()

    return mode, input_dir, output_dir, str(resolved_root)


def _append_if_value(container: list[str], flag: str, value: Path | str | None) -> None:
    """Append ``flag`` and ``value`` to ``container`` when ``value`` is set."""

    if value is None:
        return
    container.extend([flag, str(value)])


@app.command()
def plan(
    *,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Base DocsToKG data directory override"),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="DocTags mode (auto|pdf|html)", case_sensitive=False),
    ] = "auto",
    input_dir: Annotated[
        Path | None,
        typer.Option("--input", help="DocTags input directory"),
    ] = None,
    doctags_output: Annotated[
        Path | None,
        typer.Option(
            "--doc-output",
            "--doctags-output",
            help="DocTags output directory",
        ),
    ] = None,
    doctags_resume: Annotated[
        bool,
        typer.Option("--doctags-resume/--no-doctags-resume", help="Resume DocTags stage"),
    ] = False,
    doctags_force: Annotated[
        bool,
        typer.Option("--doctags-force/--no-doctags-force", help="Force DocTags stage"),
    ] = False,
    doctags_verify_hash: Annotated[
        bool,
        typer.Option(
            "--doctags-verify-hash/--no-doctags-verify-hash",
            help="Recompute DocTags hashes when resuming",
        ),
    ] = False,
    doctags_overwrite: Annotated[
        bool,
        typer.Option(
            "--doctags-overwrite/--no-doctags-overwrite",
            help="Overwrite HTML outputs when planning",
        ),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="DocTags log level override"),
    ] = None,
    chunk_in_dir: Annotated[
        Path | None,
        typer.Option("--chunk-in-dir", help="Chunk stage input directory"),
    ] = None,
    chunk_out_dir: Annotated[
        Path | None,
        typer.Option("--chunk-out-dir", help="Chunk stage output directory"),
    ] = None,
    chunk_resume: Annotated[
        bool,
        typer.Option("--chunk-resume/--no-chunk-resume", help="Resume chunk stage"),
    ] = False,
    chunk_force: Annotated[
        bool,
        typer.Option("--chunk-force/--no-chunk-force", help="Force chunk stage"),
    ] = False,
    chunk_verify_hash: Annotated[
        bool,
        typer.Option(
            "--chunk-verify-hash/--no-chunk-verify-hash",
            help="Recompute chunk hashes when resuming",
        ),
    ] = False,
    embed_chunks_dir: Annotated[
        Path | None,
        typer.Option("--embed-chunks-dir", help="Embedding stage chunks directory"),
    ] = None,
    embed_out_dir: Annotated[
        Path | None,
        typer.Option("--embed-out-dir", help="Embedding stage vectors directory"),
    ] = None,
    embed_resume: Annotated[
        bool,
        typer.Option("--embed-resume/--no-embed-resume", help="Resume embedding stage"),
    ] = False,
    embed_force: Annotated[
        bool,
        typer.Option("--embed-force/--no-embed-force", help="Force embedding stage"),
    ] = False,
    embed_verify_hash: Annotated[
        bool,
        typer.Option(
            "--embed-verify-hash/--no-embed-verify-hash",
            help="Recompute embedding hashes when resuming",
        ),
    ] = False,
    embed_validate_only: Annotated[
        bool,
        typer.Option(
            "--embed-validate-only/--no-embed-validate-only",
            help="Validate embeddings without generating new vectors",
        ),
    ] = False,
    embed_vector_format: Annotated[
        str | None,
        typer.Option("--embed-vector-format", help="Embedding vector format (parquet|jsonl)"),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", help="Render output as 'pretty' text or 'json'", case_sensitive=False),
    ] = "pretty",
) -> None:
    """Preview DocParsing work for doctags, chunk, and embed stages."""

    output_mode = (output or "pretty").lower()
    if output_mode not in {"pretty", "json"}:
        raise typer.BadParameter("Output must be either 'pretty' or 'json'", param_hint="--output")

    doctags_args: list[str] = []
    if data_root is not None:
        _append_if_value(doctags_args, "--data-root", data_root)
    _append_if_value(doctags_args, "--mode", mode)
    _append_if_value(doctags_args, "--input", input_dir)
    _append_if_value(doctags_args, "--output", doctags_output)
    if doctags_resume:
        doctags_args.append("--resume")
    if doctags_force:
        doctags_args.append("--force")
    if doctags_verify_hash:
        doctags_args.append("--verify-hash")
    if doctags_overwrite:
        doctags_args.append("--overwrite")
    _append_if_value(doctags_args, "--log-level", log_level)

    chunk_args: list[str] = []
    if data_root is not None:
        _append_if_value(chunk_args, "--data-root", data_root)
    _append_if_value(chunk_args, "--in-dir", chunk_in_dir)
    _append_if_value(chunk_args, "--out-dir", chunk_out_dir)
    if chunk_resume:
        chunk_args.append("--resume")
    if chunk_force:
        chunk_args.append("--force")
    if chunk_verify_hash:
        chunk_args.append("--verify-hash")

    embed_args: list[str] = []
    if data_root is not None:
        _append_if_value(embed_args, "--data-root", data_root)
    _append_if_value(embed_args, "--chunks-dir", embed_chunks_dir)
    _append_if_value(embed_args, "--out-dir", embed_out_dir)
    if embed_resume:
        embed_args.append("--resume")
    if embed_force:
        embed_args.append("--force")
    if embed_verify_hash:
        embed_args.append("--verify-hash")
    if embed_validate_only:
        embed_args.append("--validate-only")
    _append_if_value(embed_args, "--format", embed_vector_format)

    from . import planning as planning_module

    plans = [
        planning_module.plan_doctags(doctags_args),
        planning_module.plan_chunk(chunk_args),
        planning_module.plan_embed(embed_args),
    ]
    planning_module.display_plan(plans, stream=sys.stdout, output_mode=output_mode)


def main() -> None:
    """Entry point used by ``python -m DocsToKG.DocParsing.core.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    main()
