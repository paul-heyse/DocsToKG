"""Unified command-line interface orchestrating DocParsing stages.

The core CLI module wires together subcommand parsers, shared validation
logic, manifest inspection tooling, and execution helpers for the DocTags,
chunking, embedding, planning, and diagnostics workflows. It keeps operator
experience consistent across orchestrated runs by centralising option parsing,
error reporting, telemetry emission, and resume handling—whether the CLI is
invoked directly or through automation.
"""

from __future__ import annotations

import argparse
import builtins
import json
import sys
import textwrap
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import typer
from typing_extensions import Annotated, Literal

from DocsToKG.DocParsing.cli_errors import (
    CLIValidationError,
    DoctagsCLIValidationError,
    format_cli_error,
)
from DocsToKG.DocParsing.config import StageConfigBase
from DocsToKG.DocParsing.env import (
    data_doctags,
    data_html,
    data_manifests,
    data_pdfs,
    detect_data_root,
)
from DocsToKG.DocParsing.io import iter_manifest_entries
from DocsToKG.DocParsing.logging import get_logger, log_event

from .cli_utils import (
    HTML_SUFFIXES,
    PDF_SUFFIXES,
    detect_mode,
    directory_contains_suffixes,
    merge_args,
)
from .models import DEFAULT_TOKENIZER
from .planning import display_plan, plan_chunk, plan_doctags, plan_embed

CommandHandler = Callable[[Sequence[str]], int]

try:  # Optional chunking extras may not be installed
    from DocsToKG.DocParsing.chunking.config import SOFT_BARRIER_MARGIN
except Exception:  # pragma: no cover - fallback when chunking optional deps missing
    SOFT_BARRIER_MARGIN = 64

try:  # Optional embedding extras may not be installed
    from DocsToKG.DocParsing.embedding.config import SPLADE_SPARSITY_WARN_THRESHOLD_PCT
except Exception:  # pragma: no cover - fallback when embedding optional deps missing
    SPLADE_SPARSITY_WARN_THRESHOLD_PCT = 1.0


class _ManifestHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help formatter that avoids hyphenated aliases being split across lines."""

    def __init__(self, prog: str) -> None:
        """Initialise the formatter with a wider wrap to keep alias names intact."""

        super().__init__(prog, width=120)

    def _split_lines(self, text: str, width: int) -> list[str]:
        """Wrap help text without breaking on intra-stage hyphens."""

        lines: list[str] = []
        for paragraph in text.splitlines():
            normalized = self._whitespace_matcher.sub(" ", paragraph).strip()
            if not normalized:
                lines.append("")
                continue
            wrapper = textwrap.TextWrapper(width=width, break_on_hyphens=False)
            lines.extend(wrapper.wrap(normalized))
        return lines


# NOTE: ``known_stages`` MUST remain in sync with the manifest filenames under
# ``Data/Manifests``. The values are derived from the canonical filenames to
# avoid drifting stage identifiers across the CLI and the pipeline writers.
_MANIFEST_FILENAMES = (
    "docparse.doctags-html.manifest.jsonl",
    "docparse.doctags-pdf.manifest.jsonl",
    "docparse.chunks.manifest.jsonl",
    "docparse.embeddings.manifest.jsonl",
)
known_stages = [filename.split(".")[1] for filename in _MANIFEST_FILENAMES]
known_stage_set = frozenset(known_stages)
STAGE_ALIASES: Dict[str, Sequence[str]] = {
    "doctags": ("doctags-html", "doctags-pdf"),
    "chunk": ("chunks",),
    "embed": ("embeddings",),
}

CLI_DESCRIPTION = """\
Unified DocParsing CLI

Examples:
  python -m DocsToKG.DocParsing.core.cli all --resume
  python -m DocsToKG.DocParsing.core.cli chunk --data-root Data
  python -m DocsToKG.DocParsing.core.cli embed --resume
  python -m DocsToKG.DocParsing.core.cli doctags --mode pdf --workers 2
  python -m DocsToKG.DocParsing.core.cli token-profiles --doctags-dir Data/DocTagsFiles
  python -m DocsToKG.DocParsing.core.cli manifest --stage chunk --tail 10
  python -m DocsToKG.DocParsing.core.cli plan --data-root Data --mode auto
"""

app = typer.Typer(
    help=CLI_DESCRIPTION.strip(),
    add_completion=True,
    rich_markup_mode="markdown",
)

__all__ = [
    "CLI_DESCRIPTION",
    "CommandHandler",
    "app",
    "build_doctags_parser",
    "chunk",
    "doctags",
    "embed",
    "main",
    "manifest",
    "plan",
    "run_all",
    "token_profiles",
]


def _run_stage(handler: Callable[[Sequence[str]], int], argv: Sequence[str] | None = None) -> int:
    """Execute a stage handler while normalising CLI validation errors."""

    try:
        return handler([] if argv is None else list(argv))
    except CLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2


def build_doctags_parser(prog: str = "docparse doctags") -> argparse.ArgumentParser:
    """Create an :mod:`argparse` parser configured for DocTags conversion."""

    from DocsToKG.DocParsing import doctags as doctags_module

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
        "--log-level",
        type=lambda value: str(value).upper(),
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity applied to the DocTags stage (default: %(default)s).",
    )
    doctags_module.add_data_root_option(parser)
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
        "--vllm-wait-timeout",
        type=int,
        default=None,
        help=(
            "Seconds to wait for vLLM readiness before failing (PDF mode only; "
            "defaults to the PDF runner setting)"
        ),
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
    doctags_module.add_resume_force_options(
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


def _resolve_doctags_paths(args: argparse.Namespace) -> tuple[str, Path, Path, str]:
    """Resolve DocTags input/output directories and mode."""

    resolved_root = (
        detect_data_root(args.data_root) if args.data_root is not None else detect_data_root()
    )

    html_default_in = data_html(resolved_root, ensure=False)
    pdf_default_in = data_pdfs(resolved_root, ensure=False)
    doctags_default_out = data_doctags(resolved_root, ensure=False)

    mode = args.mode
    if args.in_dir is not None:
        input_dir = args.in_dir.expanduser().resolve()
        if mode == "auto":
            try:
                mode = detect_mode(input_dir)
            except ValueError as exc:
                raise DoctagsCLIValidationError(
                    option="--mode",
                    message=str(exc),
                    hint="Specify --mode html or --mode pdf to override auto-detection",
                ) from exc
    else:
        if mode == "auto":
            html_present = directory_contains_suffixes(html_default_in, HTML_SUFFIXES)
            pdf_present = directory_contains_suffixes(pdf_default_in, PDF_SUFFIXES)
            if html_present and not pdf_present:
                mode = "html"
            elif pdf_present and not html_present:
                mode = "pdf"
            elif html_present and pdf_present:
                raise DoctagsCLIValidationError(
                    option="--mode",
                    message=(
                        "Cannot auto-detect mode: found HTML sources in "
                        f"{html_default_in} and PDF sources in {pdf_default_in}"
                    ),
                    hint="Specify --mode html or --mode pdf to disambiguate the sources",
                )
            else:
                raise DoctagsCLIValidationError(
                    option="--mode",
                    message=(
                        "Cannot auto-detect mode: expected HTML files in "
                        f"{html_default_in} or PDF files in {pdf_default_in}"
                    ),
                    hint="Provide --input or set --mode html/--mode pdf explicitly",
                )
        input_dir = html_default_in if mode == "html" else pdf_default_in

    output_dir = (
        args.out_dir.expanduser().resolve() if args.out_dir is not None else doctags_default_out
    )
    return mode, input_dir, output_dir, str(resolved_root)


def doctags(argv: Sequence[str] | None = None) -> int:
    """Execute the DocTags conversion subcommand."""

    from DocsToKG.DocParsing import doctags as doctags_module

    parser = build_doctags_parser()
    parsed = parser.parse_args([] if argv is None else list(argv))
    logger = get_logger(__name__, level=parsed.log_level)

    raw_served_names = parsed.served_model_names
    normalized_served_model_names = doctags_module._normalize_served_model_names(raw_served_names)
    has_served_name_overrides = raw_served_names is not None
    parsed.served_model_names = (
        normalized_served_model_names if has_served_name_overrides else raw_served_names
    )

    try:
        mode, input_dir, output_dir, resolved_root = _resolve_doctags_paths(parsed)
    except CLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2

    parsed.in_dir = input_dir
    parsed.out_dir = output_dir

    raw_served_model_names = parsed.served_model_names
    normalized_served_model_names: tuple[str, ...] | None = None
    if raw_served_model_names is not None:
        coerced_served_model_names = StageConfigBase._coerce_str_tuple(raw_served_model_names, None)
        normalized_served_model_names = doctags_module._normalize_served_model_names(
            coerced_served_model_names
        )
    parsed.served_model_names = normalized_served_model_names

    logger.info(
        "Unified DocTags conversion",
        extra={
            "extra_fields": {
                "mode": mode,
                "data_root": resolved_root,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "workers": parsed.workers,
                "resume": parsed.resume,
                "force": parsed.force,
                "overwrite": parsed.overwrite,
                "model": parsed.model,
                "served_model_names": normalized_served_model_names,
                "gpu_memory_utilization": parsed.gpu_memory_utilization,
                "vllm_wait_timeout": parsed.vllm_wait_timeout,
            }
        },
    )

    base_overrides = {
        "data_root": parsed.data_root,
        "input": input_dir,
        "output": output_dir,
        "workers": parsed.workers,
        "resume": parsed.resume,
        "force": parsed.force,
        "log_level": parsed.log_level,
    }

    if mode == "html":
        html_overrides = {**base_overrides, "overwrite": parsed.overwrite}
        html_args = merge_args(doctags_module.html_build_parser(), html_overrides)
        return doctags_module.html_main(html_args)

    overrides = {
        **base_overrides,
        "model": parsed.model,
        "served_model_names": (
            normalized_served_model_names if has_served_name_overrides else None
        ),
        "gpu_memory_utilization": parsed.gpu_memory_utilization,
        "vllm_wait_timeout": parsed.vllm_wait_timeout,
    }
    if parsed.vllm_wait_timeout is not None:
        overrides["vllm_wait_timeout"] = parsed.vllm_wait_timeout
    pdf_args = merge_args(doctags_module.pdf_build_parser(), overrides)
    return doctags_module.pdf_main(pdf_args)


def _build_doctags_cli_args(
    mode: str,
    log_level: str,
    data_root: Optional[Path],
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    workers: Optional[int],
    model: Optional[str],
    vllm_wait_timeout: Optional[int],
    served_model_names: Sequence[str],
    gpu_memory_utilization: Optional[float],
    resume: bool,
    force: bool,
    verify_hash: bool,
    overwrite: bool,
) -> List[str]:
    """Compose CLI argv for the DocTags subcommand based on Typer inputs."""
    argv: List[str] = []
    normalized_mode = mode.lower()
    normalized_log_level = log_level.upper()

    _append_option(argv, "--mode", normalized_mode, default="auto")
    _append_option(argv, "--log-level", normalized_log_level, default="INFO")
    _append_option(argv, "--data-root", data_root, formatter=str)
    _append_option(argv, "--input", input_dir, formatter=str)
    _append_option(argv, "--output", output_dir, formatter=str)
    _append_option(argv, "--workers", workers)
    _append_option(argv, "--model", model)
    _append_option(argv, "--vllm-wait-timeout", vllm_wait_timeout)
    _append_multi_values(argv, "--served-model-name", served_model_names)
    _append_option(argv, "--gpu-memory-utilization", gpu_memory_utilization)
    _append_flag(argv, "--resume", resume)
    _append_flag(argv, "--force", force)
    _append_flag(argv, "--verify-hash", verify_hash)
    _append_flag(argv, "--overwrite", overwrite)
    return argv


def _build_chunk_cli_args(
    data_root: Optional[Path],
    config: Optional[Path],
    profile: Optional[str],
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    min_tokens: int,
    max_tokens: int,
    log_level: str,
    tokenizer_model: Optional[str],
    soft_barrier_margin: int,
    structural_markers: Optional[Path],
    serializer_provider: Optional[str],
    workers: int,
    shard_count: int,
    shard_index: int,
    validate_only: bool,
    inject_anchors: bool,
    resume: bool,
    force: bool,
    verify_hash: bool,
) -> List[str]:
    """Compose CLI argv for the chunk subcommand based on Typer inputs."""
    argv: List[str] = []
    normalized_log_level = log_level.upper()

    _append_option(argv, "--data-root", data_root, formatter=str)
    _append_option(argv, "--config", config, formatter=str)
    _append_option(argv, "--profile", profile)
    _append_option(argv, "--in-dir", input_dir, formatter=str)
    _append_option(argv, "--out-dir", output_dir, formatter=str)
    _append_option(argv, "--min-tokens", min_tokens, default=256)
    _append_option(argv, "--max-tokens", max_tokens, default=512)
    _append_option(argv, "--log-level", normalized_log_level, default="INFO")
    _append_option(argv, "--tokenizer-model", tokenizer_model)
    _append_option(argv, "--soft-barrier-margin", soft_barrier_margin, default=SOFT_BARRIER_MARGIN)
    _append_option(argv, "--structural-markers", structural_markers, formatter=str)
    _append_option(argv, "--serializer-provider", serializer_provider)
    _append_option(argv, "--workers", workers, default=1)
    _append_option(argv, "--shard-count", shard_count, default=1)
    _append_option(argv, "--shard-index", shard_index, default=0)
    _append_flag(argv, "--validate-only", validate_only)
    _append_flag(argv, "--inject-anchors", inject_anchors)
    _append_flag(argv, "--resume", resume)
    _append_flag(argv, "--force", force)
    _append_flag(argv, "--verify-hash", verify_hash)
    return argv


def _build_embed_cli_args(
    data_root: Optional[Path],
    config: Optional[Path],
    profile: Optional[str],
    log_level: str,
    no_cache: bool,
    shard_count: int,
    shard_index: int,
    chunks_dir: Optional[Path],
    out_dir: Optional[Path],
    vector_format: str,
    bm25_k1: float,
    bm25_b: float,
    batch_size_splade: int,
    batch_size_qwen: int,
    splade_max_active_dims: Optional[int],
    splade_model_dir: Optional[Path],
    splade_attn: str,
    qwen_dtype: str,
    qwen_quant: Optional[str],
    qwen_model_dir: Optional[Path],
    qwen_dim: int,
    tensor_parallel: int,
    sparsity_warn_threshold_pct: float,
    sparsity_report_top_n: int,
    files_parallel: int,
    validate_only: bool,
    plan_only: bool,
    offline: bool,
    resume: bool,
    force: bool,
    verify_hash: bool,
) -> List[str]:
    """Compose CLI argv for the embed subcommand based on Typer inputs."""
    argv: List[str] = []
    normalized_log_level = log_level.upper()

    _append_option(argv, "--data-root", data_root, formatter=str)
    _append_option(argv, "--config", config, formatter=str)
    _append_option(argv, "--profile", profile)
    _append_option(argv, "--log-level", normalized_log_level, default="INFO")
    _append_flag(argv, "--no-cache", no_cache)
    _append_option(argv, "--shard-count", shard_count, default=1)
    _append_option(argv, "--shard-index", shard_index, default=0)
    _append_option(argv, "--chunks-dir", chunks_dir, formatter=str)
    _append_option(argv, "--out-dir", out_dir, formatter=str)
    _append_option(argv, "--vector-format", vector_format, default="jsonl")
    _append_option(argv, "--bm25-k1", bm25_k1, default=1.5)
    _append_option(argv, "--bm25-b", bm25_b, default=0.75)
    _append_option(argv, "--batch-size-splade", batch_size_splade, default=32)
    _append_option(argv, "--batch-size-qwen", batch_size_qwen, default=64)
    _append_option(argv, "--splade-max-active-dims", splade_max_active_dims)
    _append_option(argv, "--splade-model-dir", splade_model_dir, formatter=str)
    _append_option(argv, "--splade-attn", splade_attn, default="auto")
    _append_option(argv, "--qwen-dtype", qwen_dtype, default="bfloat16")
    _append_option(argv, "--qwen-quant", qwen_quant)
    _append_option(argv, "--qwen-model-dir", qwen_model_dir, formatter=str)
    _append_option(argv, "--qwen-dim", qwen_dim, default=2560)
    _append_option(argv, "--tp", tensor_parallel, default=1)
    _append_option(argv, "--sparsity-warn-threshold-pct", sparsity_warn_threshold_pct, default=SPLADE_SPARSITY_WARN_THRESHOLD_PCT)
    _append_option(argv, "--sparsity-report-top-n", sparsity_report_top_n, default=10)
    _append_option(argv, "--files-parallel", files_parallel, default=1)
    _append_flag(argv, "--validate-only", validate_only)
    _append_flag(argv, "--plan-only", plan_only)
    _append_flag(argv, "--offline", offline)
    _append_flag(argv, "--resume", resume)
    _append_flag(argv, "--force", force)
    _append_flag(argv, "--verify-hash", verify_hash)
    return argv


def _build_token_profiles_cli_args(
    data_root: Optional[Path],
    config: Optional[Path],
    doctags_dir: Optional[Path],
    sample_size: int,
    max_chars: int,
    baseline: str,
    tokenizers: Sequence[str],
    window_min: int,
    window_max: int,
    log_level: str,
) -> List[str]:
    """Compose CLI argv for the token-profiles subcommand."""
    argv: List[str] = []
    normalized_log_level = log_level.upper()

    _append_option(argv, "--data-root", data_root, formatter=str)
    _append_option(argv, "--config", config, formatter=str)
    _append_option(argv, "--doctags-dir", doctags_dir, formatter=str)
    _append_option(argv, "--sample-size", sample_size, default=20)
    _append_option(argv, "--max-chars", max_chars, default=4000)
    _append_option(argv, "--baseline", baseline, default=DEFAULT_TOKENIZER)
    _append_multi_values(argv, "--tokenizer", tokenizers)
    _append_option(argv, "--window-min", window_min, default=256)
    _append_option(argv, "--window-max", window_max, default=512)
    _append_option(argv, "--log-level", normalized_log_level, default="INFO")
    return argv


def _build_manifest_cli_args(
    stages: Sequence[str],
    data_root: Optional[Path],
    tail: int,
    summarize: bool,
    raw: bool,
) -> List[str]:
    """Compose CLI argv for the manifest subcommand."""
    argv: List[str] = []
    for stage in stages:
        if stage:
            argv.extend(["--stage", stage])
    _append_option(argv, "--data-root", data_root, formatter=str)
    _append_option(argv, "--tail", tail, default=0)
    _append_flag(argv, "--summarize", summarize)
    _append_flag(argv, "--raw", raw)
    return argv


def _build_run_all_cli_args(
    data_root: Optional[Path],
    log_level: str,
    resume: bool,
    force: bool,
    mode: str,
    doctags_in_dir: Optional[Path],
    doctags_out_dir: Optional[Path],
    overwrite: bool,
    vllm_wait_timeout: Optional[int],
    chunk_out_dir: Optional[Path],
    chunk_workers: Optional[int],
    chunk_min_tokens: Optional[int],
    chunk_max_tokens: Optional[int],
    structural_markers: Optional[Path],
    chunk_shard_count: Optional[int],
    chunk_shard_index: Optional[int],
    embed_out_dir: Optional[Path],
    embed_offline: bool,
    embed_validate_only: bool,
    sparsity_warn_threshold_pct: Optional[float],
    embed_shard_count: Optional[int],
    embed_shard_index: Optional[int],
    embed_format: Optional[str],
    embed_no_cache: bool,
    plan_only: bool,
) -> List[str]:
    """Compose CLI argv for the all/plan subcommands."""
    argv: List[str] = []
    normalized_log_level = log_level.upper()
    normalized_mode = mode.lower()

    _append_option(argv, "--data-root", data_root, formatter=str)
    _append_option(argv, "--log-level", normalized_log_level, default="INFO")
    _append_flag(argv, "--resume", resume)
    _append_flag(argv, "--force", force)
    _append_option(argv, "--mode", normalized_mode, default="auto")
    _append_option(argv, "--doctags-in-dir", doctags_in_dir, formatter=str)
    _append_option(argv, "--doctags-out-dir", doctags_out_dir, formatter=str)
    _append_flag(argv, "--overwrite", overwrite)
    _append_option(argv, "--vllm-wait-timeout", vllm_wait_timeout)
    _append_option(argv, "--chunk-out-dir", chunk_out_dir, formatter=str)
    _append_option(argv, "--chunk-workers", chunk_workers)
    _append_option(argv, "--chunk-min-tokens", chunk_min_tokens)
    _append_option(argv, "--chunk-max-tokens", chunk_max_tokens)
    _append_option(argv, "--structural-markers", structural_markers, formatter=str)
    _append_option(argv, "--chunk-shard-count", chunk_shard_count)
    _append_option(argv, "--chunk-shard-index", chunk_shard_index)
    _append_option(argv, "--embed-out-dir", embed_out_dir, formatter=str)
    _append_flag(argv, "--embed-offline", embed_offline)
    _append_flag(argv, "--embed-validate-only", embed_validate_only)
    _append_option(argv, "--sparsity-warn-threshold-pct", sparsity_warn_threshold_pct)
    _append_option(argv, "--embed-shard-count", embed_shard_count)
    _append_option(argv, "--embed-shard-index", embed_shard_index)
    _append_option(argv, "--embed-format", embed_format)
    _append_flag(argv, "--embed-no-cache", embed_no_cache)
    _append_flag(argv, "--plan", plan_only)
    return argv


def _chunk_import_error_messages(exc: ImportError) -> List[str]:
    """Return user-facing error lines when the chunking module is unavailable."""

    missing = getattr(exc, "name", None)
    if not missing:
        message_text = str(exc)
        if message_text.startswith("No module named"):
            parts = message_text.split("'")
            missing = parts[1] if len(parts) >= 2 else message_text
        else:
            missing = message_text
    friendly_message = (
        "DocsToKG.DocParsing.chunking could not be imported because the optional "
        f"dependency '{missing}' is not installed. Install the appropriate extras, "
        'for example `pip install "DocsToKG[docling,gpu]"` to enable this module.'
    )
    follow_up = (
        "Optional DocTags/chunking dependencies are required for `docparse chunk`. "
        "Install them with `pip install DocsToKG[gpu12x]` or `pip install transformers`."
    )
    return [friendly_message, follow_up]


def _import_chunk_module():
    """Import the chunking module, refreshing the DocsToKG package cache."""

    import DocsToKG.DocParsing as docparsing_pkg

    docparsing_pkg._MODULE_CACHE.pop("chunking", None)
    sys.modules.pop("DocsToKG.DocParsing.chunking", None)
    docparsing_pkg.__dict__.pop("chunking", None)

    chunk_module = builtins.__import__("DocsToKG.DocParsing.chunking", fromlist=("chunking",))
    docparsing_pkg._MODULE_CACHE["chunking"] = chunk_module
    docparsing_pkg.__dict__["chunking"] = chunk_module
    return chunk_module


def chunk(argv: Sequence[str] | None = None) -> int:
    """Execute the Docling chunker subcommand."""

    try:
        chunk_module = _import_chunk_module()
    except ImportError as exc:
        for line in _chunk_import_error_messages(exc):
            print(line, file=sys.stderr)
        return 1

    parser = chunk_module.build_parser()
    parser.prog = "docparse chunk"
    args = parser.parse_args([] if argv is None else list(argv))
    try:
        return chunk_module.main(args)
    except CLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2


def embed(argv: Sequence[str] | None = None) -> int:
    """Execute the embedding pipeline subcommand."""

    from DocsToKG.DocParsing import embedding as embedding_module

    parser = embedding_module.build_parser()
    parser.prog = "docparse embed"
    args = parser.parse_args([] if argv is None else list(argv))
    try:
        return embedding_module.main(args)
    except CLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2


def token_profiles(argv: Sequence[str] | None = None) -> int:
    """Execute the tokenizer profiling subcommand."""

    try:
        from DocsToKG.DocParsing import token_profiles as token_profiles_module
    except ImportError as exc:  # pragma: no cover - exercised via CLI test
        root_cause = exc.__cause__
        if root_cause is not None:
            cause_message = str(root_cause)
            if cause_message:
                print(cause_message, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "Optional dependency 'transformers' is required for `docparse token-profiles`. "
            "Install it with `pip install transformers`.",
            file=sys.stderr,
        )
        return 1

    parser = token_profiles_module.build_parser()
    parser.prog = "docparse token-profiles"
    args = parser.parse_args([] if argv is None else list(argv))
    return token_profiles_module.main(args)


def plan(argv: Sequence[str] | None = None) -> int:
    """Display the doctags → chunk → embed plan without executing."""

    args = [] if argv is None else list(argv)
    if not any(option in args for option in ("--plan", "--plan-only")):
        args.append("--plan")
    return run_all(args)


def manifest(argv: Sequence[str] | None = None) -> int:
    """Inspect pipeline manifest artifacts via CLI."""

    return _run_stage(_manifest_main, argv)


def _build_manifest_parser() -> argparse.ArgumentParser:
    """Construct the manifest inspection parser."""

    parser = argparse.ArgumentParser(
        prog="docparse manifest",
        description="Inspect DocParsing manifest artifacts",
        formatter_class=_ManifestHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        default=None,
        help=(
            "Manifest stage to inspect (repeatable). Supported stages: doctags-html, "
            "doctags-pdf, chunks, embeddings.\n"
            "Aliases: 'doctags' selects doctags-html and doctags-pdf; 'chunk' selects chunks;\n"
            "'embed' selects embeddings. Defaults to stages discovered from manifest "
            "files; falls back to embeddings when no manifests are present."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="DocsToKG data root override used when resolving manifests",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Print the last N manifest entries",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print per-stage status and duration summary",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output tail entries as JSON instead of human-readable text",
    )
    return parser


def _manifest_main(argv: Sequence[str]) -> int:
    """Implementation for the ``docparse manifest`` command."""

    parser = _build_manifest_parser()
    args = parser.parse_args(list(argv))
    manifest_dir = data_manifests(args.data_root, ensure=False)
    logger = get_logger(__name__, base_fields={"stage": "manifest"})

    if not manifest_dir.exists():
        log_event(
            logger,
            "warning",
            "Manifest directory is missing",
            stage="manifest",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="MANIFEST_DIR_MISSING",
            manifest_dir=str(manifest_dir),
            data_root=str(args.data_root) if args.data_root is not None else None,
        )
        print(
            "No manifest directory found. Run a DocParsing stage to generate manifests.",
        )
        return 0

    discovered: List[str] = []
    for path in sorted(manifest_dir.glob("docparse.*.manifest.jsonl")):
        parts = path.name.split(".")
        if len(parts) >= 4:
            stage = parts[1]
            if stage not in discovered:
                discovered.append(stage)

    allowed_stage_set = set(known_stage_set).union(discovered)
    canonical_display = ", ".join(sorted(known_stage_set))

    if args.stages:
        seen: List[str] = []
        for raw_stage in args.stages:
            trimmed = raw_stage.strip()
            if not trimmed:
                continue
            normalized = trimmed.lower()
            alias_targets = STAGE_ALIASES.get(normalized)
            if alias_targets is None:
                resolved = [normalized] if normalized in allowed_stage_set else []
            else:
                resolved = [stage for stage in alias_targets if stage in allowed_stage_set]
                if not resolved and normalized in allowed_stage_set:
                    resolved = [normalized]
            if not resolved:
                discovered_message = ", ".join(discovered) if discovered else "<none>"
                raise CLIValidationError(
                    option="--stage",
                    message=(
                        "Unsupported stage "
                        f"'{trimmed}'. Canonical stages: {canonical_display}. "
                        f"Discovered stages: {discovered_message}"
                    ),
                    hint="Choose a supported manifest stage.",
                )
            for stage in resolved:
                if stage not in seen:
                    seen.append(stage)
        stages = seen
    else:
        stages = discovered
    if not stages:
        stages = ["embeddings"]
    summary_order = list(stages)

    tail_count = max(0, int(args.tail))
    need_summary = bool(args.summarize or not tail_count)
    tail_entries: Deque[Dict[str, Any]] = deque(maxlen=tail_count or None)
    status_counter: Optional[OrderedDict[str, Counter[str]]] = None
    duration_totals: Optional[OrderedDict[str, float]] = None
    total_entries: Optional[OrderedDict[str, int]] = None
    if need_summary:
        status_counter = OrderedDict((stage, Counter()) for stage in summary_order)
        duration_totals = OrderedDict((stage, 0.0) for stage in summary_order)
        total_entries = OrderedDict((stage, 0) for stage in summary_order)

    if tail_count and not need_summary:
        entry_iter = iter_manifest_entries(
            stages,
            args.data_root,
            limit=tail_count,
        )
    else:
        entry_iter = iter_manifest_entries(stages, args.data_root)

    entry_found = False
    for entry in entry_iter:
        entry_found = True
        if tail_count:
            tail_entries.append(entry)
        if (
            need_summary
            and total_entries is not None
            and status_counter is not None
            and duration_totals is not None
        ):
            stage = entry.get("stage", "unknown")
            if stage not in total_entries:
                total_entries[stage] = 0
                status_counter[stage] = Counter()
                duration_totals[stage] = 0.0
            status = entry.get("status", "unknown")
            total_entries[stage] += 1
            status_counter[stage][status] += 1
            try:
                duration_totals[stage] += float(entry.get("duration_s", 0.0))
            except (TypeError, ValueError):
                continue

    if not entry_found:
        log_event(
            logger,
            "warning",
            "No manifest entries located",
            stage="manifest",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="NO_MANIFEST_ENTRIES",
            stages=stages,
        )
        print("No manifest entries found for the requested stages.")
        return 0

    if tail_count:
        print(f"docparse manifest tail (last {len(tail_entries)} entries)")
        if args.raw:
            for entry in tail_entries:
                print(json.dumps(entry, ensure_ascii=False))
        else:
            for entry in tail_entries:
                timestamp = entry.get("timestamp", "")
                stage = entry.get("stage", "unknown")
                doc_id = entry.get("doc_id", "unknown")
                status = entry.get("status", "unknown")
                duration = entry.get("duration_s")
                line = f"{timestamp} [{stage}] {doc_id} status={status}"
                if duration is not None:
                    line += f" duration={duration}"
                error = entry.get("error")
                if error:
                    line += f" error={error}"
                print(line)

    if (
        need_summary
        and total_entries is not None
        and status_counter is not None
        and duration_totals is not None
    ):
        print("\nManifest summary")
        for stage, total in total_entries.items():
            duration = round(duration_totals[stage], 3)
            print(f"- {stage}: total={total} duration_s={duration}")
            status_map = status_counter[stage]
            if status_map:
                statuses = ", ".join(
                    f"{name}={count}" for name, count in sorted(status_map.items())
                )
                print(f"  statuses: {statuses}")

    log_event(
        logger,
        "info",
        "Manifest inspection completed",
        stage="manifest",
        doc_id="__aggregate__",
        input_hash=None,
        error_code="MANIFEST_OK",
        tail_count=tail_count,
        summarize=bool(args.summarize or not tail_count),
        stages=stages,
    )
    return 0


def _build_run_all_parser() -> argparse.ArgumentParser:
    """Create the parser shared by `docparse all` and `docparse plan`."""

    parser = argparse.ArgumentParser(
        prog="docparse all",
        description="Run doctags → chunk → embed in sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="DocsToKG data root override passed to all stages",
    )
    parser.add_argument(
        "--log-level",
        type=lambda value: str(value).upper(),
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity applied to all stages",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each stage by skipping outputs with matching manifests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration in each stage even when outputs exist",
    )
    parser.add_argument(
        "--mode", choices=["auto", "html", "pdf"], default="auto", help="DocTags conversion mode"
    )
    parser.add_argument(
        "--doctags-in-dir", type=Path, default=None, help="Override DocTags input directory"
    )
    parser.add_argument(
        "--doctags-out-dir", type=Path, default=None, help="Override DocTags output directory"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow rewriting DocTags outputs (HTML mode only)"
    )
    parser.add_argument(
        "--vllm-wait-timeout",
        type=int,
        default=None,
        help="Seconds to wait for vLLM readiness during the DocTags stage",
    )
    parser.add_argument(
        "--chunk-out-dir",
        type=Path,
        default=None,
        help="Output directory override for chunk JSONL files",
    )
    parser.add_argument(
        "--chunk-workers", type=int, default=None, help="Worker processes for the chunk stage"
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
        "--chunk-shard-count",
        type=int,
        default=None,
        help="Total number of shards for the chunk stage",
    )
    parser.add_argument(
        "--chunk-shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for the chunk stage",
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
        "--sparsity-warn-threshold-pct",
        dest="sparsity_warn_threshold_pct",
        type=float,
        default=None,
        help="Override SPLADE sparsity warning threshold for the embed stage",
    )
    parser.add_argument(
        "--embed-shard-count",
        type=int,
        default=None,
        help="Total number of shards for the embed stage (defaults to chunk shard count)",
    )
    parser.add_argument(
        "--embed-shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for the embed stage (defaults to chunk shard index)",
    )
    parser.add_argument(
        "--embed-format",
        choices=["jsonl", "parquet"],
        default=None,
        help="Vector output format for the embed stage",
    )
    parser.add_argument(
        "--embed-no-cache",
        action="store_true",
        help="Disable Qwen cache reuse during the embed stage",
    )
    parser.add_argument(
        "--plan",
        "--plan-only",
        action="store_true",
        help="Show a plan of the files each stage would touch instead of running",
    )
    return parser


def _build_stage_args(args: argparse.Namespace) -> tuple[List[str], List[str], List[str]]:
    """Construct argument lists for doctags/chunk/embed stages."""

    chunk_shard_count = args.chunk_shard_count
    chunk_shard_index = args.chunk_shard_index
    embed_shard_count = (
        args.embed_shard_count if args.embed_shard_count is not None else chunk_shard_count
    )
    embed_shard_index = (
        args.embed_shard_index if args.embed_shard_index is not None else args.chunk_shard_index
    )

    doctags_args: List[str] = ["--log-level", args.log_level]
    chunk_args: List[str] = ["--log-level", args.log_level]
    embed_args: List[str] = ["--log-level", args.log_level]

    if args.data_root:
        doctags_args.extend(["--data-root", str(args.data_root)])
        chunk_args.extend(["--data-root", str(args.data_root)])
        embed_args.extend(["--data-root", str(args.data_root)])
    if args.resume:
        doctags_args.append("--resume")
        chunk_args.append("--resume")
        embed_args.append("--resume")
    if args.force:
        doctags_args.append("--force")
        chunk_args.append("--force")
        embed_args.append("--force")

    if args.mode != "auto":
        doctags_args.extend(["--mode", args.mode])
    if args.doctags_in_dir:
        doctags_args.extend(["--in-dir", str(args.doctags_in_dir)])
    if args.doctags_out_dir:
        doctags_args.extend(["--out-dir", str(args.doctags_out_dir)])
        chunk_args.extend(["--in-dir", str(args.doctags_out_dir)])
    if args.overwrite:
        doctags_args.append("--overwrite")
    if args.vllm_wait_timeout is not None:
        doctags_args.extend(["--vllm-wait-timeout", str(args.vllm_wait_timeout)])

    if args.chunk_out_dir:
        chunk_args.extend(["--out-dir", str(args.chunk_out_dir)])
        embed_args.extend(["--chunks-dir", str(args.chunk_out_dir)])
    if args.chunk_workers is not None:
        chunk_args.extend(["--workers", str(args.chunk_workers)])
    if args.chunk_min_tokens is not None:
        chunk_args.extend(["--min-tokens", str(args.chunk_min_tokens)])
    if args.chunk_max_tokens is not None:
        chunk_args.extend(["--max-tokens", str(args.chunk_max_tokens)])
    if args.structural_markers:
        chunk_args.extend(["--structural-markers", str(args.structural_markers)])
    if chunk_shard_count is not None:
        chunk_args.extend(["--shard-count", str(chunk_shard_count)])
    if chunk_shard_index is not None:
        chunk_args.extend(["--shard-index", str(chunk_shard_index)])

    if args.embed_out_dir:
        embed_args.extend(["--out-dir", str(args.embed_out_dir)])
    if args.embed_offline:
        embed_args.append("--offline")
    if args.embed_validate_only:
        embed_args.append("--validate-only")
    if args.embed_format:
        embed_args.extend(["--format", args.embed_format])
    if args.embed_no_cache:
        embed_args.append("--no-cache")
    if embed_shard_count is not None:
        embed_args.extend(["--shard-count", str(embed_shard_count)])
    if embed_shard_index is not None:
        embed_args.extend(["--shard-index", str(embed_shard_index)])
    if args.sparsity_warn_threshold_pct is not None:
        embed_args.extend(["--sparsity-warn-threshold-pct", str(args.sparsity_warn_threshold_pct)])

    return doctags_args, chunk_args, embed_args


def run_all(argv: Sequence[str] | None = None) -> int:
    """Execute DocTags conversion, chunking, and embedding sequentially."""

    parser = _build_run_all_parser()

    args = parser.parse_args([] if argv is None else list(argv))
    plan_only = bool(args.plan)
    logger = get_logger(__name__, level=args.log_level)

    extra: Dict[str, Any] = {
        "resume": bool(args.resume),
        "force": bool(args.force),
        "mode": args.mode,
        "log_level": args.log_level,
    }
    if args.data_root:
        extra["data_root"] = str(args.data_root)
    for field_name in (
        "chunk_shard_count",
        "chunk_shard_index",
        "embed_shard_count",
        "embed_shard_index",
        "embed_format",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            extra[field_name] = value
    if plan_only:
        extra["plan_only"] = True
    message = "docparse plan preview" if plan_only else "docparse plan"
    logger.info(message, extra={"extra_fields": extra})

    doctags_args, chunk_args, embed_args = _build_stage_args(args)

    if args.plan:
        plans: List[Dict[str, Any]] = []
        try:
            plans.append(plan_doctags(doctags_args))
        except Exception as exc:  # pragma: no cover - defensive path
            plans.append(
                {
                    "stage": "doctags",
                    "mode": args.mode,
                    "input_dir": None,
                    "output_dir": None,
                    "process": [],
                    "skip": [],
                    "notes": [f"DocTags plan unavailable ({exc})"],
                }
            )
        try:
            plans.append(plan_chunk(chunk_args))
        except Exception as exc:  # pragma: no cover - defensive path
            plans.append(
                {
                    "stage": "chunk",
                    "input_dir": None,
                    "output_dir": None,
                    "process": [],
                    "skip": [],
                    "notes": [f"Chunk plan unavailable ({exc})"],
                }
            )
        try:
            plans.append(plan_embed(embed_args))
        except Exception as exc:  # pragma: no cover - defensive path
            plans.append(
                {
                    "stage": "embed",
                    "operation": "unknown",
                    "chunks_dir": None,
                    "vectors_dir": None,
                    "process": [],
                    "skip": [],
                    "notes": [f"Embed plan unavailable ({exc})"],
                }
            )
        display_plan(plans, stream=sys.stdout)
        return 0

    exit_code = doctags(doctags_args)
    if exit_code != 0:
        log_event(
            logger,
            "error",
            "DocTags stage failed",
            stage="docparse_all",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="DOCTAGS_STAGE_FAILED",
            exit_code=exit_code,
        )
        return exit_code

    exit_code = chunk(chunk_args)
    if exit_code != 0:
        log_event(
            logger,
            "error",
            "Chunk stage failed",
            stage="docparse_all",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="CHUNK_STAGE_FAILED",
            exit_code=exit_code,
        )
        return exit_code

    exit_code = embed(embed_args)
    if exit_code != 0:
        log_event(
            logger,
            "error",
            "Embedding stage failed",
            stage="docparse_all",
            doc_id="__aggregate__",
            input_hash=None,
            error_code="EMBED_STAGE_FAILED",
            exit_code=exit_code,
        )
        return exit_code

    logger.info("docparse all completed", extra={"extra_fields": {"status": "success"}})
    return 0


_DEFAULT_SENTINEL = object()


def _append_option(
    argv: List[str],
    flag: str,
    value: Any,
    *,
    formatter: Callable[[Any], str] = str,
    default: Any = _DEFAULT_SENTINEL,
) -> None:
    """Append an option and value when ``value`` is set and differs from ``default``."""

    if value is None:
        return
    if default is not _DEFAULT_SENTINEL and value == default:
        return
    argv.extend([flag, formatter(value)])


def _append_flag(argv: List[str], flag: str, enabled: bool) -> None:
    """Append a flag when ``enabled`` is True."""

    if enabled:
        argv.append(flag)


def _append_multi_values(
    argv: List[str], flag: str, values: Sequence[Any], *, formatter: Callable[[Any], str] = str
) -> None:
    """Append an option multiple times for each value."""

    if not values:
        return
    for item in values:
        argv.extend([flag, formatter(item)])


LogLevelOption = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
DoctagsModeOption = Literal["auto", "html", "pdf"]
VectorFormatOption = Literal["jsonl", "parquet"]
SpladeAttnOption = Literal["auto", "flash", "sdpa", "eager"]
QwenDTypeOption = Literal["float32", "bfloat16", "float16", "int8"]
ChunkProfileOption = Literal["cpu-small", "gpu-default", "gpu-max", "bert-compat"]
EmbedProfileOption = Literal["cpu-small", "gpu-default", "gpu-max"]


@app.command("doctags")
def _doctags_cli(
    mode: Annotated[
        DoctagsModeOption,
        typer.Option(
            "--mode",
            help="Select conversion backend; auto infers from input directory.",
        ),
    ] = "auto",
    log_level: Annotated[
        LogLevelOption,
        typer.Option(
            "--log-level",
            help="Logging verbosity applied to the DocTags stage.",
            show_default=True,
        ),
    ] = "INFO",
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="Override DocsToKG Data directory. Defaults to auto-detection or $DOCSTOKG_DATA_ROOT.",
        ),
    ] = None,
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "--in-dir",
            help="Directory containing HTML or PDF sources (defaults vary by mode).",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "--out-dir",
            help="Destination for generated .doctags files (defaults vary by mode).",
        ),
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option(
            "--workers",
            help="Worker processes to launch; backend defaults used when omitted.",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            help="Override vLLM model path or identifier for PDF conversion.",
        ),
    ] = None,
    vllm_wait_timeout: Annotated[
        Optional[int],
        typer.Option(
            "--vllm-wait-timeout",
            help="Seconds to wait for vLLM readiness before failing (PDF mode only; defaults to the PDF runner setting).",
        ),
    ] = None,
    served_model_name: Annotated[
        Optional[List[str]],
        typer.Option(
            "--served-model-name",
            metavar="NAME",
            help="Model alias to expose from vLLM (repeatable).",
        ),
    ] = None,
    gpu_memory_utilization: Annotated[
        Optional[float],
        typer.Option(
            "--gpu-memory-utilization",
            help="Fraction of GPU memory allocated to the vLLM server.",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip documents whose outputs already exist with matching content hash.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force reprocessing even when resume criteria are satisfied.",
        ),
    ] = False,
    verify_hash: Annotated[
        bool,
        typer.Option(
            "--verify-hash",
            help="Recompute input hashes before skipping resumed items. This validates manifest entries at the cost of additional I/O.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite existing DocTags files (HTML mode only).",
        ),
    ] = False,
    pdf: Annotated[
        bool,
        typer.Option(
            "--pdf",
            help="Alias for --mode pdf.",
            hidden=True,
        ),
    ] = False,
    html: Annotated[
        bool,
        typer.Option(
            "--html",
            help="Alias for --mode html.",
            hidden=True,
        ),
    ] = False,
) -> None:
    """Typer command implementation for `docparse doctags`."""
    if pdf and html:
        raise typer.BadParameter("Cannot combine --pdf and --html aliases.")
    effective_mode = "pdf" if pdf else "html" if html else mode

    argv = _build_doctags_cli_args(
        effective_mode,
        log_level,
        data_root,
        input_dir,
        output_dir,
        workers,
        model,
        vllm_wait_timeout,
        served_model_name or [],
        gpu_memory_utilization,
        resume,
        force,
        verify_hash,
        overwrite,
    )
    exit_code = doctags(argv)
    raise typer.Exit(code=exit_code)


@app.command("chunk")
def _chunk_cli(
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="DocsToKG data root override used when resolving inputs.",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Path to stage config file (JSON/YAML/TOML).",
        ),
    ] = None,
    profile: Annotated[
        Optional[ChunkProfileOption],
        typer.Option(
            "--profile",
            help="Preset for workers/token windows/tokenizer (cpu-small, gpu-default, gpu-max, bert-compat).",
        ),
    ] = None,
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--in-dir",
            help="DocTags input directory (defaults to data_root/DocTagsFiles).",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            help="Chunk output directory (defaults to data_root/ChunkedDocTagFiles).",
        ),
    ] = None,
    min_tokens: Annotated[
        int,
        typer.Option(
            "--min-tokens",
            help="Minimum tokens per chunk passed to the chunk stage.",
            show_default=True,
        ),
    ] = 256,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Maximum tokens per chunk passed to the chunk stage.",
            show_default=True,
        ),
    ] = 512,
    log_level: Annotated[
        LogLevelOption,
        typer.Option(
            "--log-level",
            help="Logging verbosity for console output.",
            show_default=True,
        ),
    ] = "INFO",
    tokenizer_model: Annotated[
        Optional[str],
        typer.Option(
            "--tokenizer-model",
            help="Tokenizer identifier used to compute token windows (defaults to profile/default).",
        ),
    ] = None,
    soft_barrier_margin: Annotated[
        int,
        typer.Option(
            "--soft-barrier-margin",
            help="Token margin to retain around soft barriers.",
            show_default=True,
        ),
    ] = SOFT_BARRIER_MARGIN,
    structural_markers: Annotated[
        Optional[Path],
        typer.Option(
            "--structural-markers",
            "--heading-markers",
            help="Optional path to structural marker overrides JSON.",
        ),
    ] = None,
    serializer_provider: Annotated[
        Optional[str],
        typer.Option(
            "--serializer-provider",
            help="Serializer implementation to persist chunk outputs (default provider).",
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help="Worker processes for the chunk stage.",
            show_default=True,
        ),
    ] = 1,
    shard_count: Annotated[
        int,
        typer.Option(
            "--shard-count",
            help="Total number of shards for the chunk stage.",
            show_default=True,
        ),
    ] = 1,
    shard_index: Annotated[
        int,
        typer.Option(
            "--shard-index",
            help="Zero-based shard index for the chunk stage.",
            show_default=True,
        ),
    ] = 0,
    validate_only: Annotated[
        bool,
        typer.Option(
            "--validate-only",
            help="Validate existing outputs and exit.",
        ),
    ] = False,
    inject_anchors: Annotated[
        bool,
        typer.Option(
            "--inject-anchors",
            help="Inject anchor metadata into outputs.",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip DocTags whose chunk outputs already exist with matching hash.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force reprocessing even when resume criteria are satisfied.",
        ),
    ] = False,
    verify_hash: Annotated[
        bool,
        typer.Option(
            "--verify-hash",
            help="Recompute input hashes before skipping resumed items. This validates manifest entries at the cost of additional I/O.",
        ),
    ] = False,
) -> None:
    """Typer command implementation for `docparse chunk`."""
    argv = _build_chunk_cli_args(
        data_root,
        config,
        profile,
        input_dir,
        output_dir,
        min_tokens,
        max_tokens,
        log_level,
        tokenizer_model,
        soft_barrier_margin,
        structural_markers,
        serializer_provider,
        workers,
        shard_count,
        shard_index,
        validate_only,
        inject_anchors,
        resume,
        force,
        verify_hash,
    )
    exit_code = chunk(argv)
    raise typer.Exit(code=exit_code)


@app.command("embed")
def _embed_cli(
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="DocsToKG data root override passed to the embedding stage.",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Path to stage config file (JSON/YAML/TOML).",
        ),
    ] = None,
    profile: Annotated[
        Optional[EmbedProfileOption],
        typer.Option(
            "--profile",
            help="Preset controlling batch sizes, SPLADE backend, and Qwen settings.",
        ),
    ] = None,
    log_level: Annotated[
        LogLevelOption,
        typer.Option(
            "--log-level",
            help="Logging verbosity for console output.",
            show_default=True,
        ),
    ] = "INFO",
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Disable Qwen LLM caching between batches (debug).",
        ),
    ] = False,
    shard_count: Annotated[
        int,
        typer.Option(
            "--shard-count",
            help="Total number of shards for distributed runs.",
            show_default=True,
        ),
    ] = 1,
    shard_index: Annotated[
        int,
        typer.Option(
            "--shard-index",
            help="Zero-based shard index to process.",
            show_default=True,
        ),
    ] = 0,
    chunks_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--chunks-dir",
            help="Override path to chunk files (auto-detected relative to data root).",
        ),
    ] = None,
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "--vectors-dir",
            help="Directory where vector outputs will be written (auto-detected).",
        ),
    ] = None,
    vector_format: Annotated[
        VectorFormatOption,
        typer.Option(
            "--vector-format",
            "--format",
            help="Vector output format.",
            show_default=True,
        ),
    ] = "jsonl",
    bm25_k1: Annotated[
        float,
        typer.Option("--bm25-k1", help="BM25 k1 parameter.", show_default=True),
    ] = 1.5,
    bm25_b: Annotated[
        float,
        typer.Option("--bm25-b", help="BM25 b parameter.", show_default=True),
    ] = 0.75,
    batch_size_splade: Annotated[
        int,
        typer.Option(
            "--batch-size-splade", help="SPLADE batch size.", show_default=True
        ),
    ] = 32,
    batch_size_qwen: Annotated[
        int,
        typer.Option("--batch-size-qwen", help="Qwen batch size.", show_default=True),
    ] = 64,
    splade_max_active_dims: Annotated[
        Optional[int],
        typer.Option(
            "--splade-max-active-dims",
            help="Optional SPLADE sparsity cap.",
        ),
    ] = None,
    splade_model_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--splade-model-dir",
            help="Explicit path to the SPLADE model directory (defaults to DocsToKG cache).",
        ),
    ] = None,
    splade_attn: Annotated[
        SpladeAttnOption,
        typer.Option(
            "--splade-attn",
            help="SPLADE attention backend preference order.",
            show_default=True,
        ),
    ] = "auto",
    qwen_dtype: Annotated[
        QwenDTypeOption,
        typer.Option("--qwen-dtype", help="Qwen inference dtype.", show_default=True),
    ] = "bfloat16",
    qwen_quant: Annotated[
        Optional[str],
        typer.Option("--qwen-quant", help="Optional Qwen quantisation preset (nf4, awq, ...)."),
    ] = None,
    qwen_model_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--qwen-model-dir",
            help="Explicit path to the Qwen model directory (defaults to DocsToKG cache).",
        ),
    ] = None,
    qwen_dim: Annotated[
        int,
        typer.Option("--qwen-dim", help="Qwen embedding dimension.", show_default=True),
    ] = 2560,
    tensor_parallel: Annotated[
        int,
        typer.Option("--tp", "--tensor-parallel", help="Tensor parallel degree.", show_default=True),
    ] = 1,
    sparsity_warn_threshold_pct: Annotated[
        float,
        typer.Option(
            "--sparsity-warn-threshold-pct",
            help="Override SPLADE sparsity warning threshold.",
            show_default=True,
        ),
    ] = SPLADE_SPARSITY_WARN_THRESHOLD_PCT,
    sparsity_report_top_n: Annotated[
        int,
        typer.Option(
            "--sparsity-report-top-n",
            help="Number of SPLADE terms to report in sparsity summaries.",
            show_default=True,
        ),
    ] = 10,
    files_parallel: Annotated[
        int,
        typer.Option(
            "--files-parallel",
            help="Process up to N chunk files concurrently during embedding.",
            show_default=True,
        ),
    ] = 1,
    validate_only: Annotated[
        bool,
        typer.Option(
            "--validate-only",
            help="Validate existing vectors in --out-dir and exit.",
        ),
    ] = False,
    plan_only: Annotated[
        bool,
        typer.Option(
            "--plan-only",
            help="Show resume/skip plan and exit without generating embeddings.",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Disable network access by setting TRANSFORMERS_OFFLINE=1. All models must already exist in local caches.",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip chunk files whose vector outputs already exist with matching hash.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force reprocessing even when resume criteria are satisfied.",
        ),
    ] = False,
    verify_hash: Annotated[
        bool,
        typer.Option(
            "--verify-hash",
            help="Recompute input hashes before skipping resumed items. This validates manifest entries at the cost of additional I/O.",
        ),
    ] = False,
) -> None:
    """Typer command implementation for `docparse embed`."""
    argv = _build_embed_cli_args(
        data_root,
        config,
        profile,
        log_level,
        no_cache,
        shard_count,
        shard_index,
        chunks_dir,
        out_dir,
        vector_format,
        bm25_k1,
        bm25_b,
        batch_size_splade,
        batch_size_qwen,
        splade_max_active_dims,
        splade_model_dir,
        splade_attn,
        qwen_dtype,
        qwen_quant,
        qwen_model_dir,
        qwen_dim,
        tensor_parallel,
        sparsity_warn_threshold_pct,
        sparsity_report_top_n,
        files_parallel,
        validate_only,
        plan_only,
        offline,
        resume,
        force,
        verify_hash,
    )
    exit_code = embed(argv)
    raise typer.Exit(code=exit_code)


@app.command("token-profiles")
def _token_profiles_cli(
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="DocsToKG data root override. Defaults to auto-detection or $DOCSTOKG_DATA_ROOT.",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Optional path to JSON/YAML/TOML config.",
        ),
    ] = None,
    doctags_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--doctags-dir",
            help="Directory containing DocTags files.",
        ),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            help="Number of DocTags files to sample (<=0 means all).",
            show_default=True,
        ),
    ] = 20,
    max_chars: Annotated[
        int,
        typer.Option(
            "--max-chars",
            help="Trim samples to this many characters (<=0 keeps full text).",
            show_default=True,
        ),
    ] = 4000,
    baseline: Annotated[
        str,
        typer.Option(
            "--baseline",
            help="Tokenizer treated as baseline for ratios.",
            show_default=True,
        ),
    ] = DEFAULT_TOKENIZER,
    tokenizer: Annotated[
        Optional[List[str]],
        typer.Option(
            "--tokenizer",
            metavar="NAME",
            help="Additional tokenizer identifier to profile (repeatable).",
        ),
    ] = None,
    window_min: Annotated[
        int,
        typer.Option(
            "--window-min",
            help="Reference min tokens scaled by observed ratios.",
            show_default=True,
        ),
    ] = 256,
    window_max: Annotated[
        int,
        typer.Option(
            "--window-max",
            help="Reference max tokens scaled by observed ratios.",
            show_default=True,
        ),
    ] = 512,
    log_level: Annotated[
        LogLevelOption,
        typer.Option(
            "--log-level",
            help="Logging verbosity for structured output.",
            show_default=True,
        ),
    ] = "INFO",
) -> None:
    """Typer command implementation for `docparse token-profiles`."""
    argv = _build_token_profiles_cli_args(
        data_root,
        config,
        doctags_dir,
        sample_size,
        max_chars,
        baseline,
        tokenizer or [],
        window_min,
        window_max,
        log_level,
    )
    exit_code = token_profiles(argv)
    raise typer.Exit(code=exit_code)


@app.command("manifest")
def _manifest_cli(
    stages: Annotated[
        Optional[List[str]],
        typer.Option(
            "--stage",
            help="Manifest stage to inspect (repeatable). Supported stages: doctags-html, doctags-pdf, chunks, embeddings.",
            metavar="STAGE",
        ),
    ] = None,
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="DocsToKG data root override used when resolving manifests.",
        ),
    ] = None,
    tail: Annotated[
        int,
        typer.Option("--tail", help="Print the last N manifest entries.", show_default=True),
    ] = 0,
    summarize: Annotated[
        bool,
        typer.Option(
            "--summarize",
            help="Print per-stage status and duration summary.",
        ),
    ] = False,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Output tail entries as JSON instead of human-readable text.",
        ),
    ] = False,
) -> None:
    """Typer command implementation for `docparse manifest`."""
    argv = _build_manifest_cli_args(stages or [], data_root, tail, summarize, raw)
    exit_code = manifest(argv)
    raise typer.Exit(code=exit_code)


@app.command("plan")
def _plan_cli(
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="DocsToKG data root override passed to all stages.",
        ),
    ] = None,
    log_level: Annotated[
        LogLevelOption,
        typer.Option(
            "--log-level",
            help="Logging verbosity applied to all stages.",
            show_default=True,
        ),
    ] = "INFO",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume each stage by skipping outputs with matching manifests.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force regeneration in each stage even when outputs exist.",
        ),
    ] = False,
    mode: Annotated[
        DoctagsModeOption,
        typer.Option(
            "--mode",
            help="DocTags conversion mode.",
            show_default=True,
        ),
    ] = "auto",
    doctags_in_dir: Annotated[
        Optional[Path],
        typer.Option("--doctags-in-dir", help="Override DocTags input directory."),
    ] = None,
    doctags_out_dir: Annotated[
        Optional[Path],
        typer.Option("--doctags-out-dir", help="Override DocTags output directory."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow rewriting DocTags outputs (HTML mode only).",
        ),
    ] = False,
    vllm_wait_timeout: Annotated[
        Optional[int],
        typer.Option(
            "--vllm-wait-timeout",
            help="Seconds to wait for vLLM readiness during the DocTags stage.",
        ),
    ] = None,
    chunk_out_dir: Annotated[
        Optional[Path],
        typer.Option("--chunk-out-dir", help="Output directory override for chunk JSONL files."),
    ] = None,
    chunk_workers: Annotated[
        Optional[int],
        typer.Option("--chunk-workers", help="Worker processes for the chunk stage."),
    ] = None,
    chunk_min_tokens: Annotated[
        Optional[int],
        typer.Option("--chunk-min-tokens", help="Minimum tokens per chunk passed to the chunk stage."),
    ] = None,
    chunk_max_tokens: Annotated[
        Optional[int],
        typer.Option("--chunk-max-tokens", help="Maximum tokens per chunk passed to the chunk stage."),
    ] = None,
    structural_markers: Annotated[
        Optional[Path],
        typer.Option(
            "--structural-markers",
            help="Structural marker configuration forwarded to the chunk stage.",
        ),
    ] = None,
    chunk_shard_count: Annotated[
        Optional[int],
        typer.Option("--chunk-shard-count", help="Total number of shards for the chunk stage."),
    ] = None,
    chunk_shard_index: Annotated[
        Optional[int],
        typer.Option("--chunk-shard-index", help="Zero-based shard index for the chunk stage."),
    ] = None,
    embed_out_dir: Annotated[
        Optional[Path],
        typer.Option("--embed-out-dir", help="Output directory override for embedding JSONL files."),
    ] = None,
    embed_offline: Annotated[
        bool,
        typer.Option(
            "--embed-offline",
            help="Run the embedding stage with TRANSFORMERS_OFFLINE=1.",
        ),
    ] = False,
    embed_validate_only: Annotated[
        bool,
        typer.Option(
            "--embed-validate-only",
            help="Skip embedding generation and only validate existing vectors.",
        ),
    ] = False,
    sparsity_warn_threshold_pct: Annotated[
        Optional[float],
        typer.Option(
            "--sparsity-warn-threshold-pct",
            help="Override SPLADE sparsity warning threshold for the embed stage.",
        ),
    ] = None,
    embed_shard_count: Annotated[
        Optional[int],
        typer.Option(
            "--embed-shard-count",
            help="Total number of shards for the embed stage (defaults to chunk shard count).",
        ),
    ] = None,
    embed_shard_index: Annotated[
        Optional[int],
        typer.Option(
            "--embed-shard-index",
            help="Zero-based shard index for the embed stage (defaults to chunk shard index).",
        ),
    ] = None,
    embed_format: Annotated[
        Optional[VectorFormatOption],
        typer.Option("--embed-format", help="Vector output format for the embed stage."),
    ] = None,
    embed_no_cache: Annotated[
        bool,
        typer.Option(
            "--embed-no-cache",
            help="Disable Qwen cache reuse during the embed stage.",
        ),
    ] = False,
) -> None:
    """Typer command implementation for `docparse plan`."""
    argv = _build_run_all_cli_args(
        data_root,
        log_level,
        resume,
        force,
        mode,
        doctags_in_dir,
        doctags_out_dir,
        overwrite,
        vllm_wait_timeout,
        chunk_out_dir,
        chunk_workers,
        chunk_min_tokens,
        chunk_max_tokens,
        structural_markers,
        chunk_shard_count,
        chunk_shard_index,
        embed_out_dir,
        embed_offline,
        embed_validate_only,
        sparsity_warn_threshold_pct,
        embed_shard_count,
        embed_shard_index,
        embed_format,
        embed_no_cache,
        plan_only=True,
    )
    exit_code = run_all(argv)
    raise typer.Exit(code=exit_code)


@app.command("all")
def _all_cli(
    data_root: Annotated[
        Optional[Path],
        typer.Option(
            "--data-root",
            help="DocsToKG data root override passed to all stages.",
        ),
    ] = None,
    log_level: Annotated[
        LogLevelOption,
        typer.Option(
            "--log-level",
            help="Logging verbosity applied to all stages.",
            show_default=True,
        ),
    ] = "INFO",
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume each stage by skipping outputs with matching manifests.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Force regeneration in each stage even when outputs exist.",
        ),
    ] = False,
    mode: Annotated[
        DoctagsModeOption,
        typer.Option(
            "--mode",
            help="DocTags conversion mode.",
            show_default=True,
        ),
    ] = "auto",
    doctags_in_dir: Annotated[
        Optional[Path],
        typer.Option("--doctags-in-dir", help="Override DocTags input directory."),
    ] = None,
    doctags_out_dir: Annotated[
        Optional[Path],
        typer.Option("--doctags-out-dir", help="Override DocTags output directory."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Allow rewriting DocTags outputs (HTML mode only).",
        ),
    ] = False,
    vllm_wait_timeout: Annotated[
        Optional[int],
        typer.Option(
            "--vllm-wait-timeout",
            help="Seconds to wait for vLLM readiness during the DocTags stage.",
        ),
    ] = None,
    chunk_out_dir: Annotated[
        Optional[Path],
        typer.Option("--chunk-out-dir", help="Output directory override for chunk JSONL files."),
    ] = None,
    chunk_workers: Annotated[
        Optional[int],
        typer.Option("--chunk-workers", help="Worker processes for the chunk stage."),
    ] = None,
    chunk_min_tokens: Annotated[
        Optional[int],
        typer.Option("--chunk-min-tokens", help="Minimum tokens per chunk passed to the chunk stage."),
    ] = None,
    chunk_max_tokens: Annotated[
        Optional[int],
        typer.Option("--chunk-max-tokens", help="Maximum tokens per chunk passed to the chunk stage."),
    ] = None,
    structural_markers: Annotated[
        Optional[Path],
        typer.Option(
            "--structural-markers",
            help="Structural marker configuration forwarded to the chunk stage.",
        ),
    ] = None,
    chunk_shard_count: Annotated[
        Optional[int],
        typer.Option("--chunk-shard-count", help="Total number of shards for the chunk stage."),
    ] = None,
    chunk_shard_index: Annotated[
        Optional[int],
        typer.Option("--chunk-shard-index", help="Zero-based shard index for the chunk stage."),
    ] = None,
    embed_out_dir: Annotated[
        Optional[Path],
        typer.Option("--embed-out-dir", help="Output directory override for embedding JSONL files."),
    ] = None,
    embed_offline: Annotated[
        bool,
        typer.Option(
            "--embed-offline",
            help="Run the embedding stage with TRANSFORMERS_OFFLINE=1.",
        ),
    ] = False,
    embed_validate_only: Annotated[
        bool,
        typer.Option(
            "--embed-validate-only",
            help="Skip embedding generation and only validate existing vectors.",
        ),
    ] = False,
    sparsity_warn_threshold_pct: Annotated[
        Optional[float],
        typer.Option(
            "--sparsity-warn-threshold-pct",
            help="Override SPLADE sparsity warning threshold for the embed stage.",
        ),
    ] = None,
    embed_shard_count: Annotated[
        Optional[int],
        typer.Option(
            "--embed-shard-count",
            help="Total number of shards for the embed stage (defaults to chunk shard count).",
        ),
    ] = None,
    embed_shard_index: Annotated[
        Optional[int],
        typer.Option(
            "--embed-shard-index",
            help="Zero-based shard index for the embed stage (defaults to chunk shard index).",
        ),
    ] = None,
    embed_format: Annotated[
        Optional[VectorFormatOption],
        typer.Option("--embed-format", help="Vector output format for the embed stage."),
    ] = None,
    embed_no_cache: Annotated[
        bool,
        typer.Option(
            "--embed-no-cache",
            help="Disable Qwen cache reuse during the embed stage.",
        ),
    ] = False,
    plan_only: Annotated[
        bool,
        typer.Option(
            "--plan",
            "--plan-only",
            help="Show a plan of the files each stage would touch instead of running.",
        ),
    ] = False,
) -> None:
    """Typer command implementation for `docparse all`."""
    argv = _build_run_all_cli_args(
        data_root,
        log_level,
        resume,
        force,
        mode,
        doctags_in_dir,
        doctags_out_dir,
        overwrite,
        vllm_wait_timeout,
        chunk_out_dir,
        chunk_workers,
        chunk_min_tokens,
        chunk_max_tokens,
        structural_markers,
        chunk_shard_count,
        chunk_shard_index,
        embed_out_dir,
        embed_offline,
        embed_validate_only,
        sparsity_warn_threshold_pct,
        embed_shard_count,
        embed_shard_index,
        embed_format,
        embed_no_cache,
        plan_only,
    )
    exit_code = run_all(argv)
    raise typer.Exit(code=exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by `python -m DocsToKG.DocParsing.core.cli`."""

    args = [] if argv is None else list(argv)
    app(args=args, prog_name="docparse")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
