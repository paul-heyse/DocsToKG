"""Unified CLI entry points for DocParsing stages."""

from __future__ import annotations

import argparse
import builtins
import json
import sys
import textwrap
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from DocsToKG.DocParsing.cli_errors import (
    CLIValidationError,
    DoctagsCLIValidationError,
    format_cli_error,
)
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
from .planning import display_plan, plan_chunk, plan_doctags, plan_embed

CommandHandler = Callable[[Sequence[str]], int]


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

__all__ = [
    "CLI_DESCRIPTION",
    "COMMANDS",
    "CommandHandler",
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

    output_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else doctags_default_out
    return mode, input_dir, output_dir, str(resolved_root)


def doctags(argv: Sequence[str] | None = None) -> int:
    """Execute the DocTags conversion subcommand."""

    from DocsToKG.DocParsing import doctags as doctags_module

    parser = build_doctags_parser()
    parsed = parser.parse_args([] if argv is None else list(argv))
    logger = get_logger(__name__, level=parsed.log_level)

    try:
        mode, input_dir, output_dir, resolved_root = _resolve_doctags_paths(parsed)
    except CLIValidationError as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 2

    parsed.in_dir = input_dir
    parsed.out_dir = output_dir

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
                "served_model_names": parsed.served_model_names,
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
        "served_model_names": parsed.served_model_names,
        "gpu_memory_utilization": parsed.gpu_memory_utilization,
        "vllm_wait_timeout": parsed.vllm_wait_timeout,
    }
    if parsed.vllm_wait_timeout is not None:
        overrides["vllm_wait_timeout"] = parsed.vllm_wait_timeout
    pdf_args = merge_args(doctags_module.pdf_build_parser(), overrides)
    return doctags_module.pdf_main(pdf_args)


def chunk(argv: Sequence[str] | None = None) -> int:
    """Execute the Docling chunker subcommand."""

    try:
        import DocsToKG.DocParsing as docparsing_pkg

        docparsing_pkg._MODULE_CACHE.pop("chunking", None)
        sys.modules.pop("DocsToKG.DocParsing.chunking", None)
        docparsing_pkg.__dict__.pop("chunking", None)

        chunk_module = builtins.__import__("DocsToKG.DocParsing.chunking", fromlist=("chunking",))
        docparsing_pkg._MODULE_CACHE["chunking"] = chunk_module
        docparsing_pkg.__dict__["chunking"] = chunk_module
    except ImportError as exc:
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
        print(friendly_message, file=sys.stderr)
        print(
            "Optional DocTags/chunking dependencies are required for `docparse chunk`. "
            "Install them with `pip install DocsToKG[gpu12x]` or `pip install transformers`.",
            file=sys.stderr,
        )
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


def _manifest_main(argv: Sequence[str]) -> int:
    """Implementation for the ``docparse manifest`` command."""

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

    tail_count = max(0, int(args.tail))
    need_summary = bool(args.summarize or not tail_count)
    tail_entries: Deque[Dict[str, Any]] = deque(maxlen=tail_count or None)
    status_counter: Optional[Dict[str, Counter]] = None
    duration_totals: Optional[Dict[str, float]] = None
    total_entries: Optional[Dict[str, int]] = None
    if need_summary:
        status_counter = defaultdict(Counter)
        duration_totals = defaultdict(float)
        total_entries = defaultdict(int)

    entry_found = False
    for entry in iter_manifest_entries(stages, args.data_root):
        entry_found = True
        if tail_count:
            tail_entries.append(entry)
        if need_summary and total_entries is not None and status_counter is not None and duration_totals is not None:
            stage = entry.get("stage", "unknown")
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

    if need_summary and total_entries is not None and status_counter is not None and duration_totals is not None:
        summary = {
            stage: {
                "total": total_entries[stage],
                "statuses": dict(status_counter[stage]),
                "duration_s": round(duration_totals[stage], 3),
            }
            for stage in total_entries
        }
        print("\nManifest summary")
        for stage in sorted(summary):
            data = summary[stage]
            print(f"- {stage}: total={data['total']} duration_s={data['duration_s']}")
            status_map = data.get("statuses", {})
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
        embed_args.extend(
            ["--sparsity-warn-threshold-pct", str(args.sparsity_warn_threshold_pct)]
        )

    return doctags_args, chunk_args, embed_args


def run_all(argv: Sequence[str] | None = None) -> int:
    """Execute DocTags conversion, chunking, and embedding sequentially."""

    parser = argparse.ArgumentParser(
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

    args = parser.parse_args([] if argv is None else list(argv))
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
    logger.info("docparse all starting", extra={"extra_fields": extra})

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


def _command(handler: CommandHandler, help_text: str) -> "_Command":
    """Package a handler and help text into a command descriptor."""

    return _Command(handler, help_text)


class _Command:
    """Callable wrapper storing handler metadata for subcommands."""

    __slots__ = ("handler", "help")

    def __init__(self, handler: CommandHandler, help: str) -> None:
        """Store the callable ``handler`` and help text for the command."""

        self.handler = handler
        self.help = help


COMMANDS: Dict[str, _Command] = {
    "all": _command(run_all, "Run doctags → chunk → embed sequentially"),
    "chunk": _command(chunk, "Run the Docling hybrid chunker"),
    "embed": _command(embed, "Generate BM25/SPLADE/dense vectors"),
    "doctags": _command(doctags, "Convert HTML/PDF corpora into DocTags"),
    "token-profiles": _command(
        token_profiles, "Print token count ratios for DocTags samples across tokenizers"
    ),
    "plan": _command(
        plan, "Show the projected doctags → chunk → embed actions without running stages"
    ),
    "manifest": _command(manifest, "Inspect manifest artifacts (tail/summarize)"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to one of the DocParsing subcommands."""

    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS.keys(), help="CLI to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the command")
    parsed = parser.parse_args(argv)
    command = COMMANDS[parsed.command]
    return command.handler(parsed.args)
