# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.DocParsing.pipelines",
#   "purpose": "Implements DocsToKG.DocParsing.pipelines behaviors and helpers",
#   "sections": [
#     {
#       "id": "_looks_like_filesystem_path",
#       "name": "_looks_like_filesystem_path",
#       "anchor": "LLFP",
#       "kind": "function"
#     },
#     {
#       "id": "_expand_path",
#       "name": "_expand_path",
#       "anchor": "EP",
#       "kind": "function"
#     },
#     {
#       "id": "resolve_hf_home",
#       "name": "resolve_hf_home",
#       "anchor": "RHH",
#       "kind": "function"
#     },
#     {
#       "id": "resolve_model_root",
#       "name": "resolve_model_root",
#       "anchor": "RMR",
#       "kind": "function"
#     },
#     {
#       "id": "resolve_pdf_model_path",
#       "name": "resolve_pdf_model_path",
#       "anchor": "RPMP",
#       "kind": "function"
#     },
#     {
#       "id": "add_data_root_option",
#       "name": "add_data_root_option",
#       "anchor": "ADRO",
#       "kind": "function"
#     },
#     {
#       "id": "add_resume_force_options",
#       "name": "add_resume_force_options",
#       "anchor": "ARFO",
#       "kind": "function"
#     },
#     {
#       "id": "prepare_data_root",
#       "name": "prepare_data_root",
#       "anchor": "PDR",
#       "kind": "function"
#     },
#     {
#       "id": "resolve_pipeline_path",
#       "name": "resolve_pipeline_path",
#       "anchor": "RPP",
#       "kind": "function"
#     },
#     {
#       "id": "_dedupe_preserve_order",
#       "name": "_dedupe_preserve_order",
#       "anchor": "DPO",
#       "kind": "function"
#     },
#     {
#       "id": "_normalize_served_model_names",
#       "name": "_normalize_served_model_names",
#       "anchor": "NSMN",
#       "kind": "function"
#     },
#     {
#       "id": "detect_vllm_version",
#       "name": "detect_vllm_version",
#       "anchor": "DVV",
#       "kind": "function"
#     },
#     {
#       "id": "validate_served_models",
#       "name": "validate_served_models",
#       "anchor": "VSM",
#       "kind": "function"
#     },
#     {
#       "id": "pdf_build_parser",
#       "name": "pdf_build_parser",
#       "anchor": "PBP",
#       "kind": "function"
#     },
#     {
#       "id": "pdf_parse_args",
#       "name": "pdf_parse_args",
#       "anchor": "PPA",
#       "kind": "function"
#     },
#     {
#       "id": "pdf_task",
#       "name": "PdfTask",
#       "anchor": "PDFT",
#       "kind": "class"
#     },
#     {
#       "id": "pdf_conversion_result",
#       "name": "PdfConversionResult",
#       "anchor": "PDFC",
#       "kind": "class"
#     },
#     {
#       "id": "_normalize_status",
#       "name": "_normalize_status",
#       "anchor": "NS",
#       "kind": "function"
#     },
#     {
#       "id": "_safe_float",
#       "name": "_safe_float",
#       "anchor": "SF",
#       "kind": "function"
#     },
#     {
#       "id": "normalize_conversion_result",
#       "name": "normalize_conversion_result",
#       "anchor": "NCR",
#       "kind": "function"
#     },
#     {
#       "id": "port_is_free",
#       "name": "port_is_free",
#       "anchor": "PIF",
#       "kind": "function"
#     },
#     {
#       "id": "probe_models",
#       "name": "probe_models",
#       "anchor": "PM",
#       "kind": "function"
#     },
#     {
#       "id": "probe_metrics",
#       "name": "probe_metrics",
#       "anchor": "PM1",
#       "kind": "function"
#     },
#     {
#       "id": "stream_logs",
#       "name": "stream_logs",
#       "anchor": "SL",
#       "kind": "function"
#     },
#     {
#       "id": "start_vllm",
#       "name": "start_vllm",
#       "anchor": "SV",
#       "kind": "function"
#     },
#     {
#       "id": "wait_for_vllm",
#       "name": "wait_for_vllm",
#       "anchor": "WFV",
#       "kind": "function"
#     },
#     {
#       "id": "stop_vllm",
#       "name": "stop_vllm",
#       "anchor": "SV1",
#       "kind": "function"
#     },
#     {
#       "id": "ensure_vllm",
#       "name": "ensure_vllm",
#       "anchor": "EV",
#       "kind": "function"
#     },
#     {
#       "id": "list_pdfs",
#       "name": "list_pdfs",
#       "anchor": "LP",
#       "kind": "function"
#     },
#     {
#       "id": "pdf_convert_one",
#       "name": "pdf_convert_one",
#       "anchor": "PCO",
#       "kind": "function"
#     },
#     {
#       "id": "pdf_main",
#       "name": "pdf_main",
#       "anchor": "PM2",
#       "kind": "function"
#     },
#     {
#       "id": "html_build_parser",
#       "name": "html_build_parser",
#       "anchor": "HBP",
#       "kind": "function"
#     },
#     {
#       "id": "html_parse_args",
#       "name": "html_parse_args",
#       "anchor": "HPA",
#       "kind": "function"
#     },
#     {
#       "id": "html_task",
#       "name": "HtmlTask",
#       "anchor": "HTML",
#       "kind": "class"
#     },
#     {
#       "id": "html_conversion_result",
#       "name": "HtmlConversionResult",
#       "anchor": "HTML1",
#       "kind": "class"
#     },
#     {
#       "id": "_get_converter",
#       "name": "_get_converter",
#       "anchor": "GC",
#       "kind": "function"
#     },
#     {
#       "id": "list_htmls",
#       "name": "list_htmls",
#       "anchor": "LH",
#       "kind": "function"
#     },
#     {
#       "id": "html_convert_one",
#       "name": "html_convert_one",
#       "anchor": "HCO",
#       "kind": "function"
#     },
#     {
#       "id": "html_main",
#       "name": "html_main",
#       "anchor": "HM",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""DocParsing Pipeline Utilities

This module hosts the PDF → DocTags conversion workflow _and_ shared helpers
used by other DocParsing pipelines. It coordinates vLLM server lifecycle,
manifest bookkeeping, and CLI argument scaffolding so chunking and embedding
components can import consistent behaviours.

Key Features:
- Shared CLI helpers (`add_data_root_option`, `add_resume_force_options`,
  `prepare_data_root`, `resolve_pipeline_path`) to centralise directory and
  resume/force handling.
- PDF conversion pipeline that spins up a vLLM inference server, distributes
  work across processes, and writes DocTags with manifest telemetry.
- Utility routines for manifest updates, GPU resource configuration, and
  polite rate control against vLLM endpoints.

Usage:
    from DocsToKG.DocParsing import pipelines

    parser = pipelines.pdf_build_parser()
    args = parser.parse_args(["--data-root", "/datasets/Data"])
    exit_code = pipelines.pdf_main(args)
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess as sp
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

from DocsToKG.DocParsing._common import (
    acquire_lock,
    compute_content_hash,
    data_doctags,
    data_manifests,
    data_pdfs,
    detect_data_root,
    expand_path,
    find_free_port,
    get_logger,
    load_manifest_index,
    manifest_append,
    resolve_hash_algorithm,
    set_spawn_or_warn,
)

try:  # pragma: no cover - optional dependency
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - guard for stripped-down runtime
    InvalidVersion = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]

_LOGGER = get_logger(__name__)


# -------- Model path resolution helpers --------

PDF_MODEL_SUBDIR = Path("granite-docling-258M")


def _looks_like_filesystem_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` appears to reference a local path."""

    expanded = Path(candidate).expanduser()
    drive, _ = os.path.splitdrive(candidate)
    if drive:
        return True
    if expanded.is_absolute() or expanded.exists():
        return True
    prefixes = ["~", "."]
    if os.sep not in prefixes:
        prefixes.append(os.sep)
    alt = os.altsep
    if alt and alt not in prefixes:
        prefixes.append(alt)
    return any(candidate.startswith(prefix) for prefix in prefixes)


def _expand_path(path: str | Path) -> Path:
    """Expand a filesystem path to an absolute :class:`Path`."""

    return Path(path).expanduser().resolve()


def resolve_hf_home() -> Path:
    """Resolve the HuggingFace cache directory respecting ``HF_HOME``.

    Args:
        None

    Returns:
        Path: Absolute location of the HuggingFace cache directory.
    """

    env = os.getenv("HF_HOME")
    if env:
        return _expand_path(env)
    return Path.home().expanduser() / ".cache" / "huggingface"


def resolve_model_root() -> Path:
    """Resolve DocsToKG model root with environment override.

    Args:
        None

    Returns:
        Path: Absolute model root directory for DocsToKG artifacts.
    """

    env = os.getenv("DOCSTOKG_MODEL_ROOT")
    if env:
        return _expand_path(env)
    return resolve_hf_home()


def resolve_pdf_model_path(cli_value: str | None = None) -> str:
    """Determine PDF model path using CLI and environment precedence.

    Args:
        cli_value: Optional CLI supplied path or model identifier.

    Returns:
        str: Absolute filesystem path or HuggingFace model identifier to use.
    """

    if cli_value:
        if _looks_like_filesystem_path(cli_value):
            return str(expand_path(cli_value))
        return cli_value
    env_model = os.getenv("DOCLING_PDF_MODEL")
    if env_model:
        return str(expand_path(env_model))
    model_root = resolve_model_root()
    return str(expand_path(model_root / PDF_MODEL_SUBDIR))


# -------- Paths --------
DEFAULT_DATA_ROOT = detect_data_root()
DEFAULT_INPUT = data_pdfs(DEFAULT_DATA_ROOT)
DEFAULT_OUTPUT = data_doctags(DEFAULT_DATA_ROOT)
MANIFEST_STAGE = "doctags-pdf"


def add_data_root_option(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--data-root`` option to a CLI parser.

    Args:
        parser (argparse.ArgumentParser): Parser being configured.

    Returns:
        None

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> add_data_root_option(parser)
        >>> any(action.dest == "data_root" for action in parser._actions)
        True
    """

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or $DOCSTOKG_DATA_ROOT."
        ),
    )


def add_resume_force_options(
    parser: argparse.ArgumentParser,
    *,
    resume_help: str,
    force_help: str,
) -> None:
    """Attach ``--resume`` and ``--force`` switches to a CLI parser.

    Args:
        parser (argparse.ArgumentParser): Parser being configured.
        resume_help (str): Help text describing resume semantics.
        force_help (str): Help text describing force semantics.

    Returns:
        None

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> add_resume_force_options(
        ...     parser,
        ...     resume_help="Resume processing",
        ...     force_help="Force reprocessing",
        ... )
        >>> sorted(action.dest for action in parser._actions if action.option_strings)
        ['force', 'help', 'resume']
    """

    parser.add_argument("--resume", action="store_true", help=resume_help)
    parser.add_argument("--force", action="store_true", help=force_help)


def prepare_data_root(
    data_root_arg: Optional[Path],
    default_root: Path,
) -> Path:
    """Resolve and apply DocsToKG data-root settings for CLI pipelines.

    Args:
        data_root_arg (Path | None): CLI-supplied data-root override.
        default_root (Path): Default DocsToKG data directory.

    Returns:
        Path: Resolved data root that downstream stages should use.

    Side Effects:
        - When ``data_root_arg`` is provided, ``DOCSTOKG_DATA_ROOT`` is updated.
        - Ensures the manifests directory exists for downstream writes.

    Examples:
        >>> root = prepare_data_root(None, Path("/tmp/data"))
        >>> root.as_posix().endswith("/tmp/data")
        True
    """

    resolved = detect_data_root(data_root_arg) if data_root_arg is not None else default_root
    if data_root_arg is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved)
    data_manifests(resolved)
    return resolved


def resolve_pipeline_path(
    *,
    cli_value: Optional[Path],
    default_path: Path,
    resolved_data_root: Path,
    data_root_overridden: bool,
    resolver: Callable[[Path], Path],
) -> Path:
    """Derive a pipeline directory path respecting data-root overrides.

    Args:
        cli_value (Path | None): Path provided via CLI argument (may be ``None``).
        default_path (Path): Default path baked into the pipeline module.
        resolved_data_root (Path): Effective data root for the current invocation.
        data_root_overridden (bool): ``True`` when the CLI supplied ``--data-root``.
        resolver (Callable[[Path], Path]): Callable that derives the directory when
            a new data root is supplied (for example :func:`data_doctags`).

    Returns:
        Path: Directory path the pipeline should operate on. Callers may resolve the
        path to an absolute location if required.

    Examples:
        >>> resolve_pipeline_path(
        ...     cli_value=None,
        ...     default_path=Path("/tmp/data/DocTagsFiles"),
        ...     resolved_data_root=Path("/tmp/data"),
        ...     data_root_overridden=False,
        ...     resolver=lambda root: root / "DocTagsFiles",
        ... ).as_posix()
        '/tmp/data/DocTagsFiles'
    """

    if data_root_overridden and (cli_value is None or cli_value == default_path):
        return resolver(resolved_data_root)
    if cli_value is None:
        return default_path
    return cli_value


def _dedupe_preserve_order(names: Iterable[str]) -> List[str]:
    """Return a list containing ``names`` without duplicates while preserving order.

    Args:
        names: Iterable of candidate names that may include duplicates or empty values.

    Returns:
        List of unique names in their original encounter order.

    Examples:
        >>> _dedupe_preserve_order(["a", "b", "a", "c"])
        ['a', 'b', 'c']
    """

    seen = set()
    unique: List[str] = []
    for name in names:
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _normalize_served_model_names(raw: Optional[Iterable[Iterable[str] | str]]) -> Tuple[str, ...]:
    """Flatten CLI-provided served model names into a deduplicated tuple.

    Args:
        raw: Sequence containing strings or nested iterables of strings sourced from
            CLI options.

    Returns:
        Tuple of unique served model names with defaults applied when none were given.

    Examples:
        >>> _normalize_served_model_names([["a", "b"], "b"])
        ('a', 'b')
    """

    if not raw:
        return DEFAULT_SERVED_MODEL_NAMES

    flattened: List[str] = []
    for entry in raw:
        if isinstance(entry, str):
            flattened.append(entry)
        else:
            flattened.extend(str(item) for item in entry)
    return tuple(_dedupe_preserve_order(flattened)) or DEFAULT_SERVED_MODEL_NAMES


def detect_vllm_version() -> str:
    """Detect the installed vLLM package version for diagnostics.

    Args:
        None

    Returns:
        Version string reported by the local vLLM installation or ``"unknown"`` when
        the package cannot be imported.
    """

    try:  # pragma: no cover - requires optional dependency
        import vllm  # type: ignore
    except Exception as exc:  # pragma: no cover - logged for operator visibility
        _LOGGER.warning(
            "Unable to import vLLM for version detection",
            extra={"extra_fields": {"error": str(exc)}},
        )
        return "unknown"

    version = getattr(vllm, "__version__", "unknown")
    _LOGGER.info(
        "Detected vLLM package",
        extra={"extra_fields": {"version": version}},
    )
    if Version is not None and version not in {"unknown", ""}:
        try:
            if Version(version) < Version("0.3.0"):
                _LOGGER.warning(
                    "vLLM version %s is below the supported minimum of 0.3.0",
                    version,
                )
        except InvalidVersion:  # pragma: no cover - defensive parsing guard
            _LOGGER.debug(
                "Could not parse vLLM version string", extra={"extra_fields": {"version": version}}
            )
    return version


def validate_served_models(available: Optional[List[str]], expected: Tuple[str, ...]) -> None:
    """Ensure that at least one of the expected served model names is available.

    Args:
        available: Model aliases exposed by the running vLLM server.
        expected: Tuple of acceptable model aliases configured for conversion.

    Returns:
        None

    Raises:
        RuntimeError: If none of the expected model names are present.
    """

    if not expected:
        return

    candidates = set(available or [])
    if any(name in candidates for name in expected):
        return

    raise RuntimeError(
        "Expected model not served",
        {
            "expected": list(expected),
            "available": list(candidates),
        },
    )


def pdf_build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the PDF → DocTags converter.

    Args:
        None: Parser construction does not require inputs.

    Returns:
        Argument parser configured with all supported CLI options.

    Raises:
        ValueError: If parser configuration fails due to invalid defaults.
    """

    parser = argparse.ArgumentParser(
        description="Convert PDF corpora to DocTags with an optional vLLM backend",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or $DOCSTOKG_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder with PDFs (recurses).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Folder for Doctags output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers for PDF conversion",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path or identifier for the vLLM model to serve. "
            "Defaults to DOCLING_PDF_MODEL, DOCSTOKG_MODEL_ROOT/"
            f"{PDF_MODEL_SUBDIR}, or HF_HOME/{PDF_MODEL_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--served-model-name",
        dest="served_model_names",
        action="append",
        nargs="+",
        default=None,
        help="Model name to expose via OpenAI compatibility API (repeatable)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory the vLLM server may allocate",
    )
    parser.add_argument(
        "--vlm-prompt",
        type=str,
        default="Convert this page to docling.",
        help="Prompt passed to the VLM for PDF pages",
    )
    parser.add_argument(
        "--vlm-stop",
        action="append",
        default=["</doctag>", "<|end_of_text|>"],
        help="Stop tokens for the VLM (repeatable)",
    )
    add_resume_force_options(
        parser,
        resume_help="Skip PDFs whose DocTags already exist with matching content hash",
        force_help="Force reprocessing even when resume criteria are satisfied",
    )
    return parser


def pdf_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone execution.

    Args:
        argv: Optional CLI argument list. When ``None`` the values from
            :data:`sys.argv` are used.

    Returns:
        Namespace containing parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` detects invalid arguments.
    """

    return pdf_build_parser().parse_args(argv)


DEFAULT_SERVED_MODEL_NAMES: Tuple[str, ...] = (
    "granite-docling-258M",
    "ibm-granite/granite-docling-258M",
)
DEFAULT_GPU_MEMORY_UTILIZATION = 0.30

# -------- Settings --------
PREFERRED_PORT = 8000
PORT_SCAN_SPAN = 32
DEFAULT_WORKERS = min(12, (os.cpu_count() or 16) - 4)
WAIT_TIMEOUT_S = 300

# Thread hygiene for CPU libs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("DOCLING_CUDA_USE_FLASH_ATTENTION2", "1")

ARTIFACTS = os.environ.get("DOCLING_ARTIFACTS_PATH", "")


# -------- Utilities --------
@dataclass
class PdfTask:
    """Work item representing a single PDF conversion request.

    Attributes:
        pdf_path: Absolute path to the PDF document to convert.
        output_dir: Destination directory where DocTags are stored.
        port: vLLM HTTP port used for remote inference.
        input_hash: Content hash representing the PDF for change detection.
        doc_id: Identifier derived from the PDF path for manifest entries.
        output_path: Final DocTags artifact location.
        served_model_names: Collection of aliases configured for the vLLM server.
        inference_model: Primary model name used when issuing chat completions.
        vlm_prompt: Prompt text passed to the VLM for PDF page conversion.
        vlm_stop: Stop tokens used to terminate VLM generation.

    Examples:
        >>> task = PdfTask(
        ...     Path("/tmp/sample.pdf"),
        ...     Path("/tmp/out"),
        ...     8000,
        ...     "hash",
        ...     "doc",
        ...     Path("/tmp/out/doc.doctags"),
        ...     ("granite-docling-258M",),
        ...     "granite-docling-258M",
        ...     "Convert this page to docling.",
        ...     ("</doctag>", "<|end_of_text|>"),
        ... )
        >>> task.doc_id
        'doc'
    """

    pdf_path: Path
    output_dir: Path
    port: int
    input_hash: str
    doc_id: str
    output_path: Path
    served_model_names: Tuple[str, ...]
    inference_model: str
    vlm_prompt: str
    vlm_stop: Tuple[str, ...]

    def __getitem__(self, index: int) -> Path:
        """Provide tuple-like access for compatibility with legacy tests.

        Args:
            index: Position requested by tuple-style accessors.

        Returns:
            PDF path when ``index`` is ``0``.

        Raises:
            IndexError: If ``index`` is not ``0``.
        """

        if index == 0:
            return self.pdf_path
        raise IndexError("PdfTask only supports index 0")


@dataclass
class PdfConversionResult:
    """Structured result returned by worker processes.

    Attributes:
        doc_id: Document identifier associated with the conversion.
        status: Outcome string such as ``"success"`` or ``"failure"``.
        duration_s: Worker runtime in seconds.
        input_path: Original PDF path recorded for manifest entries.
        input_hash: Content hash used to detect stale outputs.
        output_path: Location of the produced DocTags file.
        error: Optional error detail captured during conversion.

    Examples:
        >>> PdfConversionResult("doc", "success", 1.0, "in.pdf", "hash", "out.doctags")
        PdfConversionResult(doc_id='doc', status='success', duration_s=1.0, input_path='in.pdf', input_hash='hash', output_path='out.doctags', error=None)
    """

    doc_id: str
    status: str
    duration_s: float
    input_path: str
    input_hash: str
    output_path: str
    error: Optional[str] = None


def _normalize_status(raw: Optional[str]) -> str:
    """Coerce legacy status strings into the canonical vocabulary.

    Args:
        raw: Status string emitted by historical workers or manifests.

    Returns:
        Canonical status string (``"success"``, ``"skip"``, or ``"failure"``).
    """

    if not raw:
        return "success"

    normalized = raw.strip().lower()
    if normalized in {"ok", "success", "succeeded", "done"}:
        return "success"
    if normalized in {"skip", "skipped", "skipping"}:
        return "skip"
    if normalized in {"fail", "failed", "error", "exception"}:
        return "failure"
    return normalized


def _safe_float(value: Any) -> float:
    """Convert the supplied value to ``float`` when possible.

    Args:
        value: Object that may represent a numeric scalar.

    Returns:
        Floating point representation of ``value`` or ``0.0`` if conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def normalize_conversion_result(result: Any, task: Optional[PdfTask] = None) -> PdfConversionResult:
    """Adapt heterogeneous worker return values into :class:`PdfConversionResult`.

    Historically the converter returned a tuple ``(doc_id, status)``.  The
    refactor switched to ``PdfConversionResult`` which broke a regression test
    that still emits tuples.  This helper accepts both shapes—as well as dicts
    produced by ad-hoc stubs—and populates missing metadata from the associated
    :class:`PdfTask` when available.

    Args:
        result: Object returned by a worker invocation.
        task: Related :class:`PdfTask` used to back-fill metadata when needed.

    Returns:
        Normalised :class:`PdfConversionResult` instance encapsulating the
        conversion outcome.
    """

    if isinstance(result, PdfConversionResult):
        return result

    if isinstance(result, dict):
        payload: Dict[str, Any] = dict(result)
        doc_id = payload.get("doc_id") or (task.doc_id if task else "unknown")
        status = _normalize_status(payload.get("status"))
        return PdfConversionResult(
            doc_id=str(doc_id),
            status=status,
            duration_s=_safe_float(payload.get("duration_s", 0.0)),
            input_path=str(payload.get("input_path") or (task.pdf_path if task else "")),
            input_hash=str(payload.get("input_hash") or (task.input_hash if task else "")),
            output_path=str(payload.get("output_path") or (task.output_path if task else "")),
            error=payload.get("error"),
        )

    if isinstance(result, (tuple, list)):
        doc_id = (
            result[0]
            if len(result) > 0 and result[0] is not None
            else (task.doc_id if task else "unknown")
        )
        status = _normalize_status(result[1] if len(result) > 1 else None)
        duration = _safe_float(result[2] if len(result) > 2 else 0.0)
        input_path = (
            result[3]
            if len(result) > 3 and result[3] is not None
            else (task.pdf_path if task else "")
        )
        output_path = (
            result[4]
            if len(result) > 4 and result[4] is not None
            else (task.output_path if task else "")
        )
        input_hash = (
            result[5]
            if len(result) > 5 and result[5] is not None
            else (task.input_hash if task else "")
        )
        error = result[6] if len(result) > 6 else None

        return PdfConversionResult(
            doc_id=str(doc_id),
            status=status,
            duration_s=duration,
            input_path=str(input_path),
            input_hash=str(input_hash),
            output_path=str(output_path),
            error=None if error is None else str(error),
        )

    # Fallback for unexpected return types
    doc_id = task.doc_id if task else "unknown"
    return PdfConversionResult(
        doc_id=doc_id,
        status="failure",
        duration_s=0.0,
        input_path=str(task.pdf_path if task else ""),
        input_hash=str(task.input_hash if task else ""),
        output_path=str(task.output_path if task else ""),
        error=f"Unsupported result type: {type(result)!r}",
    )


def port_is_free(port: int) -> bool:
    """Determine whether a TCP port on localhost is currently available.

    Args:
        port: Port number to probe on the loopback interface.

    Returns:
        True when the port is unused; otherwise False.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) != 0


def probe_models(
    port: int, timeout=2.5
) -> Tuple[Optional[List[str]], Optional[str], Optional[int]]:
    """Inspect the `/v1/models` endpoint exposed by a vLLM HTTP server.

    Args:
        port: HTTP port where the vLLM server is expected to listen.
        timeout: Seconds to wait for the HTTP request before aborting.

    Returns:
        Tuple containing the list of model identifiers (if any), the raw response
        body, and the HTTP status code. Missing models or connection failures are
        represented by `(None, <error>, None)`.
    """
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        r = requests.get(url, timeout=timeout)
        raw = r.text
        if r.headers.get("content-type", "").startswith("application/json"):
            try:
                data = r.json()
            except Exception:
                data = None
        else:
            data = None
        names = []
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for m in data["data"]:
                mid = m.get("id") or m.get("name")
                if mid:
                    names.append(mid)
        return names if names else [], raw, r.status_code
    except Exception as e:
        return None, str(e), None


def probe_metrics(port: int, timeout=2.5) -> Tuple[bool, Optional[int]]:
    """Check whether the vLLM `/metrics` endpoint is healthy.

    Args:
        port: HTTP port where the vLLM server should expose metrics.
        timeout: Seconds to wait for the HTTP response before aborting.

    Returns:
        Tuple of `(is_healthy, status_code)` where `is_healthy` is True when the
        endpoint responds with HTTP 200.
    """
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        r = requests.get(url, timeout=timeout)
        return (r.status_code == 200), r.status_code
    except Exception:
        return (False, None)


def stream_logs(proc: sp.Popen, prefix: str = "[vLLM] ", tail: Optional[Deque[str]] = None):
    """Continuously stream stdout lines from a child process to the console.

    Args:
        proc: Running subprocess whose stdout should be tailed.
        prefix: Text prefix applied to each emitted log line for readability.
        tail: Optional deque that accumulates the most recent log lines.

    Returns:
        None: This routine streams output for side effects only.
    """
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        s = line.rstrip()
        if not s:
            continue
        if tail is not None:
            tail.append(s)
        _LOGGER.info(
            "vLLM stdout",
            extra={
                "extra_fields": {
                    "source": "vllm",
                    "line": prefix + s,
                }
            },
        )


def start_vllm(
    port: int,
    model_path: str,
    served_model_names: Tuple[str, ...],
    gpu_memory_utilization: float,
) -> sp.Popen:
    """Launch a vLLM server process on the requested port.

    Args:
        port: Port on which the vLLM HTTP server should listen.
        model_path: Local directory or HF repository containing model weights.
        served_model_names: Aliases registered with the OpenAI-compatible API.
        gpu_memory_utilization: Fraction of GPU memory the server may allocate.

    Returns:
        Started subprocess handle for the vLLM server.

    Raises:
        SystemExit: If the `vllm` executable is not present on `PATH`.
    """
    if shutil.which("vllm") is None:
        _LOGGER.error("'vllm' not found on PATH", extra={"extra_fields": {"cmd": "vllm"}})
        sys.exit(1)

    cmd = [
        "vllm",
        "serve",
        str(model_path),
        "--port",
        str(port),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
    ]
    for name in served_model_names:
        cmd.extend(["--served-model-name", name])

    env = os.environ.copy()
    env.setdefault("VLLM_LOG_LEVEL", "INFO")  # INFO so we can see useful lines
    _LOGGER.info(
        "Starting vLLM",
        extra={"extra_fields": {"command": cmd, "env_log_level": env.get("VLLM_LOG_LEVEL")}},
    )
    proc = sp.Popen(cmd, env=env, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
    tail: Deque[str] = deque(maxlen=50)
    setattr(proc, "_log_tail", tail)
    # Start log thread
    t = threading.Thread(target=stream_logs, args=(proc, "[vLLM] ", tail), daemon=True)
    t.start()
    return proc


def wait_for_vllm(port: int, proc: sp.Popen, timeout_s: int = WAIT_TIMEOUT_S) -> List[str]:
    """Poll the vLLM server until `/v1/models` responds with success.

    Args:
        port: HTTP port where the server is expected to listen.
        proc: Subprocess handle representing the running vLLM instance.
        timeout_s: Maximum time in seconds to wait for readiness.

    Returns:
        Model names reported by the server upon readiness.

    Raises:
        RuntimeError: If the server exits prematurely or fails to become ready
            within the allotted timeout.
    """
    _LOGGER.info(
        "Probing vLLM",
        extra={"extra_fields": {"port": port, "timeout_s": timeout_s}},
    )
    t0 = time.time()
    with tqdm(total=timeout_s, unit="s", desc="vLLM warmup", leave=True) as bar:
        while True:
            # If vLLM crashed, stop early and print the last lines
            if proc.poll() is not None:
                try:
                    tail_lines = list(getattr(proc, "_log_tail", []))
                    leftover = proc.stdout.read() or ""
                    combined_tail = tail_lines[-50:]
                    tail_text = "\n".join(combined_tail) if combined_tail else leftover[-800:]
                    _LOGGER.error(
                        "vLLM exited while waiting",
                        extra={
                            "extra_fields": {
                                "port": port,
                                "tail": tail_text,
                            }
                        },
                    )
                finally:
                    message = (
                        f"vLLM exited early with code {proc.returncode}. "
                        "Last log lines:\n"
                        f"{tail_text}\n"
                        "Common causes include missing model weights, incompatible CUDA drivers, "
                        "or insufficient GPU memory."
                    )
                    raise RuntimeError(message)
            names, raw, status = probe_models(port)
            if status == 200:
                _LOGGER.info(
                    "vLLM models available",
                    extra={
                        "extra_fields": {
                            "port": port,
                            "models": names,
                        }
                    },
                )
                return names or []
            # (…keep your existing logging, metrics probe, and tqdm update…)
            if int(time.time() - t0) >= timeout_s:
                raise RuntimeError(f"Timed out waiting for vLLM on port {port}")
            time.sleep(1)
            bar.update(1)


def stop_vllm(proc: Optional[sp.Popen], own: bool, grace=10):
    """Terminate a managed vLLM process if this script launched it.

    Args:
        proc: Subprocess handle returned by `start_vllm`, or None.
        own: Indicates whether the caller owns the process lifetime.
        grace: Seconds to wait for graceful shutdown before forcing exit.

    Returns:
        None.
    """
    if not own or proc is None or proc.poll() is not None:
        return
    _LOGGER.info(
        "Stopping vLLM",
        extra={"extra_fields": {"grace_seconds": grace}},
    )
    try:
        proc.terminate()
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < grace:
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass


def ensure_vllm(
    preferred: int,
    model_path: str,
    served_model_names: Tuple[str, ...],
    gpu_memory_utilization: float,
) -> Tuple[int, Optional[sp.Popen], bool]:
    """Ensure a vLLM server is available, launching one when necessary.

    Args:
        preferred: Preferred TCP port for the server.
        model_path: Model repository or path passed to the vLLM CLI.
        served_model_names: Aliases that should be exposed via the OpenAI API.
        gpu_memory_utilization: Fractional GPU memory reservation for the server.

    Returns:
        Tuple containing `(port, process, owns_process)` where `process` is the
        managed subprocess handle (or None if reusing an existing server) and
        `owns_process` indicates whether the caller should terminate it.

    Raises:
        RuntimeError: If the launched or reused server does not expose the expected
            model aliases, indicating a misconfiguration.
    """
    # 1) If preferred is free, start there
    if port_is_free(preferred):
        proc = start_vllm(preferred, model_path, served_model_names, gpu_memory_utilization)
        names = wait_for_vllm(preferred, proc)
        validate_served_models(names, served_model_names)
        return preferred, proc, True

    # 2) If something is already on preferred, reuse if it's vLLM (any models list)
    names, raw, status = probe_models(preferred)
    if status == 200:
        validate_served_models(names, served_model_names)
        _LOGGER.info(
            "Reusing vLLM",
            extra={
                "extra_fields": {
                    "port": preferred,
                    "models": names,
                }
            },
        )
        return preferred, None, False

    # 3) Otherwise, pick a new free port
    alt = find_free_port(preferred + 1, PORT_SCAN_SPAN)
    _LOGGER.info(
        "Launching vLLM on alternate port",
        extra={
            "extra_fields": {
                "preferred_port": preferred,
                "alternate_port": alt,
            }
        },
    )
    proc = start_vllm(alt, model_path, served_model_names, gpu_memory_utilization)
    names = wait_for_vllm(alt, proc)
    validate_served_models(names, served_model_names)
    return alt, proc, True


def list_pdfs(root: Path) -> List[Path]:
    """Collect PDF files under a directory recursively.

    Args:
        root: Directory whose subtree should be scanned for PDFs.

    Returns:
        Sorted list of paths to PDF files.
    """
    return sorted([p for p in root.rglob("*.pdf") if p.is_file()])


# -------- Docling worker --------
def pdf_convert_one(task: PdfTask) -> PdfConversionResult:
    """Convert a single PDF into DocTags using a remote vLLM-backed pipeline.

    Args:
        task: Description of the conversion request, including paths and port.

    Returns:
        Populated :class:`PdfConversionResult` reporting success, skip, or failure.
    """

    start = time.perf_counter()
    pdf_path = task.pdf_path
    out_dir = task.output_dir
    port = task.port
    out_path = task.output_path
    inference_model = task.inference_model

    try:
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

        # Imports (exact modules)
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions  # <-- correct module
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="skip",
                duration_s=0.0,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
            )

        # Accelerator (use CUDA; keep CPU thread count small per worker)
        accel = AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CUDA)

        # Remote VLM (OpenAI-compatible) -> vLLM
        api = ApiVlmOptions(
            url=f"http://127.0.0.1:{port}/v1/chat/completions",
            params=dict(
                # use the name you served via --served-model-name
                model=inference_model,
                max_tokens=4096,  # <-- IMPORTANT for vLLM
                skip_special_tokens=False,
                temperature=0.1,
                stop=list(task.vlm_stop),
            ),
            prompt=task.vlm_prompt,
            timeout=120,
            scale=2.0,  # image scale for the request
            response_format=ResponseFormat.DOCTAGS,
        )

        # VLM pipeline options (this is the only options object used for VlmPipeline)
        vlm_opts = VlmPipelineOptions(
            enable_remote_services=True,
            accelerator_options=accel,
            do_picture_description=True,
            do_picture_classification=True,
            # ensure Docling renders page bitmaps for the VLM:
            generate_page_images=True,
            images_scale=2.0,  # render at 2x for fidelity
        )
        vlm_opts.vlm_options = api

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_opts,  # <-- only this options object is used
                )
            }
        )

        result = converter.convert(pdf_path, raises_on_error=False)
        if result.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
            err_msgs = []
            for err in getattr(result, "errors", []) or []:
                msg = getattr(err, "error_message", None)
                if msg:
                    err_msgs.append(msg)
            detail = f"status={getattr(result.status, 'value', result.status)}"
            if err_msgs:
                detail += " " + "; ".join(err_msgs)
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error=detail,
            )

        if result.document is None:
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error="empty-document",
            )

        try:
            with acquire_lock(out_path):
                if out_path.exists():
                    return PdfConversionResult(
                        doc_id=task.doc_id,
                        status="skip",
                        duration_s=time.perf_counter() - start,
                        input_path=str(pdf_path),
                        input_hash=task.input_hash,
                        output_path=str(out_path),
                    )
                result.document.save_as_doctags(out_path)
        except TimeoutError as exc:
            return PdfConversionResult(
                doc_id=task.doc_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(pdf_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error=str(exc),
            )
        return PdfConversionResult(
            doc_id=task.doc_id,
            status="success",
            duration_s=time.perf_counter() - start,
            input_path=str(pdf_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
        )
    except Exception as exc:  # pragma: no cover - exercised during integration runs
        return PdfConversionResult(
            doc_id=task.doc_id,
            status="failure",
            duration_s=time.perf_counter() - start,
            input_path=str(pdf_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
            error=str(exc),
        )


# -------- Main --------
def pdf_main(args: argparse.Namespace | None = None) -> int:
    """Coordinate vLLM startup and parallel DocTags conversion.

    Args:
        args: Optional argument namespace injected during programmatic use.

    Returns:
        Process exit code, where ``0`` indicates success.

    Raises:
        RuntimeError: If vLLM fails to start, becomes unhealthy, or conversion
            retries exhaust without success.
        ValueError: If required configuration (such as auto-detected mode) is invalid.
    """

    logger = get_logger(__name__)
    set_spawn_or_warn(logger)
    import multiprocessing as mp

    logger.info(
        "Multiprocessing configuration",
        extra={
            "extra_fields": {
                "start_method": mp.get_start_method(allow_none=True),
                "cpu_count": os.cpu_count(),
            }
        },
    )

    if isinstance(args, argparse.Namespace):
        args = args
    else:
        args = pdf_parse_args() if args is None else pdf_parse_args(args)

    if not hasattr(args, "served_model_names"):
        args.served_model_names = None
    if not hasattr(args, "model"):
        args.model = None
    if not hasattr(args, "gpu_memory_utilization"):
        args.gpu_memory_utilization = DEFAULT_GPU_MEMORY_UTILIZATION
    if not hasattr(args, "workers"):
        args.workers = DEFAULT_WORKERS
    if not hasattr(args, "resume"):
        args.resume = False
    if not hasattr(args, "force"):
        args.force = False
    if not hasattr(args, "input"):
        args.input = DEFAULT_INPUT
    if not hasattr(args, "output"):
        args.output = DEFAULT_OUTPUT

    served_model_names = _normalize_served_model_names(args.served_model_names)
    inference_model = served_model_names[0]
    model_path = resolve_pdf_model_path(args.model)
    args.model = model_path
    gpu_memory_utilization = float(args.gpu_memory_utilization)

    vllm_version = detect_vllm_version()

    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else DEFAULT_DATA_ROOT
    )

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    data_manifests(resolved_root)

    if args.input == DEFAULT_INPUT and data_root_override is not None:
        input_dir = data_pdfs(resolved_root)
    else:
        input_dir = (args.input or DEFAULT_INPUT).resolve()

    if args.output == DEFAULT_OUTPUT and data_root_override is not None:
        output_dir = data_doctags(resolved_root)
    else:
        output_dir = (args.output or DEFAULT_OUTPUT).resolve()

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "I/O configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "artifacts_cache": ARTIFACTS or "",
                "model_path": model_path,
                "served_models": list(served_model_names),
                "gpu_memory_utilization": gpu_memory_utilization,
                "vllm_version": vllm_version,
            }
        },
    )

    if args.force:
        logger.info("Force mode: reprocessing all documents")
    elif args.resume:
        logger.info("Resume mode enabled: unchanged outputs will be skipped")

    preflight_start = time.perf_counter()
    port, proc, owns = ensure_vllm(
        PREFERRED_PORT,
        model_path,
        served_model_names,
        gpu_memory_utilization,
    )
    metrics_healthy, metrics_status = probe_metrics(port)
    manifest_append(
        stage=MANIFEST_STAGE,
        doc_id="__service__",
        status="success",
        duration_s=round(time.perf_counter() - preflight_start, 3),
        schema_version="docparse/1.1.0",
        served_models=list(served_model_names),
        vllm_version=vllm_version,
        port=port,
        owns_process=owns,
        metrics_healthy=metrics_healthy,
        metrics_status_code=metrics_status,
    )
    logger.info(
        "vLLM server ready",
        extra={
            "extra_fields": {
                "port": port,
                "owns_process": owns,
            }
        },
    )

    try:
        pdfs = list_pdfs(input_dir)
        if not pdfs:
            logger.warning(
                "No PDFs found",
                extra={"extra_fields": {"input_dir": str(input_dir)}},
            )
            return 0

        manifest_index = load_manifest_index(MANIFEST_STAGE, resolved_root) if args.resume else {}

        workers = max(1, int(args.workers))
        logger.info(
            "Launching workers",
            extra={
                "extra_fields": {
                    "pdf_count": len(pdfs),
                    "workers": workers,
                }
            },
        )

        tasks: List[PdfTask] = []
        ok = fail = skip = 0
        for pdf_path in pdfs:
            rel_path = pdf_path.relative_to(input_dir)
            doc_id = rel_path.as_posix()
            out_path = (output_dir / rel_path).with_suffix(".doctags")
            input_hash = compute_content_hash(pdf_path)
            manifest_entry = manifest_index.get(doc_id)
            if (
                args.resume
                and not args.force
                and out_path.exists()
                and manifest_entry
                and manifest_entry.get("input_hash") == input_hash
            ):
                logger.info(
                    "Skipping document: output exists and input unchanged",
                    extra={
                        "extra_fields": {
                            "doc_id": doc_id,
                            "output_path": str(out_path),
                        }
                    },
                )
                manifest_append(
                    stage=MANIFEST_STAGE,
                    doc_id=doc_id,
                    status="skip",
                    duration_s=0.0,
                    schema_version="docparse/1.1.0",
                    input_path=str(pdf_path),
                    input_hash=input_hash,
                    hash_alg=resolve_hash_algorithm(),
                    output_path=str(out_path),
                    parse_engine="docling-vlm",
                    model_name=inference_model,
                    served_models=list(served_model_names),
                    vllm_version=vllm_version,
                )
                skip += 1
                continue

            tasks.append(
                PdfTask(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    port=port,
                    input_hash=input_hash,
                    doc_id=doc_id,
                    output_path=out_path,
                    served_model_names=served_model_names,
                    inference_model=inference_model,
                    vlm_prompt=str(args.vlm_prompt),
                    vlm_stop=tuple(args.vlm_stop or []),
                )
            )

        if not tasks:
            logger.info(
                "Conversion summary",
                extra={
                    "extra_fields": {
                        "ok": 0,
                        "skip": skip,
                        "fail": 0,
                    }
                },
            )
            return 0

        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(pdf_convert_one, task): task for task in tasks}
            with tqdm(total=len(future_map), desc="Converting PDFs", unit="file") as pbar:
                for fut in as_completed(future_map):
                    task = future_map[fut]
                    raw_result = fut.result()
                    result = normalize_conversion_result(raw_result, task)
                    if result.status == "success":
                        ok += 1
                    elif result.status == "skip":
                        skip += 1
                    else:
                        fail += 1
                        logger.error(
                            "Conversion failed",
                            extra={
                                "extra_fields": {
                                    "doc_id": result.doc_id,
                                    "error": result.error or "unknown",
                                }
                            },
                        )

                    manifest_append(
                        stage=MANIFEST_STAGE,
                        doc_id=result.doc_id,
                        status=result.status,
                        duration_s=round(result.duration_s, 3),
                        schema_version="docparse/1.1.0",
                        input_path=result.input_path,
                        input_hash=result.input_hash,
                        hash_alg=resolve_hash_algorithm(),
                        output_path=result.output_path,
                        error=result.error,
                        parse_engine="docling-vlm",
                        model_name=task.inference_model,
                        served_models=list(task.served_model_names),
                        vllm_version=vllm_version,
                    )

                    pbar.update(1)

        logger.info(
            "Conversion summary",
            extra={
                "extra_fields": {
                    "ok": ok,
                    "skip": skip,
                    "fail": fail,
                }
            },
        )
    finally:
        stop_vllm(proc, owns, grace=10)
        logger.info("All done")

    return 0


if __name__ == "__main__":
    raise SystemExit(pdf_main())


# --------------------------------------------------------------------------- #
# HTML Pipeline
# --------------------------------------------------------------------------- #

"""
Parallel HTML → DocTags conversion pipeline.

Implements Docling HTML conversions across multiple processes while tracking
manifests, resume/force semantics, and advisory file locks. The pipeline is
used by the DocsToKG CLI to transform raw HTML corpora into DocTags ready for
chunking and embedding.
"""


from DocsToKG.DocParsing._common import (
    data_doctags,
    data_html,
    detect_data_root,
    get_logger,
)

HTML_DEFAULT_INPUT_DIR = data_html()
HTML_DEFAULT_OUTPUT_DIR = data_doctags()
HTML_MANIFEST_STAGE = "doctags-html"

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

_LOGGER = get_logger(__name__)

# keep numeric libs polite; also ensure nothing touches CUDA by mistake
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only

# per-process converter cache
_CONVERTER = None


def html_build_parser() -> argparse.ArgumentParser:
    """Construct an argument parser for the HTML → DocTags converter.

    Args:
        None: Parser initialization does not require inputs.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.

    Raises:
        None
    """

    parser = argparse.ArgumentParser(
        description="Convert HTML corpora to DocTags using Docling",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Override DocsToKG Data directory. Defaults to auto-detection or $DOCSTOKG_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=HTML_DEFAULT_INPUT_DIR,
        help="Folder with HTML files (recurses)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HTML_DEFAULT_OUTPUT_DIR,
        help="Destination for .doctags",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 1),
        help="Parallel workers",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .doctags files"
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
    return parser


def html_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional CLI argument vector. When ``None`` the values from
            :data:`sys.argv` are used.

    Returns:
        Namespace containing parsed CLI options.

    Raises:
        SystemExit: Propagated if ``argparse`` detects invalid options.
    """

    return html_build_parser().parse_args(argv)


@dataclass
class HtmlTask:
    """Work item describing a single HTML conversion job.

    Attributes:
        html_path: Absolute path to the HTML file to be converted.
        relative_id: Relative identifier for manifest entries.
        output_path: Destination DocTags path.
        input_hash: Content hash used for resume detection.
        overwrite: Flag indicating whether existing outputs should be replaced.

    Examples:
        >>> HtmlTask(Path("/tmp/a.html"), "doc", Path("/tmp/doc.doctags"), "hash", False)
        HtmlTask(html_path=PosixPath('/tmp/a.html'), relative_id='doc', output_path=PosixPath('/tmp/doc.doctags'), input_hash='hash', overwrite=False)
    """

    html_path: Path
    relative_id: str
    output_path: Path
    input_hash: str
    overwrite: bool


@dataclass
class HtmlConversionResult:
    """Structured result emitted by worker processes.

    Attributes:
        doc_id: Document identifier matching manifest entries.
        status: Conversion outcome (``"success"``, ``"skip"``, or ``"failure"``).
        duration_s: Time in seconds spent converting.
        input_path: Source HTML path recorded for auditing.
        input_hash: Content hash captured prior to conversion.
        output_path: Destination DocTags path.
        error: Optional error detail for failures.

    Examples:
        >>> HtmlConversionResult("doc", "success", 1.0, "in.html", "hash", "out.doctags")
        HtmlConversionResult(doc_id='doc', status='success', duration_s=1.0, input_path='in.html', input_hash='hash', output_path='out.doctags', error=None)
    """

    doc_id: str
    status: str
    duration_s: float
    input_path: str
    input_hash: str
    output_path: str
    error: str | None = None


def _get_converter() -> "DocumentConverter":
    """Instantiate and cache a Docling HTML converter per worker process.

    Returns:
        DocumentConverter configured for HTML input, cached for reuse within
        the worker process.
    """
    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, HTMLFormatOption

    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = DocumentConverter(
            format_options={InputFormat.HTML: HTMLFormatOption(backend=HTMLDocumentBackend)}
        )
    return _CONVERTER


def list_htmls(root: Path) -> List[Path]:
    """Enumerate HTML-like files beneath a directory tree.

    Args:
        root: Directory whose subtree should be searched for HTML files.

    Returns:
        Sorted list of discovered HTML file paths excluding normalized outputs.
    """
    exts = {".html", ".htm", ".xhtml"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.endswith(".normalized.html"):
            out.append(p)
    return sorted(out)


def html_convert_one(task: HtmlTask) -> HtmlConversionResult:
    """Convert a single HTML file to DocTags, honoring overwrite semantics.

    Args:
        task: Conversion details including paths, hash, and overwrite policy.

    Returns:
        :class:`ConversionResult` capturing the conversion status.

    Raises:
        ValueError: Propagated when Docling validation fails prior to internal handling.
    """

    start = time.perf_counter()
    try:
        out_path = task.output_path
        if out_path.exists() and not task.overwrite:
            return HtmlConversionResult(
                doc_id=task.relative_id,
                status="skip",
                duration_s=0.0,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
            )

        converter = _get_converter()
        result = converter.convert(task.html_path, raises_on_error=False)

        if result.document is None:
            return HtmlConversionResult(
                doc_id=task.relative_id,
                status="failure",
                duration_s=time.perf_counter() - start,
                input_path=str(task.html_path),
                input_hash=task.input_hash,
                output_path=str(out_path),
                error="empty-document",
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.document.save_as_doctags(out_path)
        return HtmlConversionResult(
            doc_id=task.relative_id,
            status="success",
            duration_s=time.perf_counter() - start,
            input_path=str(task.html_path),
            input_hash=task.input_hash,
            output_path=str(out_path),
        )

    except Exception as exc:  # pragma: no cover - integration failure path
        return HtmlConversionResult(
            doc_id=task.relative_id,
            status="failure",
            duration_s=time.perf_counter() - start,
            input_path=str(task.html_path),
            input_hash=task.input_hash,
            output_path=str(task.output_path),
            error=str(exc),
        )


def html_main(args: argparse.Namespace | None = None) -> int:
    """Entrypoint for parallel HTML-to-DocTags conversion across a dataset.

    Args:
        args: Optional pre-parsed CLI namespace to override command-line inputs.

    Returns:
        Process exit code, where ``0`` denotes success.
    """

    import multiprocessing as mp

    set_spawn_or_warn(_LOGGER)
    _LOGGER.info(
        "Multiprocessing configuration",
        extra={
            "extra_fields": {
                "start_method": mp.get_start_method(allow_none=True),
                "cpu_count": os.cpu_count(),
            }
        },
    )

    if isinstance(args, argparse.Namespace):
        args = args
    else:
        args = html_parse_args() if args is None else html_parse_args(args)

    if not hasattr(args, "input"):
        args.input = HTML_DEFAULT_INPUT_DIR
    if not hasattr(args, "output"):
        args.output = HTML_DEFAULT_OUTPUT_DIR
    if not hasattr(args, "workers"):
        args.workers = max(1, (os.cpu_count() or 8) - 1)
    if not hasattr(args, "overwrite"):
        args.overwrite = False
    if not hasattr(args, "resume"):
        args.resume = False
    if not hasattr(args, "force"):
        args.force = False
    if not hasattr(args, "data_root"):
        args.data_root = None

    data_root_override = args.data_root
    resolved_root = (
        detect_data_root(data_root_override)
        if data_root_override is not None
        else detect_data_root()
    )

    if data_root_override is not None:
        os.environ["DOCSTOKG_DATA_ROOT"] = str(resolved_root)

    data_manifests(resolved_root)

    if args.input == HTML_DEFAULT_INPUT_DIR and data_root_override is not None:
        input_dir: Path = data_html(resolved_root)
    else:
        input_dir = (args.input or HTML_DEFAULT_INPUT_DIR).resolve()

    if args.output == HTML_DEFAULT_OUTPUT_DIR and data_root_override is not None:
        output_dir: Path = data_doctags(resolved_root)
    else:
        output_dir = (args.output or HTML_DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info(
        "HTML conversion configuration",
        extra={
            "extra_fields": {
                "data_root": str(resolved_root),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "workers": args.workers,
            }
        },
    )

    if args.force:
        _LOGGER.info(
            "Force mode: reprocessing all documents",
            extra={"extra_fields": {"mode": "force"}},
        )
    elif args.resume:
        _LOGGER.info(
            "Resume mode enabled: unchanged outputs will be skipped",
            extra={"extra_fields": {"mode": "resume"}},
        )

    files = list_htmls(input_dir)
    if not files:
        _LOGGER.warning(
            "No HTML files found", extra={"extra_fields": {"input_dir": str(input_dir)}}
        )
        return 0

    manifest_index = load_manifest_index(HTML_MANIFEST_STAGE, resolved_root) if args.resume else {}

    tasks: List[HtmlTask] = []
    ok = fail = skip = 0
    for path in files:
        rel_path = path.relative_to(input_dir)
        doc_id = rel_path.as_posix()
        out_path = (output_dir / rel_path).with_suffix(".doctags")
        input_hash = compute_content_hash(path)
        manifest_entry = manifest_index.get(doc_id)
        if (
            args.resume
            and not args.force
            and not args.overwrite
            and out_path.exists()
            and manifest_entry
            and manifest_entry.get("input_hash") == input_hash
        ):
            _LOGGER.info(
                "Skipping HTML document",
                extra={
                    "extra_fields": {
                        "doc_id": doc_id,
                        "output_path": str(out_path),
                    }
                },
            )
            manifest_append(
                stage=HTML_MANIFEST_STAGE,
                doc_id=doc_id,
                status="skip",
                duration_s=0.0,
                schema_version="docparse/1.1.0",
                input_path=str(path),
                input_hash=input_hash,
                hash_alg=resolve_hash_algorithm(),
                output_path=str(out_path),
                parse_engine="docling-html",
            )
            skip += 1
            continue
        tasks.append(
            HtmlTask(
                html_path=path,
                relative_id=doc_id,
                output_path=out_path,
                input_hash=input_hash,
                overwrite=args.overwrite,
            )
        )

    if not tasks:
        _LOGGER.info(
            "HTML conversion summary",
            extra={
                "extra_fields": {
                    "ok": 0,
                    "skip": skip,
                    "fail": 0,
                }
            },
        )
        return 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(html_convert_one, task) for task in tasks]
        for fut in tqdm(
            as_completed(futures), total=len(futures), unit="file", desc="HTML → DocTags"
        ):
            result = fut.result()
            if result.status == "success":
                ok += 1
            elif result.status == "skip":
                skip += 1
            else:
                fail += 1
                _LOGGER.error(
                    "HTML conversion failure",
                    extra={
                        "extra_fields": {
                            "doc_id": result.doc_id,
                            "error": result.error or "conversion failed",
                        }
                    },
                )

            manifest_append(
                stage=HTML_MANIFEST_STAGE,
                doc_id=result.doc_id,
                status=result.status,
                duration_s=round(result.duration_s, 3),
                schema_version="docparse/1.1.0",
                input_path=result.input_path,
                input_hash=result.input_hash,
                hash_alg=resolve_hash_algorithm(),
                output_path=result.output_path,
                error=result.error,
                parse_engine="docling-html",
            )

    _LOGGER.info(
        "HTML conversion summary",
        extra={
            "extra_fields": {
                "ok": ok,
                "skip": skip,
                "fail": fail,
            }
        },
    )

    return 0


if __name__ == "__main__":
    import sys as _sys

    argv = _sys.argv[1:]
    if argv and argv[0] in {"--html", "--pipeline=html"}:
        if argv[0] == "--html":
            _sys.argv = [_sys.argv[0]] + argv[1:]
        else:  # --pipeline=html
            _sys.argv = [_sys.argv[0]] + [arg for arg in argv if not arg.startswith("--pipeline=")]
        raise SystemExit(html_main())

    if argv and argv[0] in {"--pdf", "--pipeline=pdf"}:
        if argv[0] == "--pdf":
            _sys.argv = [_sys.argv[0]] + argv[1:]
        else:  # --pipeline=pdf
            _sys.argv = [_sys.argv[0]] + [arg for arg in argv if not arg.startswith("--pipeline=")]
        raise SystemExit(pdf_main())

    raise SystemExit(pdf_main())
